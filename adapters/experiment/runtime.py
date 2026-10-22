"""Per-attempt HTTP trace and Linux execution without changing tool semantics."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import types
import httpx
from openai import OpenAI
from common import append, now, MODEL, RESPONSE_MODEL

HTTP_TIMEOUT_SECONDS = 900.0

class ConcurrencyObserver:
    """Count activity without limiting requests or coupling retry cooldowns."""
    def __init__(self):
        self.lock = threading.Lock(); self.active = 0; self.peak = 0

    def enter(self):
        with self.lock:
            self.active += 1; self.peak = max(self.peak, self.active)
            return self.active

    def leave(self):
        with self.lock: self.active -= 1

    def rate_limited(self):
        # SDK/native retry applies to the calling QA only. No shared cooldown.
        pass

class ServiceGate:
    def __init__(self, concurrency=6, interval=0.2):
        self.slots = threading.BoundedSemaphore(concurrency)
        self.lock = threading.Lock(); self.interval = interval; self.next_send = 0
        self.active = 0; self.peak = 0

    def enter(self):
        self.slots.acquire()
        while True:
            with self.lock:
                delay = self.next_send-time.monotonic()
                if delay <= 0:
                    self.next_send = time.monotonic()+self.interval
                    self.active += 1; self.peak = max(self.peak, self.active)
                    return self.active
            time.sleep(min(delay, 0.25))

    def leave(self):
        with self.lock: self.active -= 1
        self.slots.release()

    def rate_limited(self):
        with self.lock:
            self.next_send = max(self.next_send, time.monotonic()+5)
            self.interval = min(1.0, self.interval*1.5)

class Transport(httpx.BaseTransport):
    def __init__(self, directory, credentials, stage, gate):
        self.directory = Path(directory); self.key = credentials['api_key']
        self.stage = stage; self.gate = gate; self.counter = 0
        self.inner = httpx.HTTPTransport(retries=0)
        self.first_request = None; self.last_final = None; self.last_payload = None
        self.last_response = None; self.events = []; self.protocol_errors = []

    def record(self, event):
        # Body-only logging. Authorization, cookies and all headers are excluded.
        clean = json.loads(json.dumps(event, ensure_ascii=False).replace(self.key, '[REDACTED]'))
        append(self.directory/'http.jsonl', clean)
        # Keep usage metadata, not repeated histories, in memory.
        self.events.append({k:v for k,v in clean.items() if k not in {'request','response'}})

    def handle_request(self, request):
        body = json.loads(request.content)
        forbidden = {'reasoning', 'reasoning_effort', 'thinking'}
        if forbidden.intersection(body):
            raise ValueError('Unexpected reasoning control in actual HTTP body')
        if body.get('model') != MODEL:
            raise ValueError('Unexpected requested model')
        expected = {'qa':(32000,0.0), 'structure':(32768,0.0), 'judge':(None,0.0)}[self.stage]
        if body.get('max_tokens') != expected[0] or body.get('temperature') != expected[1]:
            raise ValueError('Unexpected output budget/temperature')
        if 'max_completion_tokens' in body or 'max_output_tokens' in body:
            raise ValueError('Unexpected alternate output budget')
        active = self.gate.enter(); self.counter += 1
        started = time.monotonic()
        if self.first_request is None: self.first_request = started
        ident = f'{os.getpid()}-{id(self)}-{self.counter}'
        common = dict(stage=self.stage, http_id=ident,
                      sdk_retry=request.headers.get('x-stainless-retry-count'),
                      timestamp=now(), monotonic_seconds=started)
        try:
            self.record(dict(common,event='request',request=body,service_active=active))
            response = self.inner.handle_request(request)
            response.read()
            ended = time.monotonic()
            try: payload = response.json()
            except Exception: payload = {'unparsed_body':response.text}
            self.last_payload = payload; self.last_response = ended
            if response.is_success:
                choices = payload.get('choices') or []
                if payload.get('model') != RESPONSE_MODEL:
                    self.protocol_errors.append('returned_model_mismatch:'+str(payload.get('model')))
                if choices:
                    choice = choices[0]; message = choice.get('message') or {}
                    if choice.get('finish_reason') == 'stop' and str(message.get('content') or '').strip() and not message.get('tool_calls'):
                        self.last_final = ended
            if response.status_code == 429: self.gate.rate_limited()
            self.record(dict(common,event='response',timestamp=now(),monotonic_seconds=ended,
                             elapsed_seconds=ended-started,http_status=response.status_code,
                             response=payload,usage=payload.get('usage')))
            return response
        except Exception as exc:
            self.record(dict(common,event='transport_error',error=str(exc),timestamp=now()))
            raise
        finally: self.gate.leave()

    def close(self): self.inner.close()

def client_for(directory, credentials, stage, gate):
    transport = Transport(directory,credentials,stage,gate)
    # Preserve official OpenAI SDK default two retries, plus the bounded official
    # outer retry helper; all actual sends share the gate and are traced.
    client = OpenAI(base_url=credentials['base_url'],api_key=credentials['api_key'],max_retries=2,
                    http_client=httpx.Client(transport=transport,timeout=HTTP_TIMEOUT_SECONDS,trust_env=False))
    client._rise_transport = transport
    return client,transport

def traced_retry(client, **kwargs):
    from rise.api_retry import chat_with_retry
    transport = client._rise_transport
    proxy = types.SimpleNamespace(**vars(time))
    def sleep(seconds):
        append(transport.directory/'retries.jsonl',dict(event='outer_retry_wait',seconds=seconds,timestamp=now()))
        time.sleep(seconds)
    proxy.sleep = sleep
    return clone(chat_with_retry, {'time':proxy})(client,**kwargs)

def clone(function, changes):
    namespace = dict(function.__globals__); namespace.update(changes)
    copied = types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied

class Sandbox:
    def __init__(self, workspace, scratch, rg_binary):
        self.workspace = Path(workspace).resolve(); self.scratch = Path(scratch).resolve()
        self.scratch.mkdir(parents=True,exist_ok=True)
        self.rg = Path(rg_binary).resolve(); self.binary = shutil.which('bwrap')
        if not self.binary or not self.rg.is_file(): raise RuntimeError('bwrap/ripgrep unavailable')

    def command(self, command):
        args = [self.binary,'--die-with-parent','--unshare-all','--clearenv','--ro-bind','/usr','/usr']
        for name in ['/bin','/sbin','/lib','/lib64']:
            p = Path(name)
            if p.is_symlink(): args += ['--symlink',str(p.readlink()),name]
            elif p.exists(): args += ['--ro-bind',name,name]
        args += ['--proc','/proc','--dev','/dev','--dir','/etc']
        if Path('/etc/ld.so.cache').exists(): args += ['--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache']
        args += ['--bind',str(self.scratch),'/tmp','--ro-bind',str(self.workspace),str(self.workspace),
                 '--dir','/tool-bin','--ro-bind',str(self.rg),'/tool-bin/rg',
                 '--setenv','PATH','/tool-bin:/usr/bin:/bin','--setenv','HOME','/tmp',
                 '--setenv','TMPDIR','/tmp','--setenv','LANG','C.UTF-8',
                 '--chdir',str(self.workspace),'--','/bin/bash','-c',command]
        return args

    def popen(self, argv, *args, **kwargs):
        if argv[:2] != ['/bin/bash','-c']: raise ValueError('Unexpected shell invocation')
        kwargs.pop('cwd',None)
        return subprocess.Popen(self.command(argv[2]),*args,**kwargs)

    def bash(self):
        from rise.tools import make_bash_tool
        proxy = types.SimpleNamespace(**vars(subprocess)); proxy.Popen = self.popen
        factory = clone(make_bash_tool, {'subprocess':proxy})
        # False disables only the upstream macOS launcher. Popen above always
        # enforces Linux bwrap; timeout, process group cleanup and output unchanged.
        return factory(self.workspace,truncate_chars=4000,enable_sandbox=False)
