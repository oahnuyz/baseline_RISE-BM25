"""Persistence, fingerprints and API accounting for the prepared-data adapter."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'deepseek-v4-flash'
RESPONSE_MODEL = 'deepseek-v4-flash-ga-260731'
UPSTREAM = '32672b59ead8381decae7d412f05bb53d399946a'

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

_append_lock = threading.RLock()
def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with _append_lock, path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()

def code_manifest():
    paths = []
    for folder in ['src/rise', 'scripts/structured', 'adapters/experiment']:
        paths.extend((ROOT / folder).glob('*.py'))
    paths.extend([ROOT / 'assets/judge.txt', ROOT / 'adapters/requirements.txt'])
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}

def usage_totals(events):
    result = dict(input_tokens=0, output_tokens=0, total_tokens=0,
                  successful_requests=0, missing_usage_requests=0, usage_complete=True)
    seen = set()
    for event in events:
        if event.get('event') != 'response' or not 200 <= event.get('http_status', 0) < 300:
            continue
        key = event['http_id']
        if key in seen:
            raise ValueError('Duplicate response in usage ledger: ' + str(key))
        seen.add(key); result['successful_requests'] += 1
        u = event.get('usage') or {}
        fields = [u.get(k) for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']]
        valid = all(type(v) is int and v >= 0 for v in fields)
        valid = valid and fields[0] + fields[1] == fields[2]
        if valid:
            for key, value in zip(['input_tokens', 'output_tokens', 'total_tokens'], fields):
                result[key] += value
        else:
            result['missing_usage_requests'] += 1
            result['usage_complete'] = False
    return result

class BuildClock:
    """Persist active build wall time, with an explicit unclean-exit uncertainty."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        for p in self.root.glob('*.json'):
            r = read_json(p)
            if r['status'] == 'running':
                r.update(status='interrupted', time_complete=False,
                         uncertainty='Unrecorded tail after last 1-second heartbeat excluded')
                atomic_json(p, r)
        self.path = self.root / (str(time.time_ns()) + '.json')
        self.start = time.monotonic(); self.stop = threading.Event(); self.error = None
        self.record = dict(started_at=now(), status='running', time_complete=True)
        self.persist(); self.thread = threading.Thread(target=self.loop, daemon=True); self.thread.start()

    def persist(self):
        self.record.update(elapsed_seconds=time.monotonic()-self.start, updated_at=now())
        atomic_json(self.path, self.record)

    def loop(self):
        while not self.stop.wait(1):
            try: self.persist()
            except Exception as exc: self.error = exc; break

    def finish(self, status):
        self.stop.set(); self.thread.join(); self.record['status'] = status; self.persist()
        if self.error: raise self.error

def clock_total(root):
    records = [read_json(p) for p in Path(root).glob('*.json')]
    return {'seconds': sum(x['elapsed_seconds'] for x in records),
            'complete': all(x['time_complete'] for x in records), 'sessions': len(records)}

def new_attempt(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name.split('_')[1]) for p in root.glob('attempt_*') if p.is_dir()]
    attempt = root / f'attempt_{max(numbers, default=0)+1}'
    attempt.mkdir()
    return attempt

def safe_id(value):
    # Exact source ID is retained in records; filenames cannot introduce paths.
    return hashlib.sha256(str(value).encode()).hexdigest()[:24]
