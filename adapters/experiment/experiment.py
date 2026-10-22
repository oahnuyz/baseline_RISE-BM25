#!/usr/bin/env python3
"""Full RISE: one independently scheduled dataset per campaign process."""
from __future__ import annotations
import argparse
import dataclasses
import fcntl
import importlib.util
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter

from common import (ROOT, MODEL, RESPONSE_MODEL, UPSTREAM, now, sha, digest, read_json, rows, atomic_json,
                    atomic_text, append, usage_totals, code_manifest, BuildClock, clock_total,
                    new_attempt, safe_id)
sys.path.insert(0,str(ROOT/'src'))
from runtime import ServiceGate, ConcurrencyObserver, client_for, Sandbox, traced_retry, clone, HTTP_TIMEOUT_SECONDS
from rise.agent import Agent
import rise.agent as agent_module
from rise.protocol import for_rise
from rise.retrieval import build_index, load_index, retrieve
from rise.rise_agent import RISE_SYSTEM_PROMPT, RISE_SYSTEM_PROMPT_STRUCTURED, RISE_USER_PROMPT
from rise.tools import make_bm25_search_tool, make_read_tool, bm25_search_tool_spec, bash_tool_spec, read_tool_spec
import pyarrow as pa
import pyarrow.parquet as pq

_spec = importlib.util.spec_from_file_location('rise_sections',ROOT/'scripts/structured/section_one_doc.py')
section = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(section)
agent_module.chat_with_retry = traced_retry

ENTERPRISE_DATASET = 'enterprise_rag_bench_10970_stratified100'

def anchor_policy(dataset_name):
    return 'enterprise_literal_newline_v1' if dataset_name == ENTERPRISE_DATASET else 'native'

def enterprise_locate(doc_text, anchor, cursor, section_idx, n_total):
    """After native failure, accept only one exact escaped-newline occurrence."""
    result = section._locate_one(doc_text, anchor, cursor, section_idx, n_total)
    if result[0] != -1 or '\n' not in anchor:
        return result
    escaped = anchor.replace('\r', r'\r').replace('\n', r'\n')
    offset = doc_text.find(escaped)
    # Count overlapping occurrences too, and require global uniqueness rather
    # than silently choosing among repeated passages using the cursor.
    if offset < cursor or doc_text.find(escaped, offset + 1) != -1:
        return result
    return offset, len(escaped), 'enterprise-literal-newline-exact', None

def utf8(path):
    return Path(path).read_bytes().decode('utf-8')

def interleave(groups):
    for batch in itertools.zip_longest(*groups):
        for item in batch:
            if item is not None: yield item

def bounded_pool(items, function, on_done=None, workers=6, activity=None):
    iterator = iter(items); results = []
    def execute(item):
        if activity is not None: activity.enter()
        try: return function(item)
        finally:
            if activity is not None: activity.leave()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = {}
        def add():
            try: item = next(iterator)
            except StopIteration: return False
            running[pool.submit(execute,item)] = item
            return True
        for _ in range(workers): add()
        while running:
            done,_ = wait(running,return_when=FIRST_COMPLETED)
            for future in done:
                item = running.pop(future); value = future.result(); results.append(value)
                if on_done: on_done(item,value)
                add()
    return results

def validate_input(data, expected_qa_count=100):
    data = Path(data).resolve(); documents = rows(data/'documents.jsonl'); questions = rows(data/'qa.jsonl')
    seen_ids=set(); expected=set()
    for d in documents:
        path = data/d['path']; rel = Path(d['path'])
        if not rel.parts or rel.parts[0]!='corpus' or not path.resolve().is_relative_to(data/'corpus') or path.is_symlink():
            raise ValueError('Invalid corpus path')
        if d['id'] in seen_ids or d['path'] in expected: raise ValueError('Duplicate document')
        if sha(path)!=d['sha256']: raise ValueError('Corpus hash mismatch: '+d['id'])
        seen_ids.add(d['id']); expected.add(d['path'])
    actual={str(p.relative_to(data)) for p in (data/'corpus').rglob('*') if p.is_file()}
    if actual!=expected: raise ValueError('Missing/extra corpus files')
    if type(expected_qa_count) is not int or expected_qa_count<1: raise ValueError('Invalid expected QA count')
    if len(questions)!=expected_qa_count or len({q['id'] for q in questions})!=expected_qa_count:
        raise ValueError(f'Expected {expected_qa_count} unique prepared QA')
    for q in questions:
        if not q['question'] or not isinstance(q['gold_answers'],list): raise ValueError('Invalid QA schema')
        if not set(q.get('document_ids',[])).issubset(seen_ids): raise ValueError('Unknown evidence doc ID')
    return documents,questions

def active_documents(documents, spec):
    excluded=spec.get('excluded_document_ids',[])
    if not isinstance(excluded,list) or any(not isinstance(x,str) for x in excluded) or len(set(excluded))!=len(excluded):
        raise ValueError('Invalid excluded document IDs')
    if not set(excluded).issubset({d['id'] for d in documents}):
        raise ValueError('Unknown excluded document ID')
    active=[d for d in documents if d['id'] not in set(excluded)]
    if not active: raise ValueError('No active corpus documents')
    return active

def concurrency_settings(plan):
    if len(plan.get('datasets',[])) != 1:
        raise ValueError('Use one dataset per campaign/process for independent admission, retries and timing')
    settings=plan.get('concurrency',{})
    if set(settings)!={'structure','qa','judge'} or any(type(v) is not int or v<1 for v in settings.values()):
        raise ValueError('Explicit positive structure/qa/judge concurrency required')
    return dict(settings)

def configuration(plan):
    settings=concurrency_settings(plan)
    coerce_final=plan.get('coerce_final',False)
    if type(coerce_final) is not bool:raise ValueError('coerce_final must be boolean')
    structured=plan.get('structured_docs',True)
    if type(structured) is not bool:raise ValueError('structured_docs must be boolean')
    return dict(upstream_commit=UPSTREAM,model=MODEL,expected_response_model=RESPONSE_MODEL,code=code_manifest(),
        installed_packages={name:importlib.metadata.version(name) for name in ['bm25s','PyStemmer','openai','httpx','numpy','scipy','pandas','pyarrow','python-dotenv','tiktoken','tqdm']},
        rg_sha256=sha(plan['rg_binary']),
        concurrency={**settings,'scope':'one_dataset_per_process','qa_unit':'active_questions',
                     'qa_http_limit':None,'cross_dataset_shared_gate':False},
        datasets=plan['datasets'],credentials_path=plan['credentials'],rg_binary=plan['rg_binary'],
        algorithm={'name':'RISE' if structured else 'RISE-BM25','structured_docs':structured,'bm25_k':1000,'k1':1.5,'b':0.75,
                   'top_preview':10,'snippet_chars':100,'bash_chars':4000,'bash_timeout':60,
                   'read_default_lines':2000,'read_line_chars':2000,'max_turns':100,
                   'wall_clock_seconds':3600,'coerce_final':coerce_final,'context_level':'level0',
                   'assistant_reasoning_history':'preserved','reasoning_trace':'preserved'},
        api={'reasoning_controls':'omitted','qa':{'max_tokens':32000,'temperature':0.0,'outer_attempts':3},
             'structure':{'max_tokens':32768,'temperature':0.0,'outer_attempts':5,'one_document_per_request':True,
                          'input_max_chars':200000,'interface':'online_chat_completions'},
             'judge':{'max_tokens':'omitted','temperature':0.0,'outer_attempts':3},
             'sdk_retries':2,'http_timeout_seconds':HTTP_TIMEOUT_SECONDS,
             'admission':{'structure_and_judge':{'separate_stage_gates':True,'initial_send_interval':0.2,
                 '429_cooldown':5,'429_interval_multiplier':1.5,'max_send_interval':1.0},
                 'qa':{'http_limit':None,'send_interval':0,'shared_429_cooldown':False}}},
        metrics={'qa_time':'first_http_send_to_final_complete_response','accuracy':'mean_score_divided_by_4',
                 'qa_denominator':'all_valid_successful_qa','cached_input':'included','judge_cost_included':False,
                 'failed_attempts':'separate','index_time':'active_wall_clock_including_wait_and_save'},
        anchor_adaptation={spec['name']:anchor_policy(spec['name']) for spec in plan['datasets']},
        patches=['read_path_component_boundary']+(['toc_tracks_inserted_headings','anchor_cursor_filter'] if structured else []),
        overrides=['DeepSeek model for all phases','omit reasoning controls']+(['online rather than Batch structure',
                   'omit YAML frontmatter assumption in structured QA prompt'] if structured else [])+[
                   'preserve reasoning_content in assistant history and response traces per user configuration',
                   'RARG judge and measurement contract','Linux bwrap','explicit per-stage concurrency'],
        scheduling='one dataset per independent process; sequential stages within that process; QA has only question admission')

class Dataset:
    def __init__(self,spec,campaign):
        self.spec=spec; self.name=spec['name']; self.campaign=campaign
        self.root=campaign.root/self.name; self.root.mkdir(parents=True,exist_ok=True)
        self.data=self.root/'input'; self.source=Path(spec['source']).resolve()
        self.documents=[]; self.questions=[]; self.searcher=None; self.doc_ids=None

    def prepare(self):
        count=self.spec.get('expected_qa_count',100)
        docs,questions=validate_input(self.source,count)
        active=active_documents(docs,self.spec)
        fingerprint={'documents':sha(self.source/'documents.jsonl'),'qa':sha(self.source/'qa.jsonl'),
                     'corpus':digest([(d['id'],d['path'],d['sha256']) for d in docs])}
        config={'campaign_signature':self.campaign.signature,'data':fingerprint,'spec':self.spec}
        cp=self.root/'config.json'
        if cp.exists() and read_json(cp)!=config: raise ValueError('Dataset/config signature changed')
        atomic_json(cp,config)
        if not (self.data/'READY.json').exists():
            self.data.mkdir(exist_ok=True)
            for d in docs:
                dst=self.data/d['path']; dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists() or sha(dst)!=d['sha256']: shutil.copy2(self.source/d['path'],dst)
            for name in ['documents.jsonl','qa.jsonl']: shutil.copy2(self.source/name,self.data/name)
            validate_input(self.data,count)
            atomic_json(self.data/'READY.json',fingerprint)
        else:
            if read_json(self.data/'READY.json')!=fingerprint: raise ValueError('Snapshot signature changed')
            validate_input(self.data,count)
        self.source_document_count=len(docs)
        self.documents=active
        # Explicit ID selection from the frozen 500-QA input; never resample.
        selected=self.spec.get('qa_ids')
        if selected is not None:
            lookup={q['id']:q for q in questions}
            if len(selected)!=self.spec['selected_qa_count'] or len(set(selected))!=len(selected) or not set(selected).issubset(lookup):
                raise ValueError('Invalid frozen QA subset')
            questions=[lookup[qid] for qid in selected]
        self.questions=questions
        self.path_map={d['id']:str(Path(d['path']).relative_to('corpus')) for d in active}
        atomic_json(self.root/'document_map.json',self.path_map)
        parquet=self.root/'corpus.parquet'
        if not parquet.exists():
            # Conversion is outside build timing and contains only document IDs/text.
            table=pa.table({'docid':[d['id'] for d in active], 'text':[utf8(self.data/d['path']) for d in active]})
            temporary=parquet.with_suffix('.tmp');pq.write_table(table,temporary);os.replace(temporary,parquet)
        self.status('prepared')

    def status(self,stage,**extra):
        atomic_json(self.root/'status.json',dict(dataset=self.name,stage=stage,updated_at=now(),**extra))

    def docroot(self,d): return self.root/'structure'/safe_id(d['id'])
    def qaroot(self,q): return self.root/'qa'/safe_id(q['id'])

    def structure_valid(self,d):
        p=self.docroot(d)/'result.json'
        if not p.exists(): return False
        r=read_json(p); target=self.root/'structured'/self.path_map[d['id']]
        return r.get('status')=='complete' and not r.get('protocol_errors') and r.get('empty_reason')!='all-failed' and r.get('signature')==self.campaign.signature and target.exists() and sha(target)==r.get('structured_sha256')

    def build_bm25(self):
        target=self.root/'index'
        if (target/'COMPLETE.json').exists():
            receipt=read_json(target/'COMPLETE.json')
            if receipt['signature']!=self.campaign.signature: raise ValueError('Index signature mismatch')
            for rel,value in receipt['files'].items():
                if sha(target/rel)!=value: raise ValueError('Index file changed')
            return
        clock=BuildClock(self.root/'build_clock')
        try:
            self.status('building_bm25')
            temp=self.root/('index_build_'+str(time.time_ns()))
            build_index(self.root/'corpus.parquet',temp,k1=1.5,b=0.75)
            files={str(p.relative_to(temp)):sha(p) for p in temp.rglob('*') if p.is_file()}
            atomic_json(temp/'COMPLETE.json',{'signature':self.campaign.signature,'files':files})
            if target.exists(): raise ValueError('Incomplete index directory requires inspection')
            os.replace(temp,target)
            clock.finish('complete')
        except BaseException:
            clock.finish('failed');raise

    def load(self):
        if self.searcher is None:self.searcher,self.doc_ids=load_index(self.root/'index')

    def qa_valid(self,q):
        p=self.qaroot(q)/'result.json'
        if not p.exists():return False
        r=read_json(p)
        return r.get('status')=='complete' and r.get('signature')==self.campaign.signature and bool(r.get('answer'))

def render_structure(raw,data,dataset_name=None):
    proposals=data.get('sections')
    if not isinstance(proposals,list):raise ValueError('Missing sections array')
    for sec in proposals:
        if isinstance(sec,dict):
            for field in ['title','description']:
                v=sec.get(field,'')
                if not isinstance(v,str) or '\n' in v or '\r' in v:
                    raise ValueError('Section metadata must be a single line: '+field)
    policy=anchor_policy(dataset_name)
    locator=section.validate_and_locate
    if policy!='native':
        # A private function namespace avoids altering the native locator used
        # by another dataset/thread. Original model proposals remain untouched.
        locator=clone(locator,{'_locate_one':enterprise_locate})
    located,skipped,warnings=locator(raw,proposals)
    for sec in located:
        if sec['strategy']=='enterprise-literal-newline-exact':
            sec['matched_anchor']=sec['anchor'].replace('\r',r'\r').replace('\n',r'\n')
            sec['anchor_transform']='actual_newlines_to_literal_escape'
    reason=None if located else ('all-failed' if proposals else 'llm-zero')
    final=section.build_final(raw,located,reason)
    # Verify body construction from original segments and provide source mapping.
    restructured=section.build_restructured(raw,located)
    # Find the inserted TOC using its position (source may itself contain a TOC).
    split=0
    if restructured.startswith('---\n'):
        idx=restructured.find('\n---\n',4)
        if idx!=-1:split=idx+len('\n---\n')
    toc_start=split+1 if split else 0
    sentinel_end=final.index('=== DOCUMENT BODY ===\n',toc_start)+len('=== DOCUMENT BODY ===\n')
    toc_block=final[toc_start:sentinel_end]
    if section._insert_toc_block(restructured,toc_block)!=final:
        raise ValueError('Original-body preservation check failed')
    toc_lines=toc_block.splitlines()
    entries=[re.match(r'- L(\d+)–(\d+): (.*)',line) for line in toc_lines]
    entries=[m for m in entries if m]
    lines=final.splitlines()
    if len(entries)!=len(located):raise ValueError('TOC entry count mismatch')
    for m,sec in zip(entries,located):
        a,b=int(m[1]),int(m[2])
        if not 1<=a<=b<=len(lines) or lines[a-1]!='## '+sec['title']:
            raise ValueError('TOC points to wrong line')
    # Char-level source segments, exact offset accounting including inserted text.
    mapping=[]; original_cursor=0; position=0
    toc_delta=len(final)-len(restructured)
    for sec in located+[{'offset':len(raw),'_end':True}]:
        end=sec['offset']; chunk=raw[original_cursor:end]
        if chunk:
            # Split spans at the frontmatter/TOC insertion point.
            cuts=[position,position+len(chunk)]
            if position<split<position+len(chunk):cuts.insert(1,split)
            for x,y in zip(cuts,cuts[1:]):
                mapped=x+(toc_delta if x>=split else 0)
                original_start=original_cursor+x-position
                if final[mapped:mapped+y-x]!=raw[original_start:original_start+y-x]:
                    raise ValueError('Character provenance mismatch')
                mapping.append({'original_start':original_start,'original_end':original_start+y-x,
                                'structured_start':mapped,'structured_end':mapped+y-x})
        position+=len(chunk);original_cursor=end
        if not sec.get('_end'):
            prefix='\n' if end>0 and raw[end-1]!='\n' else ''
            position+=len(prefix+'## '+sec['title']+'\n\n')
    return final,dict(located=located,skipped=skipped,warnings=warnings,empty_reason=reason,
                     anchor_policy=policy,adapted_anchor_count=sum(s['strategy']=='enterprise-literal-newline-exact' for s in located),
                     body_preserved=True,toc_valid=True,source_spans=mapping)

def structure_one(item):
    dataset,d=item;c=dataset.campaign;root=dataset.docroot(d)
    attempt=new_attempt(root);client,transport=client_for(attempt,c.credentials,'structure',c.gates['structure'])
    started=time.monotonic();raw=utf8(dataset.data/d['path'])
    llm_text,truncation=section.truncate_for_llm(raw)
    result=dict(id=d['id'],signature=c.signature,attempt=str(attempt.relative_to(dataset.root)),status='failed',
                source_sha256=d['sha256'],started_at=now(),truncation=truncation)
    try:
        response=traced_retry(client,model=MODEL,messages=[{'role':'user','content':section.SECTION_PROMPT.replace('<<<DOC>>>',llm_text)}],
                              response_format={'type':'json_object'},temperature=0.0,max_tokens=32768,max_retries=5)
        choice=response.choices[0]
        if transport.protocol_errors:raise ValueError('Structure returned model differs: '+str(transport.protocol_errors))
        if choice.finish_reason!='stop':raise ValueError('Structure response was not complete')
        data=section._extract_json(choice.message.content or '')
        if not isinstance(data,dict):raise ValueError('Structure output is not an object')
        final,audit=render_structure(raw,data,dataset_name=dataset.name)
        atomic_json(attempt/'sections.json',data);atomic_json(attempt/'audit.json',audit)
        target=dataset.root/'structured'/dataset.path_map[d['id']]
        atomic_text(target,final)
        # Only final accepted API success enters build token metric. All other
        # actual requests remain in http.jsonl as separate failed/retried spend.
        accepted=[e for e in transport.events if e.get('event')=='response' and 200<=e.get('http_status',0)<300][-1:]
        result.update(status='complete',structured_sha256=sha(target),usage=usage_totals(accepted),
                      all_request_usage=usage_totals(transport.events),empty_reason=audit['empty_reason'],
                      located_count=len(audit['located']),skipped_count=len(audit['skipped']),warning_count=len(audit['warnings']),
                      anchor_policy=audit['anchor_policy'],adapted_anchor_count=audit['adapted_anchor_count'],
                      protocol_errors=transport.protocol_errors)
    except Exception as exc:
        result.update(error=str(exc).replace(c.credentials['api_key'],'[REDACTED]'),
                      all_request_usage=usage_totals(transport.events))
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,finished_at=now())
        atomic_json(attempt/'result.json',result);atomic_json(root/'result.json',result);client.close()
    return result

def qa_one(item):
    dataset,q=item;c=dataset.campaign;root=dataset.qaroot(q);attempt=new_attempt(root)
    workspace=attempt/'workspace';workspace.mkdir();client,transport=client_for(attempt,c.credentials,'qa',c.gates['qa'])
    tool_events=attempt/'tools.jsonl';tool_counter=0
    def tracked_retrieve(searcher,doc_ids,query,k):
        hits=retrieve(searcher,doc_ids,query,k)
        append(attempt/'retrieval.jsonl',{'query':query,'k':k,'ranked_hits':hits,'timestamp':now()})
        return hits
    factory=clone(make_bm25_search_tool,{'retrieve':tracked_retrieve})
    search=factory(searcher=dataset.searcher,doc_ids=dataset.doc_ids,docid_to_relpath=dataset.path_map,
                   bc_plus_docs_root=(dataset.root/'structured' if c.config['algorithm'].get('structured_docs',True)
                                      else dataset.data/'corpus'),working_dir=workspace,k=1000,top_n_preview=10,snippet_chars=100)
    sandbox=Sandbox(workspace,attempt/'scratch',c.plan['rg_binary'])
    functions={'search':search,'bash':sandbox.bash(),'read':make_read_tool(workspace,default_limit=2000)}
    def wrap(name,fn):
        def call(args):
            nonlocal tool_counter
            tool_counter+=1;ident=tool_counter;start=time.monotonic()
            append(tool_events,dict(event='start',call_id=ident,name=name,args=args,timestamp=now()))
            try:
                out=fn(args)
            except Exception as exc:
                append(tool_events,dict(event='error',call_id=ident,error=str(exc),timestamp=now()));raise
            append(tool_events,dict(event='return',call_id=ident,name=name,result=out,elapsed_seconds=time.monotonic()-start,timestamp=now()))
            if name=='search':
                append(attempt/'workspace.jsonl',{'call_id':ident,'paths':sorted(str(p.relative_to(workspace)) for p in workspace.rglob('*.txt'))})
            return out
        return call
    functions={name:wrap(name,fn) for name,fn in functions.items()}
    coerce_final=c.config['algorithm']['coerce_final']
    structured=c.config['algorithm'].get('structured_docs',True)
    config=for_rise(agent_model=MODEL,max_model_calls=100,coerce_final_on_max_turns=coerce_final,structured_doc_mode=structured,
                    bm25_k=1000,reasoning_effort='omitted_user_override')
    agent=Agent(client=client,model=MODEL,tools=functions,
                tool_specs=[bm25_search_tool_spec(),bash_tool_spec(),read_tool_spec()],
                max_turns=config.max_model_calls,per_call_max_tokens=config.per_call_max_tokens,
                temperature=c.config['api']['qa']['temperature'],
                max_retries=config.api_max_retries,wall_clock_timeout_sec=config.wall_clock_timeout_sec,
                coerce_final_on_max_turns=coerce_final,context_level='level0')
    result=dict(id=q['id'],question=q['question'],signature=c.signature,status='failed',
                attempt=str(attempt.relative_to(dataset.root)),started_at=now(),coerce_final=coerce_final)
    try:
        run=agent.run(system_prompt=RISE_SYSTEM_PROMPT_STRUCTURED if structured else RISE_SYSTEM_PROMPT,user_prompt=RISE_USER_PROMPT.format(question=q['question']),
                      trace_path=attempt/'native_trace.json',trace_relpath_to_docid={v:k for k,v in dataset.path_map.items()})
        choice=((transport.last_payload or {}).get('choices') or [{}])[0]
        accepted_reasons={'done','coerced_max_turns'} if coerce_final else {'done'}
        valid=run.terminated_reason in accepted_reasons and bool(run.final_answer.strip()) and choice.get('finish_reason')=='stop'
        valid=valid and not transport.protocol_errors and transport.last_final==transport.last_response
        result.update(status='complete' if valid else 'failed',answer=run.final_answer,iteration=len(run.turns),
                      terminated_by=run.terminated_reason,finish_reason=choice.get('finish_reason'),
                      tool_counts=run.tool_call_breakdown,protocol_errors=transport.protocol_errors,
                      elapsed_seconds=(transport.last_final-transport.first_request) if valid else None,
                      native_elapsed_seconds=run.elapsed_seconds,error=run.error or None)
        # Complete native dataclass supplements upstream compact trace without
        # modifying anything sent to the model.
        atomic_json(attempt/'native_full.json',dataclasses.asdict(run))
    except Exception as exc:result['error']=str(exc).replace(c.credentials['api_key'],'[REDACTED]')
    finally:
        result.update(usage=usage_totals(transport.events),finished_at=now())
        atomic_json(attempt/'result.json',result);atomic_json(root/'result.json',result);client.close()
    return result

def judge_one(item):
    dataset,q=item;c=dataset.campaign;root=dataset.qaroot(q);qa=read_json(root/'result.json')
    answer_hash=digest(qa['answer']);jd=new_attempt(root/'judge')
    client,transport=client_for(jd,c.credentials,'judge',c.gates['judge'])
    template=utf8(ROOT/'assets/judge.txt')
    values={'question':q['question'],'gold_answers_joined_by_pipe':' | '.join(q['gold_answers']),'generated_answer':qa['answer']}
    prompt=re.sub(r'\{(question|gold_answers_joined_by_pipe|generated_answer)\}',lambda m:values[m[1]],template)
    evaluation=dict(status='failed',answer_hash=answer_hash,signature=c.signature,attempt=str(jd.relative_to(dataset.root)))
    try:
        # RARG judge contract: up to three outer attempts, SDK two retries each.
        for i in range(3):
            try:
                response=client.chat.completions.create(model=MODEL,messages=[{'role':'user','content':prompt}],
                    temperature=0.0,response_format={'type':'json_object'})
                value=json.loads(response.choices[0].message.content or '')
                if response.choices[0].finish_reason!='stop' or type(value.get('score')) is not int or not 0<=value['score']<=4 or not isinstance(value.get('reasoning'),str) or not value['reasoning'].strip():
                    raise ValueError('Invalid judge result')
                if transport.protocol_errors:raise ValueError('Judge returned model differs')
                evaluation.update(status='complete',score=value['score'],reasoning=value['reasoning']);break
            except Exception as exc:
                evaluation['error']=str(exc).replace(c.credentials['api_key'],'[REDACTED]')
                if i<2:
                    append(jd/'retries.jsonl',{'outer_attempt':i+1,'wait_seconds':2**(i+1),'timestamp':now()});time.sleep(2**(i+1))
        atomic_json(jd/'result.json',evaluation);atomic_json(root/'evaluation.json',evaluation)
    finally:client.close()
    return evaluation

class Campaign:
    def __init__(self,plan_path):
        self.plan=read_json(plan_path);self.concurrency=concurrency_settings(self.plan)
        self.root=Path(self.plan['run_root']).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'run.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.config=configuration(self.plan);self.signature=digest(self.config)
        cp=self.root/'effective_config.json'
        if cp.exists() and read_json(cp)!=self.config:raise ValueError('Campaign code/config drift; create a new campaign')
        atomic_json(cp,self.config)
        self.credentials=read_json(self.plan['credentials'])
        self.gates={'structure':ServiceGate(self.concurrency['structure']),
                    'qa':ConcurrencyObserver(),'judge':ServiceGate(self.concurrency['judge'])}
        self.activity={stage:ConcurrencyObserver() for stage in self.concurrency}
        self.datasets=[Dataset(spec,self) for spec in self.plan['datasets']]
        self.status('starting')

    def status(self,stage,**extra):
        atomic_json(self.root/'state.json',dict(stage=stage,pid=os.getpid(),updated_at=now(),signature=self.signature,
                    api_peak_by_stage={k:g.peak for k,g in self.gates.items()},
                    task_peak_by_stage={k:a.peak for k,a in self.activity.items()},
                    service_interval_by_stage={k:getattr(g,'interval',None) for k,g in self.gates.items()},**extra))

    def summaries(self):
        for ds in self.datasets:
            build=[read_json(p) for p in (ds.root/'structure').glob('*/result.json')]
            excluded=set(ds.spec.get('excluded_document_ids',[]))
            excluded_build=[r for r in build if r['id'] in excluded]
            build=[r for r in build if r['id'] not in excluded]
            goodbuild=[r for r in build if r['status']=='complete']
            qr=[read_json(p) for p in (ds.root/'qa').glob('*/result.json')]
            good=[r for r in qr if r['status']=='complete']
            evaluations=[]
            for q in ds.questions:
                p=ds.qaroot(q)/'evaluation.json'
                if ds.qa_valid(q) and p.exists():
                    e=read_json(p);r=read_json(ds.qaroot(q)/'result.json')
                    if e['status']=='complete' and e['answer_hash']==digest(r['answer']):evaluations.append(e)
            def mean(key):return sum(r[key] for r in good)/len(good) if good else None
            summary=dict(dataset=ds.name,total_documents=len(ds.documents),total_questions=len(ds.questions),
                method=self.config['algorithm']['name'],structured_docs=self.config['algorithm']['structured_docs'],
                structure_required=self.config['algorithm']['structured_docs'],
                source_documents=getattr(ds,'source_document_count',len(ds.documents)),
                excluded_document_ids=ds.spec.get('excluded_document_ids',[]),
                structure_excluded_failed=sum(r['status']!='complete' for r in excluded_build),
                structure_complete=len(goodbuild),structure_failed=len(build)-len(goodbuild),
                structure_empty_reasons=dict(Counter(r.get('empty_reason') or 'has_sections' for r in goodbuild)),
                build_tokens=sum(r['usage']['total_tokens'] for r in goodbuild),
                build_usage_complete=all(r['usage']['usage_complete'] for r in goodbuild),build_time=clock_total(ds.root/'build_clock'),
                successful_qa=len(good),failed_qa=len(qr)-len(good),pending_qa=len(ds.questions)-len(qr),
                scored_questions=len(evaluations),accuracy=sum(e['score'] for e in evaluations)/(4*len(evaluations)) if evaluations else None,
                full_score_rate=sum(e['score']==4 for e in evaluations)/len(evaluations) if evaluations else None,
                mean_qa_seconds=mean('elapsed_seconds'),mean_iteration=mean('iteration'),
                mean_qa_tokens=sum(r['usage']['total_tokens'] for r in good)/len(good) if good else None,
                qa_usage_complete=all(r['usage']['usage_complete'] for r in good),
                mean_qa_input_tokens=sum(r['usage']['input_tokens'] for r in good)/len(good) if good else None,
                mean_qa_output_tokens=sum(r['usage']['output_tokens'] for r in good)/len(good) if good else None,
                online_embedding_tokens=0,judge_cost_included=False,
                accuracy_is_final=len(good)==len(ds.questions)==len(evaluations),updated_at=now())
            atomic_json(ds.root/'summary.json',summary)

    def run(self,stop_after='all'):
        for ds in self.datasets:ds.prepare()
        self.status('prepared')
        if stop_after=='prepare':return
        if not (self.root/'preflight_passed.json').exists():raise RuntimeError('Preflight receipt required')
        if read_json(self.root/'preflight_passed.json')['signature']!=self.signature:
            raise RuntimeError('Preflight receipt is for a different code/config signature')
        for ds in self.datasets:ds.build_bm25()
        if not self.config['algorithm']['structured_docs']:
            # Official RISE-BM25 uses untouched corpus files and skips all
            # structural generation; only local BM25 construction is timed.
            for ds in self.datasets:ds.load();ds.status('index_complete')
            self.summaries();self.status('index_complete')
            if stop_after=='index':return
            self.qa_stage(smoke=True)
            if stop_after=='smoke':return
            self.qa_stage(smoke=False);self.summaries()
            complete=all(read_json(ds.root/'summary.json')['accuracy_is_final'] for ds in self.datasets)
            for ds in self.datasets:ds.status('complete' if read_json(ds.root/'summary.json')['accuracy_is_final'] else 'needs_attention')
            self.status('complete' if complete else 'needs_attention')
            return
        self.status('structuring');clocks={ds.name:BuildClock(ds.root/'build_clock') for ds in self.datasets}
        for ds in self.datasets:ds.status('structuring')
        counter=0; consecutive_failures=0
        def done(item,result):
            nonlocal counter,consecutive_failures
            counter+=1
            consecutive_failures=consecutive_failures+1 if result['status']!='complete' else 0
            if consecutive_failures>=6:
                raise RuntimeError('Six consecutive structure failures; stopped dispatch, inspect API/structure traces')
            if counter%30==0:self.summaries();self.status('structuring',completed_this_session=counter)
        try:
            jobs=interleave([[(ds,d) for d in ds.documents if not ds.structure_valid(d)] for ds in self.datasets])
            bounded_pool(jobs,structure_one,done,workers=self.concurrency['structure'],activity=self.activity['structure'])
        finally:
            for clock in clocks.values():clock.finish('finished')
            self.summaries()
        for ds in self.datasets:
            summary=read_json(ds.root/'summary.json')
            if summary['structure_complete']!=len(ds.documents) or not summary['build_usage_complete']:
                raise RuntimeError('Incomplete structural corpus or usage: '+ds.name)
            suspicious=[read_json(ds.docroot(d)/'result.json') for d in ds.documents]
            if any(r.get('empty_reason')=='all-failed' or r.get('protocol_errors') for r in suspicious):
                raise RuntimeError('Structure anchor/model integrity review required: '+ds.name)
            ds.status('index_complete');ds.load()
        if stop_after=='index':self.status('index_complete');return
        self.qa_stage(smoke=True)
        if stop_after=='smoke':return
        self.qa_stage(smoke=False)
        self.summaries()
        complete=all(read_json(ds.root/'summary.json')['accuracy_is_final'] for ds in self.datasets)
        for ds in self.datasets:ds.status('complete' if read_json(ds.root/'summary.json')['accuracy_is_final'] else 'needs_attention')
        self.status('complete' if complete else 'needs_attention')

    def qa_stage(self,smoke):
        name='smoke' if smoke else 'full'
        self.status(name+'_qa')
        groups=[]
        for ds in self.datasets:
            ds.status(name+'_qa');qs=ds.questions[:6] if smoke else ds.questions
            # On an explicit process resume, infrastructure failures may create a
            # new attempt. Never reroll deterministic budget/no-answer outcomes.
            jobs=[]
            for q in qs:
                path=ds.qaroot(q)/'result.json'
                old=read_json(path) if path.exists() else None
                if old is None or (old.get('status')=='failed' and old.get('terminated_by') in [None,'api_error']):
                    jobs.append((ds,q))
            groups.append(jobs)
        bounded_pool(interleave(groups),qa_one,lambda *_:self.summaries(),workers=self.concurrency['qa'],activity=self.activity['qa'])
        self.status(name+'_judge')
        groups=[]
        for ds in self.datasets:
            ds.status(name+'_judge');qs=ds.questions[:6] if smoke else ds.questions
            jobs=[]
            for q in qs:
                if not ds.qa_valid(q):continue
                p=ds.qaroot(q)/'evaluation.json';qa=read_json(ds.qaroot(q)/'result.json')
                e=read_json(p) if p.exists() else {}
                if e.get('status')!='complete' or e.get('answer_hash')!=digest(qa['answer']):jobs.append((ds,q))
            groups.append(jobs)
        bounded_pool(interleave(groups),judge_one,lambda *_:self.summaries(),workers=self.concurrency['judge'],activity=self.activity['judge'])
        if smoke:
            for ds in self.datasets:
                for q in ds.questions[:6]:
                    if not ds.qa_valid(q):raise RuntimeError('Smoke QA failed: '+q['id'])
                    r=read_json(ds.qaroot(q)/'result.json');p=ds.qaroot(q)/'evaluation.json'
                    if not r['usage']['usage_complete'] or not p.exists() or read_json(p)['status']!='complete':
                        raise RuntimeError('Smoke usage/judge failed: '+q['id'])
            atomic_json(self.root/'smoke_passed.json',{'signature':self.signature,'timestamp':now(),'qa_per_dataset':6})
            self.status('smoke_passed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--stop-after',choices=['prepare','index','smoke','all'],default='all')
    args=p.parse_args();c=Campaign(args.plan)
    try:c.run(args.stop_after)
    except BaseException as exc:
        c.status('needs_attention',error=str(exc).replace(c.credentials['api_key'],'[REDACTED]'));raise

if __name__=='__main__':main()
