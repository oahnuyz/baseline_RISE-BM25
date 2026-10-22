#!/usr/bin/env python3
"""Per-dataset probes, isolated from formal costs; QA admission counts questions."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
from common import ROOT,atomic_json,atomic_text,read_json,digest,now,sha
from experiment import Campaign,Dataset,structure_one,qa_one,judge_one,bounded_pool,render_structure
from rise.retrieval import build_index
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    c=Campaign(args.plan)
    try:
        c.run('prepare')
        receipt=c.root/'preflight_passed.json'
        if receipt.exists():
            if read_json(receipt)['signature']!=c.signature:raise ValueError('Preflight signature mismatch')
            print('Existing matching preflight receipt; no API repeated',flush=True);return
        c.status('preflight_structure')
        jobs=[]
        structured_mode=c.config['algorithm']['structured_docs']
        for ds in c.datasets if structured_mode else []:
            probe=copy.copy(ds);probe.root=c.root/'preflight'/ds.name
            sizes=sorted(ds.documents,key=lambda d:d['size_bytes'])
            chosen=[sizes[int((len(sizes)-1)*p)] for p in [0.25,0.5,0.75,0.95]]
            atomic_json(probe.root/'selected_documents.json',[d['id'] for d in chosen])
            for d in chosen:
                if not probe.structure_valid(d):jobs.append((probe,d))
        result=bounded_pool(jobs,structure_one,workers=c.concurrency['structure'],activity=c.activity['structure'])
        if any(r['status']!='complete' or not r['usage']['usage_complete'] or r.get('protocol_errors') or r.get('empty_reason')=='all-failed' for r in result):
            raise RuntimeError('Real structure probe failed; inspect preflight traces')
        c.status('preflight_qa')
        ds=Dataset({'name':'preflight_fixture','source':str(c.root)},c)
        ds.root=c.root/'preflight'/'qa_fixture';ds.root.mkdir(parents=True,exist_ok=True)
        ds.data=ds.root/'input'
        labels=['ALPHATEST','BETATEST','GAMMATEST','DELTATEST','EPSILONTEST','ZETATEST']
        docs=[];questions=[]
        for i in range(1001):
            label=labels[i] if i<6 else 'DISTRACTOR'+str(i)
            code=digest({'label':label,'signature':c.signature})[:16]
            text=f'Archive entry {label}.\nThe validation code recorded for {label} is {code}.\n'
            rel=f'doc{i:04d}.txt';docid=str(i)
            docs.append({'docid':docid,'text':text,'rel':rel})
            if structured_mode:
                structured,_=render_structure(text,{'sections':[]})
                atomic_text(ds.root/'structured'/rel,structured)
            else:atomic_text(ds.data/'corpus'/rel,text)
            if i<6:questions.append({'id':'probe_'+str(i),'question':f'According to the archive, what is the validation code for {label}?', 'gold_answers':[code]})
        ds.documents=[{'id':d['docid']} for d in docs];ds.questions=questions
        ds.path_map={d['docid']:d['rel'] for d in docs}
        pq.write_table(pa.table({'docid':[d['docid'] for d in docs],'text':[d['text'] for d in docs]}),ds.root/'corpus.parquet')
        if not (ds.root/'index'/'doc_ids.json').exists():build_index(ds.root/'corpus.parquet',ds.root/'index')
        ds.load()
        qa_jobs=[(ds,q) for q in questions if not ds.qa_valid(q)]
        bounded_pool(qa_jobs,qa_one,workers=c.concurrency['qa'],activity=c.activity['qa'])
        for q in questions:
            r=read_json(ds.qaroot(q)/'result.json')
            if r['status']!='complete' or not r['usage']['usage_complete'] or r['tool_counts'].get('search',0)<1:
                raise RuntimeError('Actual agent/API/tools/usage preflight failed: '+q['id'])
        c.status('preflight_judge')
        bounded_pool([(ds,q) for q in questions if not (ds.qaroot(q)/'evaluation.json').exists() or read_json(ds.qaroot(q)/'evaluation.json')['status']!='complete'],judge_one,
                     workers=c.concurrency['judge'],activity=c.activity['judge'])
        if any(read_json(ds.qaroot(q)/'evaluation.json').get('score')!=4 for q in questions):
            raise RuntimeError('Preflight answer/judge validation not 4/4')
        if len(jobs)>=c.concurrency['structure'] and c.gates['structure'].peak!=c.concurrency['structure']:
            raise RuntimeError('Structure API probe did not reach configured concurrency')
        if len(qa_jobs)>=c.concurrency['qa'] and c.activity['qa'].peak!=c.concurrency['qa']:
            raise RuntimeError('QA probe did not reach configured active-question concurrency')
        evidence={str(p.relative_to(c.root)):sha(p) for p in (c.root/'preflight').rglob('*.json')}
        atomic_json(receipt,{'signature':c.signature,'timestamp':now(),'status':'passed',
                            'api_peak_by_stage':{k:g.peak for k,g in c.gates.items()},
                            'task_peak_by_stage':{k:a.peak for k,a in c.activity.items()},
                            'structure_real_docs':4 if structured_mode else 0,'new_structure_calls':len(jobs),'new_qa':len(qa_jobs),
                            'agent_qa':6,'judge':6,'evidence':evidence,
                            'cost_policy':'all preflight costs excluded from formal metrics'})
        c.status('preflight_passed');print('PREFLIGHT PASSED',flush=True)
    except BaseException as exc:
        c.status('preflight_failed',error=str(exc).replace(c.credentials['api_key'],'[REDACTED]'));raise

if __name__=='__main__':main()
