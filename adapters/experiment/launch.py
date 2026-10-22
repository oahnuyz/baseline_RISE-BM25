#!/usr/bin/env python3
"""Detached local supervisor. Never installs or changes experimental code."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from common import ROOT,read_json,atomic_json,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);args=p.parse_args()
    plan=read_json(args.plan);root=Path(plan['run_root']);root.mkdir(parents=True,exist_ok=True)
    stop_after=plan.get('stop_after','all')
    if stop_after not in {'prepare','index','smoke','all'}:raise ValueError('Invalid stop_after stage')
    lock=(root/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    record={'supervisor_pid':os.getpid(),'started_at':now(),'status':'running'}
    atomic_json(root/'launcher.json',record)
    code=2
    for script in ['preflight.py','experiment.py']:
        command=[str(ROOT/'.venv/bin/python'),'-u','-B',str(ROOT/'adapters/experiment'/script),'--plan',str(args.plan.resolve())]
        if script=='experiment.py':command+=['--stop-after',stop_after]
        with (root/'run.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            record.update(stage=script,child_pid=child.pid);atomic_json(root/'launcher.json',record)
            code=child.wait()
        if code:break
    state=read_json(root/'state.json') if (root/'state.json').exists() else {}
    successful=code==0 and state.get('stage') in {'complete','prepared','index_complete','smoke_passed'}
    record.update(status=('complete' if state.get('stage')=='complete' else 'requested_stage_finished') if successful else 'needs_attention',
                  exit_code=code,finished_at=now(),experiment_stage=state.get('stage'))
    atomic_json(root/'launcher.json',record)

if __name__=='__main__':main()
