"""Fresh v2 ingestion measurement, isolated from frozen production artifacts."""
import json
from pathlib import Path
import time
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from common import atomic_json, sha, now
from rise.retrieval import build_index, load_index, retrieve

def measure(c, ds):
    receipt=c.root/'ingestion_timing_v2.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['signature']==c.signature
        return
    dest=c.root/'measurements/ingestion_v2'
    assert not dest.exists(),'Inspect incomplete independent measurement before retry'
    dest.mkdir(parents=True)
    c.status('ingestion_v2_measurement')
    start=time.monotonic()
    table=pa.table({'docid':[d['id'] for d in ds.documents],
                    'text':[(ds.data/d['path']).read_bytes().decode('utf-8') for d in ds.documents]})
    parquet=dest/'corpus.parquet';pq.write_table(table,parquet)
    conversion=time.monotonic()-start
    del table;gc.collect()
    assert sha(parquet)==sha(ds.root/'corpus.parquet'),'Fresh Parquet differs from production'
    start=time.monotonic();build_index(parquet,dest/'index',k1=1.5,b=0.75)
    bm25=time.monotonic()-start
    assert sha(dest/'index/doc_ids.json')==sha(ds.root/'index/doc_ids.json')
    ds.load();new,ids=load_index(dest/'index')
    checks=[]
    for q in ds.questions[:6]:
        a=retrieve(ds.searcher,ds.doc_ids,q['question'],k=1000)
        b=retrieve(new,ids,q['question'],k=1000)
        assert a==b,'Measured index retrieval differs'
        checks.append(q['id'])
    del new;gc.collect()
    atomic_json(receipt,dict(timestamp=now(),signature=c.signature,timing_version=2,
        source_run=str(c.root),measurement_directory=str(dest),documents=len(ds.documents),
        conversion_seconds=conversion,bm25_seconds=bm25,total_ingestion_seconds=conversion+bm25,
        input_api_tokens=0,output_api_tokens=0,total_api_tokens=0,
        fresh_parquet=True,fresh_bm25=True,parquet_byte_equal=True,doc_ids_equal=True,
        six_top1000_rankings_and_scores_equal=checks,production_artifacts_unchanged=True,
        cache_conditions='No OS cache flush; source was audited and original index built first; shared server remains active',
        excluded=['dataset export','input snapshot copying','integrity audits','preflight','manual waiting','index loading','post-build checks'],
        reporting='Authoritative complete v2 ingestion measurement; frozen native summary build_time is BM25-only'))
    print('INGESTION_V2_MEASURED',conversion,bm25,conversion+bm25,flush=True)
