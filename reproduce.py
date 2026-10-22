#!/usr/bin/env python3
"""Portable entry points for the EnterpriseRAGBench RISE-BM25 runner.

Only `download`, `prepare`, `plan`, and `measure` write files. No command here
starts QA or calls an LLM. Start the archived launcher explicitly after review.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def id_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def verify_inputs():
    ids = (ROOT / 'data/enterprise_10970_ids.txt').read_text(encoding='utf-8').splitlines()
    require(len(ids) == len(set(ids)) == 10970, 'Invalid small-corpus document IDs')
    subsets = read_json(ROOT / 'data/qa_subsets.json')
    for name, count in [('original100', 100), ('remaining400', 400), ('all500', 500)]:
        require(len(subsets[name]) == len(set(subsets[name])) == count, 'Invalid subset: ' + name)
    require(not set(subsets['original100']) & set(subsets['remaining400']), 'QA subsets overlap')
    require(set(subsets['all500']) == set(subsets['original100']) | set(subsets['remaining400']), 'Incomplete QA union')
    return {'small_corpus_ids': len(ids), 'subsets': '100 + 400 = 500, disjoint'}


def download(args):
    manifest = read_json(ROOT / 'data/downloads.json')
    for rel, entry in manifest.items():
        path = args.out / rel
        if path.exists():
            require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Existing download differs: ' + rel)
            print('VERIFIED', rel, flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.part')
        require(not partial.exists(), 'Inspect/remove incomplete download before retry: ' + str(partial))
        print('DOWNLOADING', rel, entry['bytes'], 'bytes', flush=True)
        with urllib.request.urlopen(entry['url'], timeout=120) as src, partial.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        require(partial.stat().st_size == entry['bytes'] and sha(partial) == entry['sha256'], 'Download checksum mismatch: ' + rel)
        partial.rename(path)


def verify_downloads(raw):
    for rel, entry in read_json(ROOT / 'data/downloads.json').items():
        path = raw / rel
        require(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Raw input differs or missing: ' + rel)


def convert_question(q, scale):
    return dict(schema_version='1.0', id='enterprise_rag_bench:' + q['question_id'],
                dataset='enterprise_rag_bench', variant='enterprise_rag_bench_' + str(scale),
                question=q['question'], gold_answers=[q['gold_answer']],
                evidence=q['answer_facts'], category=q['question_type'], document_ids=q['expected_doc_ids'],
                metadata={'source_question_id': q['question_id'], 'source_document_ids': q['expected_doc_ids'], 'original_record': q})


def prepare(args):
    verify_inputs()
    verify_downloads(args.raw)
    import pyarrow.parquet as pq
    small_order = (ROOT / 'data/enterprise_10970_ids.txt').read_text(encoding='utf-8').splitlines()
    references = read_json(ROOT / 'data/corpus_reference.json')
    if args.scale == 10970:
        order = small_order
    else:
        frozen = args.raw / 'code/data_manifests/frozen'
        order = (frozen / 'bedrock_dsids.txt').read_text(encoding='utf-8').splitlines()
        with gzip.open(frozen / 'ocean_order.txt.gz', 'rt', encoding='utf-8') as f:
            order += f.read().splitlines()
        reference = references['511959']
        require(id_hash(order) == reference['order_sha256'] and id_hash(sorted(order)) == reference['set_sha256'], 'Full-corpus order/set differs')
    require(len(order) == len(set(order)) == args.scale, 'Wrong document count')
    desired = set(order)
    out = args.out.resolve()
    staging = out.with_name('.' + out.name + '.building')
    require(not out.exists() and not staging.exists(), 'Output or incomplete staging already exists; refusing overwrite')
    staging.mkdir(parents=True)
    (staging / 'INCOMPLETE').write_text('Export has not passed validation.\n')
    records = {}
    byte_count = 0

    def export(did, source, text):
        nonlocal byte_count
        if did not in desired:
            return
        require(did not in records, 'Duplicate document: ' + did)
        require(re.fullmatch(r'[A-Za-z0-9._-]+', did) and re.fullmatch(r'[A-Za-z0-9._-]+', source), 'Unsafe source identifier')
        require(isinstance(text, str) and bool(text.strip()), 'Empty document: ' + did)
        body = text.encode('utf-8')  # Preserve real LF and literal backslash-n exactly.
        rel = 'corpus/' + source + '/' + did + '.txt'
        checksum = hashlib.sha256(body).hexdigest()
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(body)
        require(sha(target) == checksum, 'Written file checksum mismatch: ' + did)
        records[did] = dict(schema_version='1.0', id=did, dataset='enterprise_rag_bench',
                            source_id=did, path=rel, media_type='text/plain', size_bytes=len(body), sha256=checksum)
        byte_count += len(body)
        if len(records) % 25000 == 0:
            print('EXPORTED', len(records), flush=True)

    package = args.raw / 'corpus_package'
    for batch in pq.ParquetFile(package / 'reconstruction_seed/enterprise_corpus_meta.parquet').iter_batches(batch_size=2048, columns=['dsid', 'source', 'text']):
        for row in batch.to_pylist():
            export(row['dsid'], row['source'], row['text'])
    for stem in ['company_overview', 'initiatives']:
        export('__scaffold_' + stem, 'scaffold', (package / 'benchmark/generated_data' / (stem + '.md')).read_bytes().decode('utf-8').strip())
    require(set(records) == desired, 'Missing corpus documents')
    require(byte_count == references[str(args.scale)]['text_bytes'], 'Corpus byte count differs')
    small_identity = [(did, records[did]['path'], records[did]['sha256']) for did in small_order]
    require(digest(small_identity) == references['10970']['identity_sha256'], 'Small corpus document content, paths or order differ')
    subset = read_json(ROOT / 'data/qa_subsets.json')
    questions = [convert_question(q, args.scale) for q in rows(package / 'benchmark/questions.jsonl')]
    lookup = {q['id']: q for q in questions}
    require(len(questions) == len(lookup) == 500 and set(lookup) == set(subset['all500']), 'QA IDs differ')
    questions = [lookup[qid] for qid in subset['all500']]
    canonical = [{k: q[k] for k in ['id', 'question', 'gold_answers', 'document_ids', 'category']} for q in questions]
    require(digest(canonical) == subset['qa_semantic_sha256'], 'Historical question/gold/evidence IDs differ')
    require(all(q['question'].strip() and q['gold_answers'][0].strip() and set(q['document_ids']) <= desired for q in questions), 'Invalid QA or missing evidence document')
    for filename, values in [('documents.jsonl', (records[did] for did in order)), ('qa.jsonl', questions),
                             ('qa_stratified100.jsonl', (lookup[qid] for qid in subset['original100']))]:
        with (staging / filename).open('x', encoding='utf-8') as f:
            for row in values:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    shutil.copyfile(package / 'LICENSE', staging / 'LICENSE')
    write_json(staging / 'validation.json', dict(status='passed', documents=len(records), qa=500,
        text_bytes=byte_count, document_order_sha256=id_hash(order), qa_semantic_sha256=digest(canonical),
        model_api_calls=0, all_written_bytes_verified=True,
        note='Document bytes, paths, order and QA content are fixed.'))
    (staging / 'INCOMPLETE').unlink()
    staging.rename(out)
    print('PREPARED', out)


def make_plan(args):
    verify_inputs()
    source = args.data.resolve()
    dsname = 'enterprise_rag_bench_' + str(args.scale)
    require(not (source / 'INCOMPLETE').exists(), 'Prepared data is incomplete')
    require((source / 'documents.jsonl').is_file() and (source / 'qa.jsonl').is_file(), 'Prepared manifests missing')
    docs = rows(source / 'documents.jsonl')
    require(len(docs) == args.scale, 'Wrong corpus scale')
    if args.scale == 10970:
        expected = read_json(ROOT / 'data/corpus_reference.json')['10970']['identity_sha256']
        require(digest([(d['id'], d['path'], d['sha256']) for d in docs]) == expected, 'Small corpus order/content identity differs')
    else:
        require(id_hash([d['id'] for d in docs]) == read_json(ROOT / 'data/corpus_reference.json')['511959']['order_sha256'], 'Full corpus order differs')
    subsets = read_json(ROOT / 'data/qa_subsets.json')
    qs = rows(source / 'qa.jsonl')
    canonical = [{k: q[k] for k in ['id', 'question', 'gold_answers', 'document_ids', 'category']} for q in qs]
    require(digest(canonical) == subsets['qa_semantic_sha256'], 'QA content or order differs')
    ids = subsets[args.subset]
    concurrency = args.concurrency if args.concurrency is not None else 4
    require(concurrency > 0, 'Concurrency must be positive')
    plan = dict(run_root=str(args.run_root.resolve()), credentials=str(args.credentials.resolve()),
                rg_binary=str(args.rg.resolve()), structured_docs=False, coerce_final=True,
                concurrency=dict(structure=concurrency, qa=concurrency, judge=concurrency),
                datasets=[dict(name=dsname, source=str(source), expected_qa_count=500,
                               qa_ids=ids, selected_qa_count=len(ids))])
    write_json(args.out, plan)
    print('PLAN', args.out, 'QA=', len(ids), 'concurrency=', concurrency, '(no API calls made)')


def summarize(args):
    subsets = read_json(ROOT / 'data/qa_subsets.json')
    expected_ids = set(subsets[args.subset])
    seen = set()
    records = []
    compatibility = None
    for run in args.runs:
        config = read_json(run / 'effective_config.json')
        spec = config['datasets'][0]
        dataset = run / spec['name']
        identity = {k: config[k] for k in ['model', 'expected_response_model', 'code', 'installed_packages', 'rg_sha256', 'algorithm', 'api', 'metrics']}
        identity['corpus_fingerprint'] = read_json(dataset / 'config.json')['data']['corpus']
        if compatibility is None:
            compatibility = identity
        require(identity == compatibility, 'Runs differ in corpus, code, model or measurement settings')
        signature = digest(config)
        ids = spec.get('qa_ids', subsets['all500'])
        require(not seen.intersection(ids), 'Run QA sets overlap; refusing double counting')
        seen.update(ids)
        for qid in ids:
            key = hashlib.sha256(qid.encode()).hexdigest()[:24]
            qa = dataset / 'qa' / key
            result = read_json(qa / 'result.json')
            evaluation = read_json(qa / 'evaluation.json')
            require(result['id'] == qid and result['status'] == evaluation['status'] == 'complete', 'QA/judge incomplete: ' + qid)
            require(result['signature'] == evaluation['signature'] == signature, 'Stale result: ' + qid)
            require(evaluation['answer_hash'] == digest(result['answer']), 'Stale evaluation: ' + qid)
            require(result['usage']['usage_complete'], 'Incomplete token ledger: ' + qid)
            records.append((result, evaluation))
    require(seen == expected_ids and len(records) == len(expected_ids), 'Not the exact requested QA subset')
    n = len(records)
    out = dict(questions=n, accuracy=sum(e['score'] for _, e in records)/(4*n),
               full_score_rate=sum(e['score'] == 4 for _, e in records)/n,
               mean_qa_seconds=sum(r['elapsed_seconds'] for r, _ in records)/n,
               mean_iteration=sum(r['iteration'] for r, _ in records)/n,
               mean_qa_tokens=sum(r['usage']['total_tokens'] for r, _ in records)/n,
               mean_qa_input_tokens=sum(r['usage']['input_tokens'] for r, _ in records)/n,
               mean_qa_output_tokens=sum(r['usage']['output_tokens'] for r, _ in records)/n,
               judge_cost_included=False, accuracy_is_final=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def measure(args):
    # Run only against a finished campaign; Campaign owns its run.lock.
    plan = read_json(args.plan)
    run_root = Path(plan['run_root'])
    state_path = run_root / 'state.json'
    old_state = read_json(state_path)
    require(old_state['stage'] == 'complete', 'Measure only a completed, idle campaign')
    sys.path[:0] = [str(ROOT / 'adapters/experiment'), str(ROOT / 'src'), str(ROOT / 'scripts')]
    from experiment import Campaign
    from common import atomic_json
    from measure_ingestion_original import measure as original_measure
    c = Campaign(args.plan)
    try:
        for ds in c.datasets:
            status_path = ds.root / 'status.json'
            old_status = read_json(status_path)
            try:
                ds.prepare()
                require((ds.root / 'index/COMPLETE.json').is_file(), 'Production index missing')
                original_measure(c, ds)
            finally:
                atomic_json(status_path, old_status)
    finally:
        atomic_json(state_path, old_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Verify bundled corpus IDs and exact QA split; no API')
    p = sub.add_parser('download', help='Explicitly download the pinned public data (~1.4 GB)')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('prepare', help='Export exact text files/manifests from cached public data')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scale', type=int, choices=[10970, 511959], required=True)
    p = sub.add_parser('plan', help='Write one run plan; no API')
    p.add_argument('--scale', type=int, choices=[10970, 511959], required=True)
    p.add_argument('--subset', choices=['original100', 'remaining400', 'all500'], required=True)
    for name in ['data', 'run-root', 'credentials', 'rg', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--concurrency', type=int)
    p = sub.add_parser('summarize', help='Read completed runs, reject duplicates/failures, print pooled metrics')
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--subset', choices=['original100', 'remaining400', 'all500'], default='all500')
    p = sub.add_parser('measure', help='Fresh conversion + BM25 timing of an idle completed campaign; no API')
    p.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps(verify_inputs(), indent=2))
    else:
        {'download': download, 'prepare': prepare, 'plan': make_plan, 'summarize': summarize, 'measure': measure}[args.command](args)


if __name__ == '__main__':
    main()
