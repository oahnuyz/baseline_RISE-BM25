"""BM25 indexing + retrieval over the BrowseComp-Plus 100k corpus.

Uses bm25s (pure-Python, fast BM25) with PyStemmer-backed English stemming.
The index is saved to disk so build is one-shot; eval scripts just load.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import bm25s
import Stemmer
import pyarrow.parquet as pq
from tqdm import tqdm


def _stemmer() -> Stemmer.Stemmer:
    return Stemmer.Stemmer("english")


def iter_corpus(parquet_path: Path) -> Iterable[tuple[str, str]]:
    """Yield (docid, text) pairs from the BrowseComp-Plus corpus parquet."""
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["docid", "text"])
        docids = table.column("docid").to_pylist()
        texts = table.column("text").to_pylist()
        for did, txt in zip(docids, texts):
            yield str(did), txt or ""


def load_corpus(parquet_path: Path) -> tuple[list[str], list[str]]:
    """Load all (docid, text) into two parallel lists. Returns (doc_ids, texts)."""
    doc_ids: list[str] = []
    texts: list[str] = []
    for did, txt in iter_corpus(parquet_path):
        doc_ids.append(did)
        texts.append(txt)
    return doc_ids, texts


def build_index(
    parquet_path: Path,
    save_dir: Path,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict:
    """Tokenize the corpus, build a BM25 index, save it + doc_ids to disk.

    Returns a small metadata dict (counts, params, paths).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading corpus from {parquet_path} ...")
    doc_ids, texts = load_corpus(parquet_path)
    print(f"  {len(doc_ids):,} docs, {sum(len(t) for t in texts)/1e9:.2f}GB raw text")

    print("tokenizing (stemmed, english stopwords) ...")
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(), show_progress=True)

    print(f"building BM25 (k1={k1}, b={b}) ...")
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(tokens, show_progress=True)

    index_dir = save_dir / "bm25_index"
    print(f"saving index to {index_dir} ...")
    retriever.save(str(index_dir))

    docids_path = save_dir / "doc_ids.json"
    docids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

    meta = {
        "n_docs": len(doc_ids),
        "k1": k1,
        "b": b,
        "index_dir": str(index_dir),
        "doc_ids_path": str(docids_path),
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done.")
    return meta


def load_index(save_dir: Path) -> tuple[bm25s.BM25, list[str]]:
    save_dir = Path(save_dir)
    retriever = bm25s.BM25.load(str(save_dir / "bm25_index"), load_corpus=False)
    doc_ids: list[str] = json.loads((save_dir / "doc_ids.json").read_text(encoding="utf-8"))
    return retriever, doc_ids


def retrieve(
    retriever: bm25s.BM25,
    doc_ids: Sequence[str],
    query: str,
    k: int = 1000,
) -> list[tuple[str, float]]:
    """Return [(docid, score)] sorted by score desc, top-k."""
    tokens = bm25s.tokenize([query], stopwords="en", stemmer=_stemmer(), show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, show_progress=False)
    # results shape: (n_queries=1, k) with internal doc indices
    out: list[tuple[str, float]] = []
    for idx, score in zip(results[0], scores[0]):
        out.append((doc_ids[int(idx)], float(score)))
    return out


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], ks: Sequence[int]) -> dict[int, float]:
    """Recall@k for each k in ks. retrieved is ranked list of docids; gold is the relevant set."""
    gold_set = set(gold)
    if not gold_set:
        return {k: 0.0 for k in ks}
    out: dict[int, float] = {}
    for k in ks:
        topk = set(retrieved[:k])
        hits = len(topk & gold_set)
        out[k] = hits / len(gold_set)
    return out
