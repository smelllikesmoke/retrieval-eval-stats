"""Loading BEIR datasets (corpus, queries, graded relevance judgements)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATASETS = ("scifact", "nfcorpus", "fiqa", "arguana")


@dataclass
class Dataset:
    name: str
    doc_ids: list[str]
    texts: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]  # test split: qid -> {doc_id: grade > 0}

    @property
    def qids(self) -> list[str]:
        return sorted(self.qrels)

    @property
    def exclude_self(self) -> bool:
        # ArguAna queries are themselves corpus documents; BEIR removes identical ids.
        return self.name == "arguana"


def load(name: str) -> Dataset:
    root = DATA / name
    doc_ids, texts = [], []
    with open(root / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            doc_ids.append(d["_id"])
            texts.append(f'{d.get("title", "")} {d["text"]}'.strip())
    queries = {}
    with open(root / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / "test.tsv") as f:
        next(f)
        for line in f:
            qid, did, grade = line.rstrip("\n").split("\t")
            if int(grade) > 0:
                qrels.setdefault(qid, {})[did] = int(grade)
    return Dataset(name, doc_ids, texts, queries, qrels)
