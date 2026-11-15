PY ?= python
DATASETS = scifact nfcorpus fiqa arguana
BEIR = https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets

all: data run analysis

data:
	mkdir -p data && cd data && for d in $(DATASETS); do [ -d $$d ] || (curl -sSfLO $(BEIR)/$$d.zip && unzip -q $$d.zip); done

run:        ## stage 1: all retrievers on all datasets -> results/per_query.csv, bm25_grid.npz
	$(PY) -m reteval.run

analysis:   ## stage 2: statistics, models, figures -> results/tables.md, summary.json, fig_*.png
	$(PY) -m reteval.analysis

test:
	$(PY) -m pytest -q tests

.PHONY: all data run analysis test
