# Convenience targets.  On Windows run the commands inside a shell with `make`
# (e.g. Git Bash) or copy the commands directly.
PY ?= python

.PHONY: install data synthetic preprocess preprocess-debug test debug train-small train-final seed2 beam evaluate ablations collect analysis attention graphs synthetic-attention notebooks reports clean

install:
	$(PY) -m pip install -r requirements.txt

data:                     ## download + validate Spider and generate the synthetic set
	$(PY) scripts/download_data.py --dataset all

synthetic:
	$(PY) scripts/download_data.py --dataset synthetic

preprocess:               ## Spider preprocessing (+ oracle execution check)
	$(PY) scripts/preprocess.py --config configs/small.yaml --oracle-exec

preprocess-debug:
	$(PY) scripts/preprocess.py --config configs/debug.yaml --oracle-exec

test:
	$(PY) -m pytest tests -q

debug: synthetic preprocess-debug   ## end-to-end run on the synthetic set (minutes, CPU ok)
	$(PY) scripts/train.py --config configs/debug.yaml --eval-splits dev

train-small:              ## the controlled FULL model
	$(PY) scripts/train.py --config configs/small.yaml --output-dir experiments/full_model/full --eval-splits dev

train-final:              ## final model (BERT-medium, 40 epochs), dev + test
	$(PY) scripts/train.py --config configs/full.yaml --eval-splits dev test

seed2:                    ## second seed of the controlled FULL model
	$(PY) scripts/train.py --config configs/small.yaml --set seed=43 --output-dir experiments/full_model/full_seed43 --eval-splits dev

beam:
	$(PY) scripts/evaluate.py --checkpoint experiments/full_model/final/best.pt --split dev --beam-size 5

evaluate:
	$(PY) scripts/evaluate.py --checkpoint experiments/full_model/full/best.pt --split dev

ablations:                ## all baselines, ablations and data-scaling runs (resumable)
	$(PY) scripts/run_ablation.py

collect:
	$(PY) scripts/collect_results.py

analysis:                 ## all post-training analyses (tables, errors, joins, attention, cross-checks, notebooks)
	$(PY) scripts/run_analyses.py

attention:
	$(PY) scripts/run_attention_analysis.py --models rat=experiments/full_model/full/best.pt \
	  vanilla=experiments/baseline/baseline_b_vanilla_transformer/best.pt

graphs:
	$(PY) scripts/build_schema_graphs.py --config configs/small.yaml

synthetic-attention:      ## vanilla vs relation-aware attention on a synthetic relational task
	$(PY) scripts/run_synthetic_relation_experiment.py

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

reports:                  ## assemble reports, README result tables and PROJECT_STATUS.md
	$(PY) -m pytest tests -p no:cacheprovider > results/metrics/test_results.txt
	$(PY) scripts/build_report.py
	$(PY) scripts/build_status.py

clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
