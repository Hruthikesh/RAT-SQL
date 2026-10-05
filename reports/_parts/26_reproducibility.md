## 26. Reproducibility

* Environment: Python 3.13.3, PyTorch 2.14.0+cu126, transformers 5.17.0 (exact list in
  `requirements.txt`); each run stores its resolved config, parameter counts, library versions,
  device and seed (`run_info.json`).
* Determinism: preprocessing is deterministic; seeds are fixed for Python, NumPy and PyTorch;
  GPU training is not bitwise deterministic (CUDA atomics in scatter / index-put backward).
* Data provenance: official Spider archive (Google-Drive id in `scripts/download_data.py`),
  validated counts; strict database-disjoint splits verified in `data/processed/spider/stats.json`.
* Every table and plot is regenerated from saved metric / prediction files; commands are listed
  in `README.md` §10 and `results/README.md`.
* Checkpoints are not distributed (size); every run can be re-trained with the logged command.
