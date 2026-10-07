# Reproduce the toy

Use Python 3.12 and `requirements.txt`. The pinned training environment was tested on an Apple M2 Max CPU. Other platforms have not been checked. Installing dependencies needs package access; digits itself is bundled with scikit-learn.

## The short run

```sh
python run.py
```

This checks the actual modulator source, regenerates the four train/validation arrays and trains seed 12001 with G1/G8 for five epochs each. Expect minutes on a CPU; compilation time varies. It writes to a new `fresh_runs/` directory. It refuses an existing output path.

To execute the notebook without a Jupyter frontend:

```sh
python execute_notebook.py
```

The executed notebook is saved under `fresh_runs/`, leaving the checked-in notebook unchanged. It reads 750 saved epoch predictions, recomputes both cohort comparisons, checks pairing, runs source-level checks and trains the original first pair. Exact comparison with saved probability hashes is part of the final cell.

## More training, only when requested

```sh
# Prepare/check the supplement without training.
python run.py --study supplement --prepare-only

# Fixed seed 13001, G1/G8, twenty epochs each.
python run.py --study supplement

# All 40 original models, five epochs each.
python run.py --all

# All 50 supplemental models, 550 training epochs in total.
python run.py --study supplement --all
```

The full studies cost much more than a pair. The primary wrapper runs sequentially; the supplement uses two workers. Both retain the frozen seeds and budgets. Do not choose a better checkpoint or group after examining results. These commands do not overwrite the historical results.

## What is saved here

`results/epochs.csv` includes every epoch's prediction vector and accuracy. `pairing.json` includes initial-weight hashes, random-stream hashes and the supplement's internal encoder-key hashes. The notebook recomputes accuracy and compares pairing records. The exported hashes are evidence of correspondence, not a replacement for raw arrays.

`results/provenance.json` identifies the original 90 full run records by SHA-256. Their probability arrays and batch traces are kept in the local research archive, not this repository. New runs write full records again. The demo therefore does not pretend to re-audit omitted raw records.

`studies/` holds the original frozen protocols, source manifests and source snapshots. `src/run_grouped_*.py` is preserved byte-for-byte. Run it through `run.py`, which supplies the portable paths. Editing a frozen runner or model triggers a hash failure.

This repository uses JAX/ngclearn. snnTorch was an optional example in the supervisor's email, so I have not added an unused dependency or changed frameworks to match its name.
