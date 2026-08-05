# Quantum Conditional Boltzmann Machines on Real Financial Time Series

A scaling study re-evaluating the four architectures from Hellstern et al. (2026),
*"Variational Quantum Conditional Boltzmann Machines for Time-Series Forecasting"*
(arXiv:2607.24065), on **real observed market data** instead of synthetic
Gaussian-Process data, across a **range of training-set sizes** (250 -> 4000 obs).

## Project structure

```
qcrbm_project/
├── README.md                  <- you are here
├── requirements.txt           <- pip dependencies
├── check_environment.py       <- run FIRST: verifies your local install works
├── run_local.py               <- run the full pipeline locally (no Colab needed)
├── docs/
│   └── project_overview.docx  <- full write-up: base paper, gaps, novelties, roadmap
├── src/
│   ├── data.py                <- data pipeline (download, returns, splits, diagnostics)
│   ├── stats.py                <- paired significance testing (Holm, Wilcoxon, power)
│   ├── train.py                <- training loops for all four architectures
│   └── models/
│       ├── crbm.py             <- classical Gaussian-Bernoulli CRBM
│       ├── qcrbm.py            <- hybrid quantum-classical QCRBM
│       ├── qfeatureqrbm.py     <- lag-feature quantum QRBM
│       └── qqrbm.py            <- full-register quantum-quantum QRBM
└── notebooks/
    └── main_colab.ipynb        <- run this in Google Colab, top to bottom
```

## Quickstart (local / any machine, no Colab)

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Verify your install BEFORE touching the network (trains all 4 models
#    on random synthetic data for a few steps -- no yfinance download needed)
python check_environment.py

# 3. Run the full pipeline (downloads real data, builds splits, runs a
#    smoke test, then the full scaling grid, stats, and figures)
python run_local.py

# Trim the run for speed on a first pass:
python run_local.py --models CRBM QCRBM --seeds 0 1 --epochs 10
```

`run_local.py` checkpoints after every single (asset, train_size, seed, model)
run to `results/scaling_grid_results.csv` -- if it's interrupted, just re-run
the same command and it resumes automatically, skipping completed work.

## Quickstart (Google Colab)

1. Upload this whole folder to Google Drive (e.g. `MyDrive/qcrbm_project`).
2. Open `notebooks/main_colab.ipynb` in Colab.
3. Run cells top to bottom. The notebook:
   - mounts Drive and adds `src/` to the Python path,
   - installs dependencies,
   - downloads and caches real market data for 7 diversified assets,
   - builds the fixed-eval-window / growing-training-prefix scaling splits,
   - trains all four architectures (CRBM, QCRBM, QFeatureQRBM, QQRBM) per
     (asset, training size, seed),
   - runs the Holm-corrected paired significance tests,
   - saves all results and figures back to Drive.

## Design notes (read before changing the data pipeline)

- **Unit of replication**: independent *assets* (not overlapping time slices of one
  series), to approximate the base paper's independent-series assumption behind
  the paired t-test / Holm correction. See `docs/project_overview.docx`,
  Drawback 1.
- **Scaling axis**: training-set size grows while the *evaluation window stays
  fixed* per asset, to avoid confounding "more data" with "different market
  regime". See Drawback 2 in the same document. Volatility/regime descriptors
  are computed alongside every training-size condition in `src/data.py`.
- **Hyperparameters**: scoped down from the base paper's full 13-experiment grid
  search to a small representative sweep (see `src/train.py` `DEFAULT_HPARAMS`),
  documented as a deliberate limitation.
- **QQRBM** is included as a *structural reference* only (fixed small register,
  no full hyperparameter search), exactly as the base paper itself treats it.

## Citation

If you build on this, cite the base paper:

> Hellstern, G., Maheshwari, D., Zaefferer, M., Braun, M., Döhler, T. (2026).
> Variational Quantum Conditional Boltzmann Machines for Time-Series Forecasting:
> Architectures, Symmetric Hyperparameter Evaluation, and a Nonlinear Benchmark.
> arXiv:2607.24065.
