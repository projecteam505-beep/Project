"""
Local (non-Colab) end-to-end driver for the scaling study.

Run this from the project root after `pip install -r requirements.txt`:

    python run_local.py

It mirrors the Colab notebook step-for-step, but reads/writes plain local
paths instead of mounting Google Drive. It also runs a small smoke test
(1 asset, 2 models, 2 seeds, 2 epochs) before committing to the full grid,
so any environment issues (e.g. a PennyLane version mismatch) surface in
seconds rather than after a long run.

Every stage is checkpointed to results/*.csv, so if the script is
interrupted (Ctrl+C, crash, laptop sleep), just re-run `python run_local.py`
-- run_scaling_grid() resumes automatically from its checkpoint and skips
anything already completed.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.data import (
    download_prices, generate_research_calibrated_prices, stylized_facts_report,
    compute_log_returns, adf_report, correlation_table, build_all_splits,
    regime_diagnostics_table, prepare_all, DEFAULT_TICKERS, DEFAULT_TRAIN_SIZES,
)
from src.train import run_scaling_grid, DEFAULT_SEEDS, DEFAULT_EPOCHS
from src.stats import compare_all_to_reference, power_analysis, window_information_ceiling


def parse_args():
    p = argparse.ArgumentParser(description="Run the QCRBM real-data scaling study locally.")
    p.add_argument("--models", nargs="+",
                    default=["CRBM", "QCRBM", "QFeatureQRBM", "QQRBM"],
                    help="Which architectures to train. Trim this list for a faster first pass, "
                         "e.g. --models CRBM QCRBM")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="Random seeds to run per (asset, train_size, model).")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help="Training epochs per run.")
    p.add_argument("--train-sizes", nargs="+", type=int, default=DEFAULT_TRAIN_SIZES,
                    help="Training-set sizes for the scaling axis.")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                    help="Asset tickers (yfinance symbols).")
    p.add_argument("--skip-smoketest", action="store_true",
                    help="Skip the small pre-flight smoke test and go straight to the full grid.")
    p.add_argument("--synthetic", action="store_true",
                    help="Use a GJR-GARCH(1,1) + Student-t proxy price series (calibrated to the "
                         "stylized facts documented in docs/data_research.md) instead of downloading "
                         "real data via yfinance. Use this when the network is restricted (e.g. a "
                         "sandboxed CI environment) and no real market data is reachable. Results "
                         "from this mode are NOT real market results -- only useful for exercising "
                         "the pipeline and reproducing the statistical *shape* of real returns.")
    p.add_argument("--hparam-search", action="store_true",
                    help="Search a small per-model hyperparameter grid (src/train.py "
                         "HPARAM_SEARCH_SPACE) and keep whichever candidate has the best "
                         "validation RMSE, instead of the single fixed DEFAULT_HPARAMS guess. "
                         "Multiplies runtime by roughly the number of candidates per model.")
    p.add_argument("--quiet-epochs", action="store_true",
                    help="Suppress the per-epoch validation RMSE/accuracy lines during training "
                         "(printed by default). Useful for very large grids where per-epoch logging "
                         "would be overwhelming.")
    p.add_argument("--results-dir", default="results",
                    help="Where to write all output CSVs / figures.")
    p.add_argument("--data-dir", default="data",
                    help="Where to cache downloaded price data.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/figures", exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    print("=" * 70)
    if args.synthetic:
        print("STEP 1: Generating research-calibrated SYNTHETIC price data (--synthetic set -- "
              "NOT real market data; see docs/data_research.md)")
    else:
        print("STEP 1: Downloading / loading market data")
    print("=" * 70)
    if args.synthetic:
        # Cache under a run-scoped filename, distinct from the canonical
        # full-universe data/prices_synthetic.csv (generated once for all
        # DEFAULT_TICKERS and delivered/committed separately) -- otherwise a
        # run against a ticker subset (e.g. --tickers AAPL) would silently
        # overwrite the canonical dataset file with just that subset.
        prices = generate_research_calibrated_prices(
            args.tickers, cache_path=f"{args.data_dir}/prices_synthetic_run.csv"
        )
    else:
        prices = download_prices(args.tickers, cache_path=f"{args.data_dir}/prices_raw.csv")
    log_returns = compute_log_returns(prices)
    log_returns.to_csv(f"{args.data_dir}/log_returns_run.csv")
    print(f"Assets: {list(log_returns.columns)}")
    print(f"History: {log_returns.index.min()} -> {log_returns.index.max()} "
          f"({len(log_returns)} observations)")

    print("\nADF stationarity report:")
    adf_df = adf_report(log_returns)
    adf_df.to_csv(f"{args.results_dir}/adf_report.csv")
    print(adf_df)

    print("\nAsset correlation table (report alongside any independence claim):")
    corr = correlation_table(log_returns)
    corr.to_csv(f"{args.results_dir}/asset_correlation.csv")
    print(corr.round(2))

    if args.synthetic:
        print("\nStylized-facts validation (does the synthetic data actually reproduce the "
              "documented properties of real returns? see docs/data_research.md):")
        facts_df = stylized_facts_report(log_returns)
        facts_df.to_csv(f"{args.results_dir}/stylized_facts_report.csv")
        print(facts_df.round(4))

    # ------------------------------------------------------------------
    # Step 2: Scaling-study splits (fixed eval window, growing training prefix)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Building scaling-study splits")
    print("=" * 70)
    splits = build_all_splits(
        log_returns=log_returns,
        tickers=list(log_returns.columns),
        train_sizes=args.train_sizes,
    )
    diag = regime_diagnostics_table(splits)
    diag.to_csv(f"{args.results_dir}/regime_diagnostics.csv", index=False)
    print(diag)

    prepared = prepare_all(splits)
    for ticker, by_size in prepared.items():
        print(f"  {ticker}: train sizes {sorted(by_size.keys())}")

    if not prepared:
        raise RuntimeError(
            "No datasets were prepared -- check that your assets have enough "
            "history for the requested --train-sizes."
        )

    # ------------------------------------------------------------------
    # Step 3: Smoke test (skip with --skip-smoketest)
    # ------------------------------------------------------------------
    if not args.skip_smoketest:
        print("\n" + "=" * 70)
        print("STEP 3: Smoke test (1 asset, CRBM + QCRBM, 2 seeds, 2 epochs)")
        print("=" * 70)
        first_ticker = list(prepared.keys())[0]
        first_size = min(prepared[first_ticker].keys())
        smoke = {first_ticker: {first_size: prepared[first_ticker][first_size]}}

        smoke_models = [m for m in ["CRBM", "QCRBM"] if m in args.models] or ["CRBM"]
        smoke_results = run_scaling_grid(
            smoke, models=smoke_models, seeds=args.seeds[:2] or [0], epochs=2,
            checkpoint_path=f"{args.results_dir}/smoketest.csv",
            verbose_epochs=not args.quiet_epochs,
        )
        print(smoke_results)
        print("Smoke test passed -- proceeding to the full grid.\n")
    else:
        print("\n[skipping smoke test per --skip-smoketest]\n")

    # ------------------------------------------------------------------
    # Step 4: Full grid
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 4: Full scaling grid "
          f"(models={args.models}, seeds={args.seeds}, epochs={args.epochs}, "
          f"hparam_search={args.hparam_search})")
    print("=" * 70)
    results = run_scaling_grid(
        prepared,
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        checkpoint_path=f"{args.results_dir}/scaling_grid_results.csv",
        use_hparam_search=args.hparam_search,
        verbose_epochs=not args.quiet_epochs,
    )

    print("\nPer-model summary (mean +/- SD across ticker/train_size/seed runs):")
    summary_cols = ["rmse", "directional_accuracy", "directional_accuracy_deadzone50", "deadzone_coverage"]
    summary_metrics = results.groupby("model")[summary_cols].agg(["mean", "std"])
    print(summary_metrics)
    print("\nNote: directional_accuracy is sign(pred) == sign(actual) on real-scale returns. "
          "~0.50 = coin flip; real financial series are close to a random walk, so this rarely "
          "exceeds the high-0.5s for a genuinely non-leaky model.")

    # ------------------------------------------------------------------
    # Step 5: Statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Holm-corrected paired significance tests")
    print("=" * 70)
    candidate_models = [m for m in args.models if m != "CRBM"]
    all_comparisons = []
    for train_size in sorted(results["train_size"].unique()):
        sub = results[results["train_size"] == train_size]
        if "CRBM" not in sub["model"].values:
            continue
        comp = compare_all_to_reference(
            sub, reference_model="CRBM",
            candidate_models=candidate_models,
        )
        comp.insert(0, "train_size", train_size)
        all_comparisons.append(comp)

    if all_comparisons:
        comparison_df = pd.concat(all_comparisons, ignore_index=True)
        comparison_df.to_csv(f"{args.results_dir}/significance_tests.csv", index=False)
        print(comparison_df)

        n_per_comparison = (
            results.groupby("train_size")["seed"].nunique().iloc[0]
            * results["ticker"].nunique()
        )
        dmin = power_analysis(n=n_per_comparison)
        print(f"\nApprox. n per comparison: {n_per_comparison}")
        print(f"Minimum detectable effect size (power=0.8, alpha=0.05): {dmin:.3f}")
    else:
        print("Not enough data to run significance tests (need CRBM as reference).")

    # ------------------------------------------------------------------
    # Step 6: Window-information-ceiling diagnostic
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Window-information-ceiling diagnostic (noise-floor check)")
    print("=" * 70)
    ceiling_rows = []
    for ticker, by_size in prepared.items():
        ds = by_size[max(by_size.keys())]
        ceiling = window_information_ceiling(ds.u_train, ds.v_train)
        ceiling_rows.append({
            "ticker": ticker,
            "nonlinear_window_floor_rmse_scaled": ceiling,
            "nonlinear_window_floor_rmse_original_scale": ceiling * ds.scaler_std,
        })
    ceiling_df = pd.DataFrame(ceiling_rows)
    ceiling_df.to_csv(f"{args.results_dir}/window_information_ceiling.csv", index=False)
    print(ceiling_df)

    # ------------------------------------------------------------------
    # Step 7: Scaling-curve figure
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: Scaling-curve figure")
    print("=" * 70)
    try:
        import matplotlib.pyplot as plt

        summary = results.groupby(["model", "train_size"])["rmse"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in summary["model"].unique():
            sub = summary[summary["model"] == model].sort_values("train_size")
            ax.errorbar(sub["train_size"], sub["mean"], yerr=sub["std"],
                        marker="o", label=model, capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Training-set size (log scale)")
        ax.set_ylabel("Test RMSE (mean ± SD across assets & seeds)")
        ax.set_title("Quantum vs. Classical Scaling Curve on Real Market Data")
        ax.legend()
        plt.tight_layout()
        fig_path = f"{args.results_dir}/figures/scaling_curve.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Saved figure to {fig_path}")
    except Exception as e:
        print(f"[warn] could not generate figure: {e}")

    print("\n" + "=" * 70)
    print(f"DONE. All results are in ./{args.results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
