"""
Training loop for the full scaling study: runs all four architectures over
the (asset x train_size x seed) grid and records test RMSE, mirroring the
base paper's evaluation protocol (RMSE on the original data scale, Sec 5.1).

Hyperparameters are SCOPED DOWN from the base paper's full 13-experiment
symmetric grid search (see docs/project_overview.docx) to a small
representative sweep, documented explicitly as a deliberate limitation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .data import PreparedDataset, DEFAULT_TRAIN_SIZES, DEFAULT_CONTEXT_LEN
from .models.crbm import CRBM, CRBMConfig
from .models.qcrbm import QCRBM, QCRBMConfig
from .models.qfeatureqrbm import QFeatureQRBM, QFeatureQRBMConfig
from .models.qqrbm import QQRBM, QQRBMConfig


# ----------------------------------------------------------------------------
# Default (scoped-down) hyperparameters
# ----------------------------------------------------------------------------
# These are small, representative choices selected on one asset/size during
# Days 6-7 of the roadmap, then reused across the full scaling grid -- NOT a
# full symmetric search like the base paper's Experiments 1/2/11/12. This is
# a deliberate scope reduction to fit the two-week timeline; see
# docs/project_overview.docx.

DEFAULT_HPARAMS = {
    "crbm": dict(H=2, k=3, sigma=0.1, lr=1e-3, weight_decay=0.0),
    "qcrbm": dict(H=4, k=3, sigma=0.1, lr=1e-3, L_Q=2, alpha_init=3.0, eta_Q=1e-3, lambda_Q=1e-3, k_Q=1),
    "qfeatureqrbm": dict(H=2, L=10, k=3, sigma=0.1, lr=1e-3, n_layers=1, lambda_align=1.0, eta_Q=1e-3, lambda_Q=1e-3),
    "qqrbm": dict(V=2, H=2, U_qq=3, L_Q=1, k=3, sigma=0.1, lr=1e-3, eta_Q=1e-3, lambda_Q=1e-3),
}

DEFAULT_SEEDS = [0, 1, 2, 3]   # 4 seeds, matching the base paper's protocol
DEFAULT_EPOCHS = 20            # matches the base paper's main-comparison budget (Sec 5.4-5.5)
DEFAULT_BATCH_SIZE = 32


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of test points where the predicted return has the same sign
    (up/down direction) as the actual return -- a classification-style
    "accuracy" metric read alongside RMSE. A coin flip scores ~0.5; real
    financial return series are close to a random walk, so even a
    genuinely-useful model typically lands in the 0.50-0.56 range. Treat any
    value near 1.0 as a red flag for leakage/overfitting rather than success.
    """
    pred_sign = np.sign(pred).ravel()
    target_sign = np.sign(target).ravel()
    return float(np.mean(pred_sign == target_sign))


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def batches(u: torch.Tensor, v: torch.Tensor, batch_size: int, generator: torch.Generator):
    n = u.shape[0]
    idx = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        if len(sel) < 2:
            continue
        yield u[sel], v[sel]


# ----------------------------------------------------------------------------
# Per-model training routines (return test-set RMSE, in the ORIGINAL scale)
# ----------------------------------------------------------------------------

def _inverse(pred_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return pred_scaled * std + mean


def train_and_eval_crbm(ds: PreparedDataset, seed: int, epochs: int = DEFAULT_EPOCHS,
                         batch_size: int = DEFAULT_BATCH_SIZE, hparams: dict = None) -> dict:
    hp = {**DEFAULT_HPARAMS["crbm"], **(hparams or {})}
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    cfg = CRBMConfig(V=1, U=ds.u_train.shape[1], seed=seed, **hp)
    model = CRBM(cfg)

    u_train, v_train = to_tensor(ds.u_train), to_tensor(ds.v_train)
    u_test, v_test = to_tensor(ds.u_test), to_tensor(ds.v_test)

    for epoch in range(epochs):
        for u_b, v_b in batches(u_train, v_train, batch_size, g):
            model.train_step(v_b, u_b)

    with torch.no_grad():
        pred = model.mean_field_predict(u_test).numpy()

    pred_orig = _inverse(pred, ds.scaler_mean, ds.scaler_std)
    target_orig = _inverse(ds.v_test, ds.scaler_mean, ds.scaler_std)
    return {
        "rmse": rmse(pred_orig, target_orig),
        "directional_accuracy": directional_accuracy(pred_orig, target_orig),
        "num_params": model.num_params(),
    }


def train_and_eval_qcrbm(ds: PreparedDataset, seed: int, epochs: int = DEFAULT_EPOCHS,
                          batch_size: int = DEFAULT_BATCH_SIZE, hparams: dict = None) -> dict:
    hp = {**DEFAULT_HPARAMS["qcrbm"], **(hparams or {})}
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    cfg = QCRBMConfig(V=1, U=ds.u_train.shape[1], seed=seed, **hp)
    model = QCRBM(cfg)

    u_train, v_train = to_tensor(ds.u_train), to_tensor(ds.v_train)
    u_test, v_test = to_tensor(ds.u_test), to_tensor(ds.v_test)

    for epoch in range(epochs):
        for u_b, v_b in batches(u_train, v_train, batch_size, g):
            model.train_step(v_b, u_b)

    with torch.no_grad():
        pred = model.mean_field_predict(u_test).numpy()

    pred_orig = _inverse(pred, ds.scaler_mean, ds.scaler_std)
    target_orig = _inverse(ds.v_test, ds.scaler_mean, ds.scaler_std)
    return {
        "rmse": rmse(pred_orig, target_orig),
        "directional_accuracy": directional_accuracy(pred_orig, target_orig),
        "num_params": model.num_params(),
    }


def train_and_eval_qfeatureqrbm(ds: PreparedDataset, seed: int, epochs: int = DEFAULT_EPOCHS,
                                 batch_size: int = DEFAULT_BATCH_SIZE, hparams: dict = None) -> dict:
    hp = {**DEFAULT_HPARAMS["qfeatureqrbm"], **(hparams or {})}
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    cfg = QFeatureQRBMConfig(V=1, U=ds.u_train.shape[1], seed=seed, **hp)
    model = QFeatureQRBM(cfg)

    # QFeatureQRBM uses a dedicated lag vector `l`; we reuse the same context
    # window `u` as the lag vector (both are the last L=U=10 values, matching
    # the base paper's fixed L=U=10 default, Sec 5.1).
    u_train, v_train = to_tensor(ds.u_train), to_tensor(ds.v_train)
    u_test, v_test = to_tensor(ds.u_test), to_tensor(ds.v_test)

    for epoch in range(epochs):
        for u_b, v_b in batches(u_train, v_train, batch_size, g):
            model.train_step(v_b, u_b)
        model.end_epoch()

    with torch.no_grad():
        pred = model.mean_field_predict(u_test).numpy()

    pred_orig = _inverse(pred, ds.scaler_mean, ds.scaler_std)
    target_orig = _inverse(ds.v_test, ds.scaler_mean, ds.scaler_std)
    return {
        "rmse": rmse(pred_orig, target_orig),
        "directional_accuracy": directional_accuracy(pred_orig, target_orig),
        "num_params": model.num_params(),
    }


def train_and_eval_qqrbm(ds: PreparedDataset, seed: int, epochs: int = DEFAULT_EPOCHS,
                          batch_size: int = DEFAULT_BATCH_SIZE, hparams: dict = None) -> dict:
    hp = {**DEFAULT_HPARAMS["qqrbm"], **(hparams or {})}
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    cfg = QQRBMConfig(seed=seed, **hp)
    model = QQRBM(cfg)

    # QQRBM restricts context to the last U_qq lag values (natural size,
    # Sec 5.5) and needs V>=2 visible qubits, so we pad the scalar target
    # with a zero dummy dimension.
    u_train_full, v_train = to_tensor(ds.u_train), to_tensor(ds.v_train)
    u_test_full, v_test = to_tensor(ds.u_test), to_tensor(ds.v_test)

    u_train = u_train_full[:, -cfg.U_qq:]
    u_test = u_test_full[:, -cfg.U_qq:]

    def pad_v(v):
        out = torch.zeros(v.shape[0], cfg.V)
        out[:, 0] = v[:, 0]
        return out

    v_train_p = pad_v(v_train)

    for epoch in range(epochs):
        for u_b, v_b in batches(u_train, v_train_p, batch_size, g):
            model.train_step(v_b, u_b)

    with torch.no_grad():
        pred = model.mean_field_predict(u_test).numpy()[:, 0:1]

    pred_orig = _inverse(pred, ds.scaler_mean, ds.scaler_std)
    target_orig = _inverse(ds.v_test, ds.scaler_mean, ds.scaler_std)
    return {
        "rmse": rmse(pred_orig, target_orig),
        "directional_accuracy": directional_accuracy(pred_orig, target_orig),
        "num_params": model.num_params(),
    }


MODEL_TRAINERS = {
    "CRBM": train_and_eval_crbm,
    "QCRBM": train_and_eval_qcrbm,
    "QFeatureQRBM": train_and_eval_qfeatureqrbm,
    "QQRBM": train_and_eval_qqrbm,
}


# ----------------------------------------------------------------------------
# Full grid runner
# ----------------------------------------------------------------------------

def run_scaling_grid(
    prepared: Dict[str, Dict[int, PreparedDataset]],
    models: List[str] = None,
    seeds: List[int] = None,
    epochs: int = DEFAULT_EPOCHS,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    """Train every model on every (asset, train_size, seed) combination.

    Checkpoints results to `checkpoint_path` after every run (important on
    Colab, where sessions can disconnect mid-grid) and resumes from it if the
    file already exists and contains completed rows.
    """
    models = models or list(MODEL_TRAINERS.keys())
    seeds = seeds or DEFAULT_SEEDS

    done = set()
    rows = []
    if checkpoint_path:
        try:
            existing = pd.read_csv(checkpoint_path)
            rows = existing.to_dict("records")
            done = set(
                (r["ticker"], r["train_size"], r["seed"], r["model"]) for r in rows
            )
            print(f"Resumed {len(done)} completed runs from {checkpoint_path}")
        except FileNotFoundError:
            pass

    total = sum(len(sizes) for sizes in prepared.values()) * len(seeds) * len(models)
    completed = len(done)

    for ticker, by_size in prepared.items():
        for train_size, ds in by_size.items():
            for seed in seeds:
                for model_name in models:
                    key = (ticker, train_size, seed, model_name)
                    if key in done:
                        continue
                    t0 = time.time()
                    try:
                        result = MODEL_TRAINERS[model_name](ds, seed, epochs=epochs)
                    except Exception as e:
                        print(f"[FAIL] {key}: {e}")
                        result = {"rmse": np.nan, "directional_accuracy": np.nan, "num_params": np.nan}
                    elapsed = time.time() - t0

                    row = {
                        "ticker": ticker,
                        "train_size": train_size,
                        "seed": seed,
                        "model": model_name,
                        "rmse": result["rmse"],
                        "directional_accuracy": result.get("directional_accuracy", np.nan),
                        "num_params": result["num_params"],
                        "elapsed_s": elapsed,
                    }
                    rows.append(row)
                    completed += 1
                    print(f"[{completed}/{total}] {key} -> RMSE={result['rmse']:.6g} "
                          f"DirAcc={result.get('directional_accuracy', float('nan')):.3f} ({elapsed:.1f}s)")

                    if checkpoint_path:
                        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .data import build_all_splits, prepare_all

    splits = build_all_splits(train_sizes=[250, 500])
    prepared = prepare_all(splits)

    results = run_scaling_grid(
        prepared,
        models=["CRBM", "QCRBM"],
        seeds=[0, 1],
        epochs=5,
        checkpoint_path="results/scaling_grid_smoketest.csv",
    )
    print(results)
