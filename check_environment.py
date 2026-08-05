"""
Standalone environment check -- verifies your local install works BEFORE
touching the network (no yfinance download needed).

Run this first, right after `pip install -r requirements.txt`:

    python check_environment.py

It trains each of the four architectures for a handful of steps on random
synthetic data. If this passes, your torch/PennyLane install is healthy and
you can move on to `run_local.py` with confidence that any errors there
come from the data/network side, not the model code.
"""
import sys
import traceback

import numpy as np
import torch


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
        print(f"[OK] {name}")
        return True
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        return False


def check_crbm():
    from src.models.crbm import CRBM, CRBMConfig
    torch.manual_seed(0)
    cfg = CRBMConfig(V=1, H=2, U=10, k=3, sigma=0.1, lr=1e-3)
    model = CRBM(cfg)
    u = torch.randn(64, cfg.U) * 0.1
    v = torch.randn(64, cfg.V) * 0.1
    for _ in range(5):
        mse = model.train_step(v, u)
    pred = model.forecast_autoregressive(u[:5], horizon=3)
    assert pred.shape == (5, 3), f"unexpected forecast shape {pred.shape}"
    print(f"  final recon MSE: {mse:.4f}, num params: {model.num_params()}")


def check_qcrbm():
    from src.models.qcrbm import QCRBM, QCRBMConfig
    torch.manual_seed(0)
    cfg = QCRBMConfig(V=1, H=4, U=10, k=2, sigma=0.1, lr=1e-3, L_Q=1, eta_Q=1e-3)
    model = QCRBM(cfg)
    u = torch.randn(16, cfg.U) * 0.1
    v = torch.randn(16, cfg.V) * 0.1
    for _ in range(2):
        recon_mse, q_loss = model.train_step(v, u)
    pred = model.mean_field_predict(u[:5])
    assert pred.shape == (5, 1), f"unexpected prediction shape {pred.shape}"
    print(f"  n_qubits: {model.n_qubits}, recon MSE: {recon_mse:.4f}, "
          f"quantum loss: {q_loss:.4f}, num params: {model.num_params()}")


def check_qfeatureqrbm():
    from src.models.qfeatureqrbm import QFeatureQRBM, QFeatureQRBMConfig
    torch.manual_seed(0)
    cfg = QFeatureQRBMConfig(V=1, H=2, L=10, n_layers=1, k=2)
    model = QFeatureQRBM(cfg)
    lag = torch.randn(16, cfg.L) * 0.1
    v = torch.randn(16, cfg.V) * 0.1
    for _ in range(2):
        recon_mse, q_loss = model.train_step(v, lag)
        model.end_epoch()
    pred = model.mean_field_predict(lag[:5])
    assert pred.shape == (5, 1), f"unexpected prediction shape {pred.shape}"
    print(f"  n_q qubits: {model.n_q}, recon MSE: {recon_mse:.4f}, "
          f"quantum loss: {q_loss:.4f}, num params: {model.num_params()}")


def check_qqrbm():
    from src.models.qqrbm import QQRBM, QQRBMConfig
    torch.manual_seed(0)
    cfg = QQRBMConfig(V=2, H=2, U_qq=3, L_Q=1, k=2)
    model = QQRBM(cfg)
    v = torch.zeros(16, cfg.V)
    v[:, 0] = torch.randn(16) * 0.1
    u_context = torch.randn(16, cfg.U_qq) * 0.1
    for _ in range(2):
        recon_mse, q_loss = model.train_step(v, u_context)
    pred = model.mean_field_predict(u_context[:5])
    assert pred.shape == (5, 2), f"unexpected prediction shape {pred.shape}"
    print(f"  n_wires: {model.n_wires}, recon MSE: {recon_mse:.4f}, "
          f"quantum loss: {q_loss:.4f}, num params: {model.num_params()}")


def check_stats():
    from src.stats import compare_all_to_reference, power_analysis
    import pandas as pd
    np.random.seed(0)
    n = 12
    df = pd.concat([
        pd.DataFrame({"ticker": ["A"] * n, "seed": range(n), "model": "CRBM",
                       "rmse": np.random.rand(n) * 0.1 + 0.5}),
        pd.DataFrame({"ticker": ["A"] * n, "seed": range(n), "model": "QCRBM",
                       "rmse": np.random.rand(n) * 0.1 + 0.52}),
    ], ignore_index=True)
    result = compare_all_to_reference(df, "CRBM", ["QCRBM"])
    assert len(result) == 1
    dmin = power_analysis(n=12)
    print(f"  comparison ran OK, minimum detectable effect at n=12: {dmin:.3f}")


def check_data_logic():
    from src.data import make_split, build_context_windows, prepare_dataset, AssetSplit, realized_vol
    import pandas as pd
    idx = pd.date_range("2015-01-01", periods=3000, freq="D")
    series = pd.Series(np.random.randn(3000) * 0.01, index=idx)
    train, val, test = make_split(series, train_size=500)
    split = AssetSplit("TEST", 500, train, val, test, realized_vol(train), realized_vol(test))
    ds = prepare_dataset(split, context_len=10)
    assert ds.u_train.shape[1] == 10
    print(f"  train/val/test shapes: {ds.u_train.shape}, {ds.u_val.shape}, {ds.u_test.shape}")


if __name__ == "__main__":
    print("Checking Python environment for the QCRBM real-data scaling study...")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import pennylane as qml
        print(f"PennyLane: {qml.__version__}")
    except ImportError:
        print("[FAIL] PennyLane is not installed -- run: pip install -r requirements.txt")
        sys.exit(1)

    results = {
        "Data pipeline logic": check("Data pipeline logic", check_data_logic),
        "Statistics module": check("Statistics module", check_stats),
        "CRBM (classical)": check("CRBM (classical)", check_crbm),
        "QCRBM (hybrid)": check("QCRBM (hybrid)", check_qcrbm),
        "QFeatureQRBM": check("QFeatureQRBM", check_qfeatureqrbm),
        "QQRBM (structural reference)": check("QQRBM (structural reference)", check_qqrbm),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\nAll checks passed. Your environment is ready -- run `python run_local.py` next.")
    else:
        print("\nSome checks failed. Fix these before running the full pipeline "
              "(paste the traceback above if you need help debugging).")
        sys.exit(1)
