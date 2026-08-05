"""
Data pipeline for the real-market scaling study.

Responsibilities:
  1. Download / cache price history for a diversified asset basket.
  2. Compute log returns and run ADF stationarity checks.
  3. Report an asset-correlation table (Drawback 1 mitigation: independence
     is approximate across assets, not exact -- we show the correlation
     structure explicitly rather than hiding it).
  4. Build the fixed-eval-window + growing-training-prefix split scheme used
     for the scaling study (Drawback 2 mitigation).
  5. Compute a realized-volatility / structural-break descriptor for every
     training-size condition, so any regime confound is visible and reportable.
  6. Build the lagged "context" tensors (u) that the CRBM family conditions on,
     matching the base paper's L=U=10 default context window.

All functions are pure / side-effect-light except `download_prices`, which
hits the network (yfinance) and optionally caches to disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:  # pragma: no cover
    adfuller = None


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

DEFAULT_TICKERS = ["AAPL", "JPM", "XOM", "SPY", "EURUSD=X", "GC=F", "BTC-USD"]

# Scaling study grid: number of *training* observations used, holding the
# evaluation window fixed per asset.
DEFAULT_TRAIN_SIZES = [250, 500, 1000, 2000, 4000]

# Context / lag window length U=L, matching the base paper's default (Sec 5.1).
DEFAULT_CONTEXT_LEN = 10

# Fraction of each asset's full history reserved as the fixed evaluation window.
DEFAULT_EVAL_FRAC = 0.15
# Fraction reserved as validation (drawn from the training pool, most recent
# portion of it), matching the base paper's 70/15/15 chronological split.
DEFAULT_VAL_FRAC = 0.15


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------

def download_prices(
    tickers: List[str] = None,
    cache_path: str = None,
    period: str = "max",
) -> pd.DataFrame:
    """Download adjusted close prices for `tickers`, aligned on a common index.

    If `cache_path` exists, load from there instead of hitting the network.
    """
    tickers = tickers or DEFAULT_TICKERS

    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import yfinance as yf

    frames = {}
    for t in tickers:
        df = yf.download(t, period=period, progress=False, auto_adjust=True)
        if df.empty:
            print(f"[warn] no data returned for {t}, skipping")
            continue
        frames[t] = df["Close"]

    prices = pd.concat(frames.values(), axis=1)
    prices.columns = list(frames.keys())
    prices = prices.dropna(how="all").ffill().dropna()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)

    return prices


def generate_synthetic_prices(
    tickers: List[str] = None,
    n_obs: int = 6000,
    seed: int = 0,
    cache_path: str = None,
) -> pd.DataFrame:
    """Generate proxy price series for environments without network access to
    a real market-data provider (e.g. yfinance blocked by a sandbox/firewall
    policy). NOT a substitute for real data -- only for exercising the full
    pipeline (splits, training, evaluation) end to end.

    Each series is a GBM-like random walk in log-price with a small amount of
    AR(1) autocorrelation and asset-specific drift/volatility, so it has the
    same *shape* (near-random-walk, heavy right-tail vol clustering absent)
    as real log-returns without pretending to model real markets.
    """
    tickers = tickers or DEFAULT_TICKERS
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n_obs)

    frames = {}
    for i, t in enumerate(tickers):
        drift = rng.uniform(-0.0002, 0.0003)
        vol = rng.uniform(0.008, 0.025)
        ar_coef = rng.uniform(-0.05, 0.10)

        noise = rng.normal(0, vol, size=n_obs)
        returns = np.zeros(n_obs)
        returns[0] = noise[0]
        for k in range(1, n_obs):
            returns[k] = drift + ar_coef * returns[k - 1] + noise[k]

        log_price = np.cumsum(returns) + 100.0
        frames[t] = pd.Series(np.exp(log_price), index=idx)

    prices = pd.concat(frames.values(), axis=1)
    prices.columns = list(frames.keys())

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_csv(cache_path)

    return prices


# ----------------------------------------------------------------------------
# Returns + stationarity
# ----------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def adf_report(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Augmented Dickey-Fuller stationarity test per asset."""
    if adfuller is None:
        raise ImportError("statsmodels is required for adf_report()")
    rows = {}
    for col in log_returns.columns:
        stat, pval, _, _, crit, _ = adfuller(log_returns[col].values)
        rows[col] = {
            "adf_stat": stat,
            "p_value": pval,
            "stationary_5pct": pval < 0.05,
            "crit_1pct": crit["1%"],
            "crit_5pct": crit["5%"],
        }
    return pd.DataFrame(rows).T


def correlation_table(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise return correlation -- report this alongside any independence
    claim (Drawback 1). Assets should be chosen so off-diagonal values are
    modest, but this is never exactly zero for real markets."""
    return log_returns.corr()


# ----------------------------------------------------------------------------
# Scaling-study splits
# ----------------------------------------------------------------------------

@dataclass
class AssetSplit:
    """Train/val/test split for one asset at one training-set size.

    The *test* window is fixed regardless of training_size (Drawback 2 fix):
    we only grow how far back the training prefix reaches, never what is
    evaluated on.
    """
    ticker: str
    train_size: int
    train: pd.Series
    val: pd.Series
    test: pd.Series
    realized_vol_train: float
    realized_vol_test: float


def make_split(
    series: pd.Series,
    train_size: int,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Chronological split with a FIXED test window and a training prefix of
    length `train_size` immediately preceding the val/test windows.

    Layout (oldest -> newest):
        [ ... unused history ... ][ train (train_size) ][ val ][ test ]

    The test window's absolute position never changes across train_size --
    only the left edge of `train` moves further back for larger train_size.
    """
    n = len(series)
    test_size = int(n * eval_frac)
    val_size = int(n * val_frac)

    test = series.iloc[-test_size:]
    val = series.iloc[-(test_size + val_size):-test_size]
    train_pool = series.iloc[: -(test_size + val_size)]

    if train_size > len(train_pool):
        raise ValueError(
            f"Requested train_size={train_size} exceeds available pool "
            f"({len(train_pool)} obs) for this asset. Reduce train_size or "
            f"choose a longer-history asset."
        )

    train = train_pool.iloc[-train_size:]
    return train, val, test


def realized_vol(series: pd.Series, annualize_factor: int = 252) -> float:
    """Simple realized volatility (annualized std of log returns)."""
    return float(series.std() * np.sqrt(annualize_factor))


def build_all_splits(
    log_returns: pd.DataFrame = None,
    tickers: List[str] = None,
    train_sizes: List[int] = None,
    cache_path: str = None,
) -> Dict[str, Dict[int, AssetSplit]]:
    """Build {ticker: {train_size: AssetSplit}} for every (asset, size) pair.

    Also returns, per split, the realized volatility of the training window
    -- this is the regime/volatility descriptor to report next to every
    scaling-curve point (Drawback 2 mitigation).
    """
    tickers = tickers or DEFAULT_TICKERS
    train_sizes = train_sizes or DEFAULT_TRAIN_SIZES

    if log_returns is None:
        prices = download_prices(tickers, cache_path=cache_path)
        log_returns = compute_log_returns(prices)

    out: Dict[str, Dict[int, AssetSplit]] = {}
    for t in log_returns.columns:
        out[t] = {}
        for n in train_sizes:
            try:
                train, val, test = make_split(log_returns[t], n)
            except ValueError as e:
                print(f"[skip] {t} @ train_size={n}: {e}")
                continue
            out[t][n] = AssetSplit(
                ticker=t,
                train_size=n,
                train=train,
                val=val,
                test=test,
                realized_vol_train=realized_vol(train),
                realized_vol_test=realized_vol(test),
            )
    return out


def regime_diagnostics_table(splits: Dict[str, Dict[int, AssetSplit]]) -> pd.DataFrame:
    """Flatten the volatility descriptors for every (asset, train_size) into a
    single reportable table -- drop this straight into the paper's results
    section next to the scaling-curve RMSE table."""
    rows = []
    for ticker, by_size in splits.items():
        for n, split in by_size.items():
            rows.append({
                "ticker": ticker,
                "train_size": n,
                "train_start": split.train.index[0],
                "train_end": split.train.index[-1],
                "realized_vol_train": split.realized_vol_train,
                "realized_vol_test": split.realized_vol_test,
                "vol_ratio_train_over_test": split.realized_vol_train / split.realized_vol_test,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Context-window (lag) tensors for the CRBM family
# ----------------------------------------------------------------------------

def build_context_windows(
    series: pd.Series,
    context_len: int = DEFAULT_CONTEXT_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D return series into (u, v) pairs for conditional models:
    u_t = (y_{t-L}, ..., y_{t-1})   [context / lag window]
    v_t = y_t                       [target visible value]

    Returns arrays of shape (n_samples, context_len) and (n_samples, 1).
    """
    values = series.values.astype(np.float32)
    n = len(values)
    if n <= context_len:
        raise ValueError(f"series too short ({n}) for context_len={context_len}")

    u = np.stack([values[i: i + context_len] for i in range(n - context_len)])
    v = values[context_len:].reshape(-1, 1)
    return u, v


def fit_standard_scaler(train_values: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) fit on TRAINING data only -- never fit on val/test,
    matching the base paper's leakage-safe protocol (Sec 5.1)."""
    mean = float(train_values.mean())
    std = float(train_values.std())
    std = std if std > 1e-8 else 1.0
    return mean, std


def apply_scaler(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


def inverse_scaler(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


@dataclass
class PreparedDataset:
    """Fully prepared, scaled, windowed tensors ready for model training,
    for one (ticker, train_size) condition."""
    ticker: str
    train_size: int
    u_train: np.ndarray
    v_train: np.ndarray
    u_val: np.ndarray
    v_val: np.ndarray
    u_test: np.ndarray
    v_test: np.ndarray
    scaler_mean: float
    scaler_std: float


def prepare_dataset(
    split: AssetSplit,
    context_len: int = DEFAULT_CONTEXT_LEN,
) -> PreparedDataset:
    """Build scaled (u, v) tensors for train/val/test from an AssetSplit.

    The scaler is fit on train only and applied to val/test (no leakage),
    matching the base paper's protocol exactly.
    """
    # Concatenate a small tail of train onto val/test so the first `context_len`
    # points of val/test still have a valid lag window drawn from real history.
    full = pd.concat([split.train, split.val, split.test])

    u_all, v_all = build_context_windows(full, context_len=context_len)

    n_train = len(split.train) - context_len
    n_val = len(split.val)
    n_test = len(split.test)

    n_train = max(n_train, 0)

    u_train, v_train = u_all[:n_train], v_all[:n_train]
    u_val, v_val = u_all[n_train:n_train + n_val], v_all[n_train:n_train + n_val]
    u_test, v_test = u_all[n_train + n_val:n_train + n_val + n_test], v_all[n_train + n_val:n_train + n_val + n_test]

    mean, std = fit_standard_scaler(v_train)

    def scale_pair(u, v):
        return apply_scaler(u, mean, std).astype(np.float32), apply_scaler(v, mean, std).astype(np.float32)

    u_train, v_train = scale_pair(u_train, v_train)
    u_val, v_val = scale_pair(u_val, v_val)
    u_test, v_test = scale_pair(u_test, v_test)

    return PreparedDataset(
        ticker=split.ticker,
        train_size=split.train_size,
        u_train=u_train, v_train=v_train,
        u_val=u_val, v_val=v_val,
        u_test=u_test, v_test=v_test,
        scaler_mean=mean, scaler_std=std,
    )


def prepare_all(
    splits: Dict[str, Dict[int, AssetSplit]],
    context_len: int = DEFAULT_CONTEXT_LEN,
) -> Dict[str, Dict[int, PreparedDataset]]:
    out: Dict[str, Dict[int, PreparedDataset]] = {}
    for ticker, by_size in splits.items():
        out[ticker] = {}
        for n, split in by_size.items():
            try:
                out[ticker][n] = prepare_dataset(split, context_len=context_len)
            except ValueError as e:
                print(f"[skip] {ticker} @ {n}: {e}")
    return out


if __name__ == "__main__":
    prices = download_prices(cache_path="data/prices_raw.csv")
    log_returns = compute_log_returns(prices)
    print("ADF stationarity report:")
    print(adf_report(log_returns))
    print("\nCorrelation table:")
    print(correlation_table(log_returns).round(2))

    splits = build_all_splits(log_returns)
    diag = regime_diagnostics_table(splits)
    print("\nRegime diagnostics (head):")
    print(diag.head())

    prepared = prepare_all(splits)
    example = prepared[list(prepared.keys())[0]][DEFAULT_TRAIN_SIZES[0]]
    print(f"\nExample prepared dataset ({example.ticker}, N={example.train_size}):")
    print("u_train", example.u_train.shape, "v_train", example.v_train.shape)
    print("u_val", example.u_val.shape, "u_test", example.u_test.shape)
