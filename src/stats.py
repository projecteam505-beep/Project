"""
Statistical testing module, mirroring Section 5.3 of Hellstern et al. (2026):

  - Paired two-sided t-test as the primary inference tool.
  - Holm-Bonferroni correction across all simultaneous comparisons in a
    given experiment (family-wise error rate control).
  - Wilcoxon signed-rank test as a robustness check (works down to n>=5).
  - Shapiro-Wilk normality check on the paired differences, to flag when the
    Wilcoxon result (rather than the t-test) should be treated as primary.
  - A post-hoc power analysis (Cohen's d, minimum detectable effect at given
    n, alpha, power) -- reported explicitly rather than silently omitted, per
    the base paper's "n=12 -> dmin~0.89" transparency (Sec 5.1).

Independence caveat (Drawback 1): in this real-market replication, "paired
observations" are (asset x seed) pairs rather than the base paper's
(independently-generated-series x seed) pairs. Report the asset correlation
table from src/data.py alongside any significance claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PairedTestResult:
    model_a: str
    model_b: str
    n: int
    mean_diff: float
    t_stat: float
    p_value: float
    p_holm: float
    wilcoxon_stat: float
    wilcoxon_p: float
    shapiro_p: float
    cohens_d: float


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for a paired/one-sample design: mean difference / std of differences."""
    sd = diffs.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(diffs.mean() / sd)


def paired_comparison(a: np.ndarray, b: np.ndarray, model_a: str, model_b: str) -> PairedTestResult:
    """Compare two paired samples (same asset/seed conditions), matching the
    base paper's per-comparison statistics (Sec 5.3)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a - b
    n = len(diffs)

    t_stat, p_value = stats.ttest_rel(a, b)

    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a, b)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan

    try:
        _, shapiro_p = stats.shapiro(diffs)
    except ValueError:
        shapiro_p = np.nan

    d = cohens_d_paired(diffs)

    return PairedTestResult(
        model_a=model_a, model_b=model_b, n=n,
        mean_diff=float(diffs.mean()),
        t_stat=float(t_stat), p_value=float(p_value), p_holm=np.nan,
        wilcoxon_stat=float(wilcoxon_stat), wilcoxon_p=float(wilcoxon_p),
        shapiro_p=float(shapiro_p), cohens_d=d,
    )


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    same order as the input."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)

    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = n - rank
        adj = p_values[idx] * factor
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def compare_all_to_reference(
    results_df: pd.DataFrame,
    reference_model: str,
    candidate_models: List[str],
    value_col: str = "rmse",
    group_cols: Tuple[str, ...] = ("ticker", "seed"),
) -> pd.DataFrame:
    """Run paired comparisons of every candidate model against `reference_model`,
    matched on (ticker, seed) so the pairing is valid, then apply Holm
    correction across all candidate comparisons in this call (one "family").

    Returns one row per candidate model with p_value, p_holm, wilcoxon_p,
    shapiro_p, cohens_d, mean_diff, n.
    """
    pivot = results_df.pivot_table(
        index=list(group_cols), columns="model", values=value_col
    )

    rows = []
    raw_p = []
    for cand in candidate_models:
        if cand not in pivot.columns or reference_model not in pivot.columns:
            continue
        sub = pivot[[reference_model, cand]].dropna()
        result = paired_comparison(
            sub[cand].values, sub[reference_model].values,
            model_a=cand, model_b=reference_model,
        )
        rows.append(result)
        raw_p.append(result.p_value)

    if raw_p:
        adjusted = holm_bonferroni(raw_p)
        for r, p_adj in zip(rows, adjusted):
            r.p_holm = p_adj

    return pd.DataFrame([vars(r) for r in rows])


def power_analysis(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable paired effect size (two-sided) at given n, alpha,
    power, matching the base paper's Sec 5.1 report (dmin ~ 0.89 at n=12)."""
    from scipy.stats import nct

    # Solve for d such that a paired t-test with df=n-1 has the requested
    # power, using a noncentral-t search (mirrors statsmodels' TTestPower but
    # avoids the extra dependency).
    df = n - 1
    t_alpha = stats.t.ppf(1 - alpha / 2, df)

    def power_at_d(d):
        ncp = d * np.sqrt(n)
        return 1 - nct.cdf(t_alpha, df, ncp) + nct.cdf(-t_alpha, df, ncp)

    lo, hi = 0.0, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_at_d(mid) < power:
            lo = mid
        else:
            hi = mid
    return round(hi, 4)


def window_information_ceiling(
    u: np.ndarray, v: np.ndarray, n_estimators: int = 200, random_state: int = 0
) -> float:
    """Empirical estimate of the best achievable RMSE from the context window
    alone (a nonlinear-window floor), following the base paper's Sec 5.5
    diagnostic: fit a gradient-boosted regressor of v on u and report its
    RMSE as an approximation of sqrt(E[Var(v|u)]).

    This tells you whether a "tie" between models is a genuine tie or
    everyone hitting the same noise floor (real financial returns typically
    have low signal-to-noise, so this check matters more here than on
    NARMA-10).
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    u_tr, u_te, v_tr, v_te = train_test_split(u, v.ravel(), test_size=0.3, random_state=random_state)
    model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(u_tr, v_tr)
    pred = model.predict(u_te)
    return float(np.sqrt(np.mean((pred - v_te) ** 2)))


if __name__ == "__main__":
    # smoke test with synthetic data
    np.random.seed(0)
    n = 12
    df = pd.DataFrame({
        "ticker": ["A"] * n,
        "seed": list(range(n)),
        "model": ["CRBM"] * n,
        "rmse": np.random.rand(n) * 0.1 + 0.5,
    })
    df2 = pd.DataFrame({
        "ticker": ["A"] * n,
        "seed": list(range(n)),
        "model": ["QCRBM"] * n,
        "rmse": np.random.rand(n) * 0.1 + 0.52,
    })
    combined = pd.concat([df, df2], ignore_index=True)

    result = compare_all_to_reference(combined, "CRBM", ["QCRBM"])
    print(result)

    dmin = power_analysis(n=12)
    print(f"Minimum detectable effect size at n=12: {dmin}")
