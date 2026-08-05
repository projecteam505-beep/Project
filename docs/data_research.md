# Data research: what real financial returns actually look like

## Why this document exists

This sandbox's network policy blocks Yahoo Finance and every other market-data
host (only PyPI and GitHub are reachable — confirmed by direct `curl` tests
against `query1.finance.yahoo.com`, `stooq.com`, `fred.stlouisfed.org`, and
general internet hosts, all of which return a 403 from the egress proxy).
`src/data.py`'s `download_prices()` therefore cannot be exercised here.

Rather than fall back to a data generator invented with no grounding (a plain
Gaussian random walk, as the earlier `generate_synthetic_prices()` was), this
document collects the well-established statistical properties of *real*
financial return series from the empirical-finance / econometrics literature,
so the synthetic generator (`generate_research_calibrated_prices()` in
`src/data.py`) can be calibrated to reproduce them. The generated data is
still **not real market history** — it is a calibrated proxy for exercising
the pipeline in this network-restricted environment. Real numbers require
running `run_local.py` (without `--synthetic`) on a machine with network
access, per the README.

## The stylized facts (Cont, 2001, and the classical literature it surveys)

Rama Cont's "Empirical properties of asset returns: stylized facts and
statistical issues" (*Quantitative Finance*, 1(2), 223–236, 2001) is the
standard reference summarizing properties that hold across equity, FX, and
commodity return series, across markets and time periods. The properties
relevant to this project's modeling task:

1. **Absence of linear autocorrelation** — `corr(r_t, r_{t+k})` for raw
   returns is statistically indistinguishable from zero for k >= 1 on daily
   data (consistent with weak-form market efficiency; Fama, 1965/1970). This
   is *why* directional accuracy near 50% is the expected, correct outcome
   for a non-leaky model, not a failure — it was the basis of the accuracy
   discussion earlier in this conversation.

2. **Heavy tails / excess kurtosis** — the unconditional distribution of
   returns is leptokurtic: sample excess kurtosis for daily equity/FX returns
   is typically in the range of roughly 3–10 (vs. 0 for a Gaussian, i.e.
   kurtosis ~6–13 vs. the Gaussian's 3), reflecting more frequent extreme
   moves than a Normal distribution predicts (Mandelbrot, 1963; Fama, 1965).

3. **Volatility clustering** — "large changes tend to be followed by large
   changes, of either sign, and small changes tend to be followed by small
   changes" (Mandelbrot, 1963). Formally, autocorrelation of `|r_t|` or
   `r_t^2` is small but positive and decays slowly over many lags, in
   contrast to the near-zero autocorrelation of raw returns. This is the
   empirical basis for ARCH (Engle, 1982, Nobel Prize 2003) and GARCH
   (Bollerslev, 1986) models of conditional variance.

4. **Leverage effect** — volatility tends to rise more following a negative
   return than following a positive return of the same magnitude (Black,
   1976). Formalized in asymmetric volatility models such as GJR-GARCH
   (Glosten, Jagannathan & Runkle, 1993) and EGARCH (Nelson, 1991).

5. **Aggregational Gaussianity** — as the return horizon increases (daily ->
   weekly -> monthly), the return distribution approaches a Gaussian shape,
   consistent with a Central-Limit-Theorem effect over a volatility-clustered
   but not-too-long-memory process.

## Calibration used in `generate_research_calibrated_prices()`

The generator implements a **GJR-GARCH(1,1) process with standardized
Student-t innovations**, which is the minimal standard model that reproduces
facts 1–4 above simultaneously (fact 5 falls out of it automatically at
longer aggregation horizons):

```
r_t        = mu + eps_t
eps_t      = sigma_t * z_t,          z_t ~ standardized Student-t(nu)
sigma_t^2  = omega + (alpha + gamma * 1[eps_{t-1} < 0]) * eps_{t-1}^2 + beta * sigma_{t-1}^2
```

Per-asset parameters are drawn independently within literature-typical
ranges for daily equity-like series (e.g. Glosten-Jagannathan-Runkle 1993;
typical fitted GARCH persistence `alpha + beta` in the 0.90–0.99 range for
liquid daily series):

| Parameter | Range used | Role |
|---|---|---|
| `omega` | 1e-6 – 4e-6 | baseline variance level |
| `alpha` | 0.03 – 0.08 | ARCH term (reaction to the last shock) |
| `gamma` | 0.03 – 0.12 | GJR asymmetry / leverage term |
| `beta` | 0.85 – 0.90 | GARCH term (volatility persistence) |
| `nu` | 4 – 8 | Student-t degrees of freedom (fat tails; lower = fatter) |
| `mu` | -0.0001 – 0.0003 | small daily drift |

`alpha + gamma/2 + beta` stays comfortably under 1 for every draw, so the
variance process is stationary (mean-reverting), matching real markets.

## Validation

Claiming a dataset matches these properties is not enough — `src/data.py`'s
`stylized_facts_report()` computes, per generated asset, the exact
diagnostics that operationalize facts 1–4 above (excess kurtosis, skew,
lag-1 ACF of raw returns, lag-1 ACF of squared returns, and a leverage-effect
proxy `corr(r_t, r_{t+1}^2)`), and this is run and reported every time the
dataset is generated (see `results/stylized_facts_report.csv` after running
`run_local.py --synthetic`). If a run doesn't show near-zero raw-return
autocorrelation, positive squared-return autocorrelation, and excess
kurtosis > 0, the calibration has a bug — this is a falsifiable check, not a
marketing claim.

## Explicit limitations

- This is **not** real market history. It reproduces the *statistical
  signature* of real returns, not any actual asset's price path, and
  contains no real macroeconomic events, earnings surprises, or regime
  shifts.
- Cross-asset correlation is not modeled (each series is generated
  independently), whereas real assets in the same market/sector have
  non-trivial correlation structure (see `correlation_table()` in
  `src/data.py`, which is still meaningful to run on this synthetic data —
  it will correctly show near-zero cross-asset correlation, unlike real
  baskets).
- Parameters are drawn once per asset from the literature-typical ranges
  above, not fit to any specific real series, so magnitudes are
  representative rather than asset-specific.

## References

- Cont, R. (2001). Empirical properties of asset returns: stylized facts and
  statistical issues. *Quantitative Finance*, 1(2), 223–236.
- Mandelbrot, B. (1963). The variation of certain speculative prices.
  *Journal of Business*, 36(4), 394–419.
- Fama, E. F. (1965). The behavior of stock-market prices. *Journal of
  Business*, 38(1), 34–105.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation. *Econometrica*,
  50(4), 987–1007.
- Bollerslev, T. (1986). Generalized autoregressive conditional
  heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Black, F. (1976). Studies of stock price volatility changes. *Proceedings
  of the 1976 Meetings of the American Statistical Association, Business and
  Economic Statistics Section*, 177–181.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation
  between the expected value and the volatility of the nominal excess return
  on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: a
  new approach. *Econometrica*, 59(2), 347–370.
