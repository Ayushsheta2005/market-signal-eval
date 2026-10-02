# Does News Sentiment Predict Next-Day Direction?

A rebuild of an earlier stock-prediction project, reframed around the question that
project could not answer: **after correct time-series validation and transaction costs,
does any model — sentiment-augmented or otherwise — beat the trivial baselines on
next-day direction?**

The honest expected answer is *no, or not by enough to trade*. That is a result. A
directional-accuracy number quoted without a baseline, a confidence interval, or costs is
not.

## What was wrong with the original

The predecessor reported **61% directional accuracy**. Running its dataset through this
pipeline:

```
$ python experiments/run_ablation.py --source legacy --csv all_stock_data.csv
  SKIP AAPL: 39 samples cannot support 6 folds with min_train=500
  SKIP AMZN: 39 samples ...
  [5 tickers, all skipped]

No ticker produced a single valid fold.
With this much data the question cannot be answered -- that is the result.
```

The source CSV holds 5 tickers over ~103 trading days. After the 63-day rolling windows
the features need, **39 usable observations per ticker remain**. At n=39 the 95% confidence
interval on an accuracy estimate is roughly ±15 percentage points, so 61% and 46% are not
distinguishable. The number was not wrong so much as undefined.

This repository is built so that failure mode cannot recur silently.

## The three things that make a directional number meaningful

**1. Validation that does not leak.** `src/validation.py` implements purged walk-forward
splitting with an embargo (López de Prado, ch. 7). Chronological ordering alone is not
enough: a label built from a forward return over *h* days is computed from prices that also
appear in the features of the next *h−1* rows, so those rows must be *purged* from training,
and a further *embargo* block dropped to absorb serial correlation. `assert_no_leakage()`
re-checks every split at runtime — this is the one place in the pipeline where a bug makes
results look *better*, so it is asserted rather than trusted.

**2. The right null hypothesis.** Equities drift up, so "always predict up" scores well
above 50%. The null is the **majority-class rate on the test window**, not a coin flip.
A model reporting 61% against an implied 50% baseline may be *losing* to a constant.
Every result is reported as excess over majority, with a one-sided binomial p-value and a
Wilson confidence interval.

**3. Transaction costs.** A signal that is accurate but flips daily pays the spread 252
times a year. `strategy_metrics` charges costs on turnover and reports gross *and* net
Sharpe side by side. The gap between them is usually where the strategy dies.

## Causality is tested, not assumed

`features.assert_causal()` perturbs prices strictly *after* a cut date and asserts that no
feature value on or before that date changes. A feature that peeks at the future moves, and
the test fails with the offending column named. It runs on every ticker inside the
experiment, not just in the test suite.

## Baseline ladder

| Baseline | Why it is there |
|---|---|
| `always_majority` | The real null. Beats 50% on drifting series. |
| `random_walk` | Base-rate coin flip. |
| `momentum` | Tomorrow repeats today — the simplest genuine signal. |
| `logistic` | Regularised linear model on price features. |
| `gbdt` | Shallow, heavily subsampled. Deep trees memorise noisy windows. |

A sentiment-augmented model must beat all five, on a Diebold–Mariano test rather than on a
higher point estimate, before the sentiment features have earned their place.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_core.py                       # 9 leakage/metric tests
.venv/bin/python experiments/run_ablation.py --source yfinance
.venv/bin/python experiments/run_ablation.py --source legacy --csv all_stock_data.csv
```

Default universe: 20 tickers × 15 years (~75,000 observations), versus ~500 in the original.

## Adding sentiment

`features.build_dataset(df, sentiment=...)` accepts a sentiment frame, and deliberately
requires the **caller** to have lagged it to the information set available at each close.
News published after the close belongs to the next trading day; merging on calendar date
without that shift is a lookahead bug that reliably manufactures a few points of accuracy.
The API will not do it for you, because doing it silently is how the bug gets shipped.

## Results

20 tickers, 2010-2024, next-day direction, 6 purged walk-forward folds per ticker,
**63,461 pooled out-of-fold predictions**, 5 bps round-trip costs:

```
model                 acc           95% CI  vs major       p    DM p   Sharpe  turnover
---------------------------------------------------------------------------------------
always_majority    0.5144    [0.511,0.518]   -0.0046   0.990   1.000     0.43     0.001
logistic           0.5087    [0.505,0.513]   -0.0103   1.000   0.015    -0.01     0.476
gbdt               0.5048    [0.501,0.509]   -0.0143   1.000   0.000    -0.04     0.661
random_walk        0.4945    [0.491,0.498]   -0.0246   1.000   0.000    -0.73     1.056
momentum           0.4944    [0.491,0.498]   -0.0246   1.000   0.000    -0.57     1.011
```

**No model beats always predicting the majority class.** Logistic regression and gradient
boosting both land *below* it, and the Diebold-Mariano test says significantly so
(p = 0.015 and p < 0.001) -- they are not merely tied with the constant, they are worse
than it.

Net of costs, every strategy that actually trades has a **negative Sharpe**. The only
positive number belongs to the baseline that almost never trades (turnover 0.001); its
0.43 Sharpe is long-equity drift, not a signal.

Read against the predecessor's 61%: on 63,461 observations the confidence intervals are
+/-0.4 points wide, and the best model in the ladder reaches 51.4%. A 61% figure is roughly
25 standard errors away from anything reproducible here. The original number came from 39
usable observations per ticker without purging, which is enough to produce it by chance.

This is a negative result, and it is the correct one. The contribution is the measurement
apparatus: the number a model gets is only as meaningful as the validation that produced it,
and this pipeline refuses to emit a number it cannot support.

### What would change the conclusion

Sentiment features are not yet wired in -- the news corpus is not in the source repository.
The experiment is built so that adding them answers a sharp question: does the sentiment
column move the Diebold-Mariano statistic against `always_majority`? If it does not, the
honest finding is that daily news sentiment carries no tradeable next-day signal, which is
consistent with most of the published literature once costs are charged.

## Layout

```
src/validation.py   purged walk-forward CV + runtime leakage assertions
src/features.py     price features with an empirical causality test
src/metrics.py      Wilson CIs, binomial test vs majority, Diebold-Mariano, net-of-cost Sharpe
src/baselines.py    the baseline ladder
src/data.py         yfinance loader, plus a reader for the original CSV
experiments/        the ablation
tests/              targets the bugs that make results look better, not worse
```

## References

- López de Prado, *Advances in Financial Machine Learning* (2018), ch. 7 — purging, embargo
- Diebold & Mariano, *Comparing Predictive Accuracy* (1995)
- Harvey, Leybourne & Newbold (1997) — small-sample DM correction
- Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014) — multiple-testing in backtests
