"""Metrics and significance tests for directional forecasts.

Directional accuracy on its own is close to meaningless. It needs three things
attached to it before it supports a claim:

1. **A confidence interval.** 61% on 500 observations has a 95% CI of roughly
   +/-4.3 points. Quoting the point estimate alone hides that.
2. **A comparison against the right null.** The null is not 50% -- equity
   indices drift up, so "always predict up" beats a coin flip. The null is the
   majority-class rate on the test window.
3. **Transaction costs.** A signal that is accurate but flips daily is not
   profitable. Accuracy without turnover and costs is not a result.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class DirectionalResult:
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    majority_class_rate: float
    excess_over_majority: float
    p_value_vs_majority: float

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        return (
            f"acc={self.accuracy:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}]  "
            f"majority={self.majority_class_rate:.4f}  "
            f"excess={self.excess_over_majority:+.4f}  p={self.p_value_vs_majority:.4f}  "
            f"n={self.n}"
        )


def directional_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, *, alpha: float = 0.05
) -> DirectionalResult:
    """Accuracy with a Wilson interval, benchmarked against majority class.

    The Wilson interval is used rather than the normal approximation because it
    stays inside [0, 1] and behaves at small n, which is the regime most of
    these test windows are in.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be the same shape")
    n = len(y_true)
    if n == 0:
        raise ValueError("empty evaluation window")

    hits = int((y_true == y_pred).sum())
    acc = hits / n

    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (acc + z**2 / (2 * n)) / denom
    half = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom

    base = max(y_true.mean(), 1 - y_true.mean())
    # One-sided binomial test: is the model better than always predicting the
    # majority class on this window?
    p = stats.binomtest(hits, n, base, alternative="greater").pvalue

    return DirectionalResult(
        n=n,
        accuracy=acc,
        ci_low=max(0.0, centre - half),
        ci_high=min(1.0, centre + half),
        majority_class_rate=base,
        excess_over_majority=acc - base,
        p_value_vs_majority=float(p),
    )


def diebold_mariano(
    errors_a: np.ndarray, errors_b: np.ndarray, *, horizon: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test that two forecasts have equal accuracy.

    Comparing two models by whose accuracy is numerically higher ignores that
    the two forecast the same series and their errors are correlated. DM tests
    the loss differential directly, with a Newey-West correction for the
    autocorrelation that a multi-period horizon induces.

    Returns (statistic, two-sided p-value). Negative statistic favours A.
    """
    d = np.asarray(errors_a, dtype=float) - np.asarray(errors_b, dtype=float)
    n = len(d)
    if n < 10:
        raise ValueError(f"DM test needs a meaningful sample; got n={n}")
    d_bar = d.mean()

    lag = max(0, horizon - 1)
    gamma0 = np.sum((d - d_bar) ** 2) / n
    var = gamma0
    for k in range(1, lag + 1):
        gk = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n
        var += 2 * (1 - k / (lag + 1)) * gk
    if var <= 0:
        return 0.0, 1.0

    stat = d_bar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction.
    corr = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat *= corr
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def strategy_metrics(
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    cost_bps: float = 5.0,
    periods_per_year: int = 252,
) -> dict:
    """Net-of-cost performance of trading the forecast.

    `positions` is the signed exposure implied by the forecast (+1 long,
    -1 short, 0 flat), aligned so that position[t] is held into return[t].
    Costs are charged on turnover, which is where most paper strategies die:
    a daily-flipping signal pays the spread 252 times a year.
    """
    returns = np.asarray(returns, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if returns.shape != positions.shape:
        raise ValueError("returns and positions must align")

    turnover = np.abs(np.diff(positions, prepend=0.0))
    costs = turnover * (cost_bps / 10_000.0)
    net = positions * returns - costs

    ann_ret = net.mean() * periods_per_year
    ann_vol = net.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    equity = np.cumprod(1 + net)
    drawdown = equity / np.maximum.accumulate(equity) - 1

    gross = positions * returns
    return {
        "ann_return_net": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe_net": float(sharpe),
        "sharpe_gross": float(
            (gross.mean() * periods_per_year)
            / (gross.std(ddof=1) * np.sqrt(periods_per_year))
            if gross.std(ddof=1) > 0
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
        "avg_turnover": float(turnover.mean()),
        "total_cost_drag": float(costs.sum()),
        "hit_rate": float((net > 0).mean()),
    }
