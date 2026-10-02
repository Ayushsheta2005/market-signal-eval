"""Tests for the parts of the pipeline whose bugs would make results LOOK BETTER.

Ordinary bugs make numbers worse and get noticed. Leakage bugs make numbers
better and get published. These tests target the second kind.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features, metrics
from src.validation import PurgedWalkForward, assert_no_leakage


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1e6, 1e8, n).astype(float),
        },
        index=idx,
    )


def test_splits_are_chronological_and_gapped():
    cv = PurgedWalkForward(n_folds=5, label_horizon=1, embargo=5, min_train=250)
    splits = list(cv.split(2000))
    assert len(splits) == 5
    for s in splits:
        assert s.train.max() < s.test.min()
    assert_no_leakage(splits, 1, 5)


def test_longer_horizon_widens_the_purge():
    a = list(PurgedWalkForward(n_folds=4, label_horizon=1, embargo=0, min_train=250).split(2000))
    b = list(PurgedWalkForward(n_folds=4, label_horizon=20, embargo=0, min_train=250).split(2000))
    gap_a = a[1].test.min() - a[1].train.max()
    gap_b = b[1].test.min() - b[1].train.max()
    assert gap_b > gap_a, "a 20-day label must purge more than a 1-day label"


def test_leakage_assertion_actually_fires():
    from src.validation import Split
    bad = [Split(train=np.arange(0, 100), test=np.arange(99, 150), fold=0)]
    try:
        assert_no_leakage(bad, 1, 5)
    except AssertionError:
        return
    raise AssertionError("overlapping split was not caught")


def test_features_do_not_see_the_future():
    # This is the test that matters: perturb only future prices and confirm
    # no past feature moves.
    features.assert_causal(_synthetic(), horizon=1, n_checks=15)


def test_label_is_strictly_forward():
    df = _synthetic(300)
    y = features.make_labels(df, horizon=1)
    r = df["Close"].pct_change().shift(-1)
    both = pd.concat([y, r], axis=1).dropna()
    assert ((both.iloc[:, 1] > 0).astype(int) == both.iloc[:, 0]).all()


def test_accuracy_ci_covers_and_narrows_with_n():
    rng = np.random.default_rng(0)
    for n in (200, 20_000):
        y = rng.integers(0, 2, n)
        p = y.copy()
        flip = rng.random(n) < 0.4
        p[flip] = 1 - p[flip]
        r = metrics.directional_accuracy(y, p)
        assert r.ci_low <= r.accuracy <= r.ci_high
    small = metrics.directional_accuracy(rng.integers(0, 2, 200), rng.integers(0, 2, 200))
    large = metrics.directional_accuracy(rng.integers(0, 2, 20000), rng.integers(0, 2, 20000))
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_majority_baseline_is_not_fifty_percent_on_drifting_series():
    # The whole reason the majority baseline exists.
    y = np.concatenate([np.ones(560), np.zeros(440)]).astype(int)
    r = metrics.directional_accuracy(y, np.ones(1000, dtype=int))
    assert abs(r.majority_class_rate - 0.56) < 1e-9
    assert abs(r.excess_over_majority) < 1e-9, "predicting the majority cannot beat it"


def test_costs_reduce_sharpe_for_a_flipping_signal():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, 1000)
    flipping = np.where(np.arange(1000) % 2 == 0, 1.0, -1.0)
    free = metrics.strategy_metrics(rets, flipping, cost_bps=0.0)
    costly = metrics.strategy_metrics(rets, flipping, cost_bps=10.0)
    assert costly["sharpe_net"] < free["sharpe_net"]
    assert costly["avg_turnover"] > 1.5


def test_dm_rejects_when_one_model_is_clearly_better():
    rng = np.random.default_rng(2)
    good = (rng.random(2000) < 0.30).astype(float)
    bad = (rng.random(2000) < 0.50).astype(float)
    stat, p = metrics.diebold_mariano(good, bad, horizon=1)
    assert stat < 0 and p < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
