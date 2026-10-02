"""The baseline ladder.

A model is only interesting relative to what it beats. These are ordered by
how embarrassing it is to lose to them, and the experiment reports all of them
so a headline number cannot be quoted without its context.

The important one is `AlwaysMajority`. Equities drift up, so on most windows
"always predict up" scores well above 50%. A model reporting 61% against an
implied 50% null may be *underperforming* a constant.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AlwaysMajority:
    """Predict the majority class of the training window, always."""

    name = "always_majority"

    def fit(self, X, y):
        self.cls_ = int(np.round(np.mean(y)))
        return self

    def predict(self, X):
        return np.full(len(X), self.cls_, dtype=int)

    def predict_proba(self, X):
        p = float(self.cls_)
        return np.column_stack([np.full(len(X), 1 - p), np.full(len(X), p)])


class RandomWalk:
    """Predict up with the training base rate, independently each day."""

    name = "random_walk"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict(self, X):
        rng = np.random.default_rng(self.seed)
        return (rng.random(len(X)) < self.p_).astype(int)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p_), np.full(len(X), self.p_)])


class Momentum:
    """Tomorrow repeats today. The simplest real signal in the data."""

    name = "momentum"

    def __init__(self, col: str = "ret_1d"):
        self.col = col

    def fit(self, X, y):
        if self.col not in X.columns:
            raise ValueError(f"momentum needs column {self.col!r}")
        return self

    def predict(self, X):
        return (X[self.col].values > 0).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])


def logistic(seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.1, random_state=seed)),
        ]
    )


def gbdt(seed: int = 0) -> GradientBoostingClassifier:
    # Deliberately shallow and heavily subsampled: with a few thousand noisy
    # rows a deep tree ensemble will fit the training window perfectly and
    # tell you nothing.
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        random_state=seed,
    )


def registry(seed: int = 0) -> dict:
    return {
        "always_majority": AlwaysMajority(),
        "random_walk": RandomWalk(seed),
        "momentum": Momentum(),
        "logistic": logistic(seed),
        "gbdt": gbdt(seed),
    }
