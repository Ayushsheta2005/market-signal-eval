"""Time-series cross-validation that does not leak.

Standard k-fold on financial data is invalid for two reasons, and both of them
inflate accuracy in the same direction:

1. **Lookahead.** A random split puts future observations in the training set.
   The model learns from days that had not happened yet.
2. **Label overlap.** A label built from a forward return over h days is
   computed from prices that also appear in the features of the next h-1
   observations. Even a chronological split leaks across the boundary unless
   those overlapping observations are removed.

This module implements purged walk-forward splitting with an embargo
(Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7). It is the
difference between a directional-accuracy number that means something and one
that does not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    fold: int

    def __repr__(self) -> str:
        return (
            f"Split(fold={self.fold}, train={len(self.train)} "
            f"[{self.train.min()}..{self.train.max()}], "
            f"test={len(self.test)} [{self.test.min()}..{self.test.max()}])"
        )


class PurgedWalkForward:
    """Expanding or rolling walk-forward splits with purge and embargo.

    Parameters
    ----------
    n_folds:
        Number of sequential test windows.
    label_horizon:
        How many periods forward the label looks. Training observations whose
        label window overlaps the test window are purged. For next-day
        direction this is 1; for a 5-day forward return it is 5.
    embargo:
        Periods dropped from the end of training *after* purging, to absorb
        serial correlation that purging alone does not remove. Expressed in
        periods, typically a small multiple of label_horizon.
    expanding:
        True grows the training window each fold (more data, changing regime
        mix). False keeps it fixed-length, which tests whether the signal
        survives regime change rather than being averaged across it.
    min_train:
        Refuse to emit a fold with less training data than this. Silently
        training on 40 observations is how spurious results happen.
    """

    def __init__(
        self,
        n_folds: int = 5,
        *,
        label_horizon: int = 1,
        embargo: int = 5,
        expanding: bool = True,
        min_train: int = 250,
    ):
        if n_folds < 2:
            raise ValueError("need at least 2 folds")
        if label_horizon < 1:
            raise ValueError("label_horizon must be >= 1")
        self.n_folds = n_folds
        self.label_horizon = label_horizon
        self.embargo = embargo
        self.expanding = expanding
        self.min_train = min_train

    def split(self, n_samples: int) -> Iterator[Split]:
        if n_samples < self.min_train + self.label_horizon + self.embargo + self.n_folds:
            raise ValueError(
                f"{n_samples} samples cannot support {self.n_folds} folds with "
                f"min_train={self.min_train}"
            )

        # Test windows tile the tail of the series. The first test window
        # starts far enough in that min_train survives *after* purge+embargo --
        # otherwise fold 0 silently trains on fewer rows than requested.
        offset = self.min_train + self.label_horizon + self.embargo
        usable = n_samples - offset
        if usable < self.n_folds:
            raise ValueError(
                f"{n_samples} samples leave only {usable} rows for {self.n_folds} "
                f"test folds after reserving min_train={self.min_train} plus "
                f"purge+embargo={self.label_horizon + self.embargo}"
            )
        fold_size = usable // self.n_folds
        if fold_size < 1:
            raise ValueError("fold size collapsed to zero; reduce n_folds")

        for k in range(self.n_folds):
            test_start = offset + k * fold_size
            test_end = (
                n_samples if k == self.n_folds - 1 else test_start + fold_size
            )
            test = np.arange(test_start, test_end)

            # Purge: a training observation at index i uses a label spanning
            # [i, i + label_horizon]. Drop it if that window reaches the test
            # window. Then embargo a further block.
            cutoff = test_start - self.label_horizon - self.embargo
            train_start = 0 if self.expanding else max(0, cutoff - self.min_train)
            train = np.arange(train_start, max(train_start, cutoff))

            if len(train) < self.min_train:
                raise ValueError(
                    f"fold {k} has only {len(train)} training samples after "
                    f"purge+embargo (min_train={self.min_train}); "
                    "lengthen the series or reduce n_folds/embargo"
                )
            yield Split(train=train, test=test, fold=k)

    def describe(self, n_samples: int) -> str:
        lines = [
            f"PurgedWalkForward: {self.n_folds} folds over {n_samples} samples",
            f"  label_horizon={self.label_horizon} embargo={self.embargo} "
            f"expanding={self.expanding}",
        ]
        for s in self.split(n_samples):
            gap = s.test.min() - s.train.max() - 1
            lines.append(f"  {s}  gap={gap}")
        return "\n".join(lines)


def assert_no_leakage(splits: list[Split], label_horizon: int, embargo: int) -> None:
    """Fail loudly if any split could leak. Called by the experiment runner.

    A test that the validation scheme itself is correct, rather than trusting
    that it was implemented right -- this is the one piece of the pipeline
    where a bug produces better-looking results instead of worse ones.
    """
    for s in splits:
        if len(s.train) == 0:
            raise AssertionError(f"fold {s.fold}: empty training set")
        if s.train.max() >= s.test.min():
            raise AssertionError(
                f"fold {s.fold}: training index {s.train.max()} is not before "
                f"test index {s.test.min()}"
            )
        gap = s.test.min() - s.train.max() - 1
        required = label_horizon + embargo - 1
        if gap < required:
            raise AssertionError(
                f"fold {s.fold}: gap of {gap} periods is smaller than the "
                f"required purge+embargo of {required}"
            )
