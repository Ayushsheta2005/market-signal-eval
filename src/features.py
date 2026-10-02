"""Price features with an enforced no-lookahead guarantee.

Every feature at row t must be computable from information available at the
close of t, and the label at row t is the direction of the return from t to
t+horizon. The single most common way student finance models produce
impressive numbers is violating this by one row.

`build_dataset` therefore returns features and labels already aligned, and
`assert_causal` re-checks the alignment empirically rather than trusting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard technical features, all backward-looking.

    Expects columns Open/High/Low/Close/Volume indexed by date, one ticker.
    Rolling windows use only past data by construction; nothing is centred and
    nothing is shifted forward.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    close = df["Close"]
    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)

    for k in (1, 2, 3, 5, 10, 21):
        out[f"ret_{k}d"] = close.pct_change(k)
    for w in (5, 10, 21, 63):
        out[f"vol_{w}d"] = ret.rolling(w).std()
        out[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1
    out["rsi_14"] = _rsi(close, 14)
    out["hl_range"] = (df["High"] - df["Low"]) / close
    out["co_gap"] = (df["Open"] - close.shift(1)) / close.shift(1)
    out["vol_z_21"] = (
        df["Volume"] - df["Volume"].rolling(21).mean()
    ) / df["Volume"].rolling(21).std()
    out["dollar_vol_log"] = np.log1p(df["Volume"] * close)
    return out


def make_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """1 if the forward return over `horizon` is positive, else 0.

    The label at t is computed from prices strictly after t, which is exactly
    why training rows near a test window must be purged.
    """
    fwd = df["Close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int).rename("label")


def forward_returns(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    return df["Close"].pct_change(horizon).shift(-horizon).rename("fwd_return")


def build_dataset(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    sentiment: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble aligned (X, y, forward_returns), dropping incomplete rows.

    `sentiment` is optional and must already be lagged by the caller to the
    information set available at the close of each date -- news published after
    the close belongs to the next trading day, and merging on calendar date
    without that shift is a lookahead bug that reliably produces a few points
    of fake accuracy.
    """
    X = price_features(df)
    if sentiment is not None:
        overlap = set(X.columns) & set(sentiment.columns)
        if overlap:
            raise ValueError(f"sentiment columns collide with price features: {overlap}")
        X = X.join(sentiment, how="left")

    y = make_labels(df, horizon)
    r = forward_returns(df, horizon)

    frame = X.join([y, r]).dropna()
    if frame.empty:
        raise ValueError("no complete rows after feature construction")
    return frame[X.columns], frame["label"], frame["fwd_return"]


def assert_causal(df: pd.DataFrame, horizon: int = 1, n_checks: int = 25) -> None:
    """Empirically verify that features do not see the future.

    Perturbs prices strictly after a cut date and confirms no feature value on
    or before that date changes. A feature that leaks will move.
    """
    rng = np.random.default_rng(0)
    base = price_features(df)
    n = len(df)
    for _ in range(n_checks):
        cut = int(rng.integers(100, n - horizon - 1))
        mutated = df.copy()
        mutated.iloc[cut + 1 :, mutated.columns.get_loc("Close")] *= 1.5
        mutated.iloc[cut + 1 :, mutated.columns.get_loc("High")] *= 1.5
        after = price_features(mutated)

        a = base.iloc[: cut + 1]
        b = after.iloc[: cut + 1]
        if not np.allclose(a.fillna(0).values, b.fillna(0).values, atol=1e-12):
            bad = [c for c in a.columns
                   if not np.allclose(a[c].fillna(0), b[c].fillna(0), atol=1e-12)]
            raise AssertionError(
                f"lookahead detected at cut={cut}: features {bad} changed when "
                "only future prices were perturbed"
            )
