#!/usr/bin/env python
"""Does any model beat the trivial baselines on next-day direction?

Runs the full baseline ladder under purged walk-forward validation across
every ticker, pools the out-of-fold predictions, and reports directional
accuracy with confidence intervals, Diebold-Mariano comparisons against the
majority-class predictor, and net-of-cost strategy metrics.

    python experiments/run_ablation.py --source yfinance
    python experiments/run_ablation.py --source legacy --csv path/to/all_stock_data.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines, data, features, metrics
from src.validation import PurgedWalkForward, assert_no_leakage

warnings.filterwarnings("ignore", category=FutureWarning)


def run_ticker(df, name, cv, horizon, seed, cost_bps):
    X, y, fwd = features.build_dataset(df, horizon=horizon)
    features.assert_causal(df, horizon=horizon)

    splits = list(cv.split(len(X)))
    assert_no_leakage(splits, cv.label_horizon, cv.embargo)

    preds = defaultdict(list)
    truth, rets = [], []
    for s in splits:
        Xtr, ytr = X.iloc[s.train], y.iloc[s.train]
        Xte, yte = X.iloc[s.test], y.iloc[s.test]
        if ytr.nunique() < 2:
            continue
        truth.append(yte.values)
        rets.append(fwd.iloc[s.test].values)
        for mname, model in baselines.registry(seed).items():
            model.fit(Xtr, ytr)
            preds[mname].append(np.asarray(model.predict(Xte)).astype(int))

    if not truth:
        raise RuntimeError(f"{name}: no usable folds")
    return (
        {k: np.concatenate(v) for k, v in preds.items()},
        np.concatenate(truth),
        np.concatenate(rets),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("yfinance", "legacy"), default="yfinance")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation.json")
    args = ap.parse_args()

    if args.source == "legacy":
        if not args.csv:
            ap.error("--source legacy requires --csv")
        panel = data.load_legacy_csv(args.csv)
    else:
        panel = data.download()

    cv = PurgedWalkForward(
        n_folds=args.folds,
        label_horizon=args.horizon,
        embargo=args.embargo,
        min_train=args.min_train,
    )

    all_preds = defaultdict(list)
    all_truth, all_rets = [], []
    skipped = {}
    for name, df in sorted(panel.items()):
        try:
            p, t, r = run_ticker(df, name, cv, args.horizon, args.seed, args.cost_bps)
        except (ValueError, RuntimeError) as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
            print(f"  SKIP {name}: {exc}", flush=True)
            continue
        print(f"  ok   {name}: {len(t)} out-of-fold predictions", flush=True)
        for k, v in p.items():
            all_preds[k].append(v)
        all_truth.append(t)
        all_rets.append(r)

    if not all_truth:
        print("\nNo ticker produced a single valid fold.")
        print("With this much data the question cannot be answered -- that is the result.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"skipped": skipped}, indent=2))
        return

    truth = np.concatenate(all_truth)
    rets = np.concatenate(all_rets)
    pooled = {k: np.concatenate(v) for k, v in all_preds.items()}

    base_err = (pooled["always_majority"] != truth).astype(float)
    rows = []
    for name, pred in pooled.items():
        d = metrics.directional_accuracy(truth, pred)
        strat = metrics.strategy_metrics(
            rets, np.where(pred == 1, 1.0, -1.0), cost_bps=args.cost_bps
        )
        try:
            dm_stat, dm_p = metrics.diebold_mariano(
                (pred != truth).astype(float), base_err, horizon=args.horizon
            )
        except ValueError:
            dm_stat, dm_p = float("nan"), float("nan")
        rows.append({"model": name, **d.to_dict(), "dm_stat_vs_majority": dm_stat,
                     "dm_p_vs_majority": dm_p, **strat})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "n_pooled": int(len(truth)),
         "skipped": skipped, "results": rows}, indent=2))
    _table(rows, len(truth))


def _table(rows, n):
    print(f"\nPooled out-of-fold predictions: {n:,}\n")
    hdr = (f"{'model':<17}{'acc':>8}{'95% CI':>17}{'vs major':>10}"
           f"{'p':>8}{'DM p':>8}{'Sharpe':>9}{'turnover':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["accuracy"]):
        ci = "[{:.3f},{:.3f}]".format(r["ci_low"], r["ci_high"])
        print(
            f"{r['model']:<17}{r['accuracy']:>8.4f}{ci:>17}"
            f"{r['excess_over_majority']:>+10.4f}{r['p_value_vs_majority']:>8.3f}"
            f"{r['dm_p_vs_majority']:>8.3f}{r['sharpe_net']:>9.2f}"
            f"{r['avg_turnover']:>10.3f}"
        )
    print(
        "\nSharpe is net of transaction costs. 'vs major' is excess over always "
        "predicting the\nmajority class -- the only comparison that matters, since "
        "equity drift makes 50% the wrong null."
    )


if __name__ == "__main__":
    main()
