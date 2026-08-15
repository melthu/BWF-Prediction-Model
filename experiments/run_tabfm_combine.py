"""
Score every cached prediction set, and every blend of them, against the baseline.

`run_tabfm.py` writes one CSV of per-match validation predictions per run. This
reads them back and answers the questions that do not need another forward pass:

  * how each model stands against the shipped LightGBM, pooled over all folds;
  * whether averaging a foundation model with the shipped one beats either.

The blend is the question worth asking even when a candidate loses on its own.
A soft-vote of LightGBM, XGBoost and CatBoost was already tried and did not help
(experiments/README.md) - the three are too correlated for averaging to buy
anything. A transformer that reads the training rows in context is a genuinely
different estimator, so its errors have a real chance of being decorrelated in a
way a fourth tree ensemble's are not.

    python3 experiments/run_tabfm_combine.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_tabfm import BASELINE, N_BOOT, PRED_DIR, RESULTS_DIR, paired_bootstrap, slug

BASE_SLUG = slug(BASELINE)


def load_preds():
    """{display name: frame}, deduplicated across the concurrent run tags.

    Every run re-scores the baseline, so the same model appears under several
    `<tag>__<model>.csv` names. They are identical by construction (same folds,
    same seed), so the first one wins and the rest are checked, not stacked.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "*.csv"))):
        model = os.path.basename(path)[:-4].split("__", 1)[-1]
        frame = pd.read_csv(path).sort_values(["year", "idx"]).reset_index(drop=True)
        if model in out:
            if not np.allclose(out[model]["p"].values, frame["p"].values):
                raise ValueError(f"{model} differs between runs - {path}")
            continue
        out[model] = frame
    return out


def auc_of(frame):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(frame["y"].values, frame["p"].values)


def per_fold_delta(frame, base):
    """ΔAUC against the baseline within each evaluation year.

    A pooled interval that excludes zero is still worth breaking apart: five
    folds all leaning the same way is a different claim from one fold carrying
    the other four.
    """
    m = frame.merge(base, on=["year", "idx"], suffixes=("_a", "_b"))
    out = {}
    for year, g in m.groupby("year"):
        out[int(year)] = float(auc_of(g.rename(columns={"y_a": "y", "p_a": "p"}))
                               - auc_of(g.rename(columns={"y_b": "y", "p_b": "p"})))
    return out


def blend(a, b, w=0.5):
    """Probability-average of two runs, aligned match-for-match."""
    m = a.merge(b, on=["year", "idx"], suffixes=("_a", "_b"))
    if len(m) != len(a):
        raise ValueError("blend operands cover different matches")
    return pd.DataFrame({"year": m["year"], "idx": m["idx"], "y": m["y_a"],
                         "p": w * m["p_a"] + (1 - w) * m["p_b"]})


def main():
    preds = load_preds()
    if BASE_SLUG not in preds:
        raise SystemExit(f"no baseline predictions in {PRED_DIR}; run run_tabfm.py first")
    base = preds[BASE_SLUG]
    print(f"{len(preds)} cached runs over {len(base):,} validation matches\n")

    rows = dict(preds)
    # Blends of every foundation model with the shipped baseline, at the two
    # weightings worth reporting: an equal vote, and a minority correction.
    for name, frame in list(preds.items()):
        if name == BASE_SLUG:
            continue
        for w, tag in ((0.5, "50_50"), (0.3, "30_70")):
            rows[f"blend_{tag}__{name}"] = blend(frame, base, w)

    print(f"{'run':<46} {'pooled AUC':>10}  {'Δ vs baseline':>14}  95% CI")
    print("-" * 92)
    results = {}
    for name, frame in sorted(rows.items(), key=lambda kv: -auc_of(kv[1])):
        auc = auc_of(frame)
        if name == BASE_SLUG:
            print(f"{name:<46} {auc:>10.4f}  {'(baseline)':>14}")
            results[name] = {"auc": float(auc)}
            continue
        b = paired_bootstrap(frame, base, n_boot=N_BOOT)
        verdict = ("BETTER" if b["ci_low"] > 0 else
                   "WORSE" if b["ci_high"] < 0 else "no difference")
        folds = per_fold_delta(frame, base)
        print(f"{name:<46} {auc:>10.4f}  {b['delta_auc']:>+14.4f}  "
              f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  {verdict}")
        print(f"{'':<46} {'per fold:':>10}  "
              + "  ".join(f"{y}:{d:+.4f}" for y, d in sorted(folds.items())))
        results[name] = {"auc": float(auc), **b, "verdict": verdict,
                         "per_fold_delta": folds}

    path = os.path.join(RESULTS_DIR, "tabfm_combined.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
