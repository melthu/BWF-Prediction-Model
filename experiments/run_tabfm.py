"""
Do tabular foundation models beat the shipped gradient-boosted model?

TabPFN and TabICL are pretrained transformers that classify a new table
*in context*: the training rows are fed as a prompt and the test rows are
predicted in one forward pass, with no gradient step on this dataset at all.
They win most public tabular benchmarks under ~10k rows, which is roughly the
size of this corpus, so the question is a fair one to ask here.

Everything is scored on `harness.evaluate` - the same rolling temporal split,
train-side mirroring and order-invariant prediction the shipped candidates are
scored on - so the numbers land on the same scale as the tables in
`experiments/README.md`. The verdict is a paired bootstrap over MATCHES against
the shipped baseline, because the harness's noise floor (~0.003 AUC) is wide
enough to swallow any difference smaller than that.

    python3 experiments/run_tabfm.py --probe          # one fold, quick look
    python3 experiments/run_tabfm.py                  # full rolling run

Needs `tabpfn` and/or `tabicl`, which are NOT in pyproject.toml - this is an
experiment, not a dependency. Install them into a throwaway venv:

    python3 -m venv --system-site-packages /tmp/tabfmenv
    /tmp/tabfmenv/bin/python -m pip install 'tabpfn==2.2.1' tabicl

Note that `tabpfn>=6` gates its weights behind a Prior Labs licence token
(TABPFN_TOKEN); the 2.x line downloads the same v2 checkpoints unauthenticated.
"""
import argparse
import json
import os
import sys
import time

# torch BEFORE anything that pulls in scikit-learn. Both ship their own copy of
# libomp, and on macOS whichever loads second hits `OMP: Error #179` and the
# process segfaults the first time TabPFN runs a forward pass - with stdout
# still buffered, so it dies silently. Importing torch first is the whole fix.
import torch  # noqa: F401  (import for its side effect, not its API)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.harness import EVAL_YEARS, evaluate, feature_cols, load_frame, metrics
from experiments.models import factory, tuned_params

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PRED_DIR = os.path.join(RESULTS_DIR, "preds")
N_BOOT = 2000
BASELINE = "lgbm (shipped baseline)"
PRED_TAG = "run"          # set from --out in main()


# --------------------------------------------------------------- wrappers

def subsample_indices(n_rows, max_context, n_members, paired=True,
                      recency=False, seed=42):
    """Row-index arrays, one per ensemble member, each within `max_context`.

    Both wrappers need this. In-context attention is quadratic in the number of
    training rows, so an unbounded context is a memory decision as much as an
    accuracy one - TabICL reading all 18,754 mirrored rows reached 6.25 GB
    resident and pushed this machine into swap.

    `paired` keeps a mirrored pair in the same subsample. `harness.evaluate`
    hands over `[originals; mirrors]`, so row i pairs with row i + n/2; feeding
    a member only one orientation of a match would reintroduce exactly the slot
    bias the mirroring exists to remove.

    `recency` takes the subsample from the tail of the chronological frame
    instead of uniformly, which is worth asking about separately: a 2013 match
    may simply be worth less context than a 2025 one.
    """
    if paired:
        if n_rows % 2:
            raise ValueError("paired=True needs an [originals; mirrors] matrix")
        n_pairs, budget = n_rows // 2, max_context // 2
    else:
        n_pairs, budget = n_rows, max_context

    if n_pairs <= budget:
        return [np.arange(n_rows)]      # one member; subsampling is moot

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_members):
        if recency:
            # Most recent `budget` matches, jittered so members differ: each
            # drops a random tenth of the window and backfills from just before
            # it, otherwise every member is the same rows.
            keep = rng.permutation(np.arange(n_pairs - budget, n_pairs))
            keep = keep[: int(budget * 0.9)]
            older = rng.choice(n_pairs - budget, budget - len(keep), replace=False)
            pick = np.concatenate([keep, older])
        else:
            pick = rng.choice(n_pairs, budget, replace=False)
        out.append(np.concatenate([pick, pick + n_pairs]) if paired else pick)
    return out


class TabPFNEnsemble:
    """TabPFN v2 over a training set larger than its pretraining limit.

    TabPFN was pretrained on synthetic tables of up to 10,000 rows, so the whole
    mirrored training slice (up to ~18,700 rows here) neither fits its prior nor
    fits in memory - an MPS run dies asking for a 10.9 GiB buffer. The standard
    workaround, and what tabpfn-extensions does, is to fit several members on
    different subsamples and average their probabilities. See
    `subsample_indices` for what `paired` and `recency` mean.
    """

    def __init__(self, n_members=4, max_context=10000, device="cpu",
                 n_estimators=1, paired=True, recency=False, random_state=42):
        self.n_members = n_members
        self.max_context = max_context
        self.device = device
        self.n_estimators = n_estimators
        self.paired = paired
        self.recency = recency
        self.random_state = random_state

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        self.members_ = []
        for idx in subsample_indices(len(X), self.max_context, self.n_members,
                                     self.paired, self.recency, self.random_state):
            clf = TabPFNClassifier(
                device=self.device, n_jobs=1, n_estimators=self.n_estimators,
                random_state=self.random_state, ignore_pretraining_limits=True,
            )
            clf.fit(X[idx], y[idx])
            self.members_.append(clf)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members_], axis=0)
        return np.column_stack([1.0 - p, p])


class TabICLEnsemble:
    """TabICL v2, subsampled to the same context budget as TabPFN.

    TabICL is built for tables an order of magnitude larger than TabPFN's
    pretraining limit, so the obvious run is on every training row. That run
    was abandoned: at 18,754 mirrored rows it reached 6.25 GB resident, drove
    this machine into swap, and spent its time in uninterruptible page-in wait
    at 20% CPU. Capping it at the same `max_context` as TabPFN costs it the one
    advantage it has, but makes the head-to-head a like-for-like one - both
    models then see the same number of rows, and any difference is the model.
    """

    def __init__(self, n_members=4, max_context=4000, device="cpu",
                 n_estimators=2, paired=True, random_state=42):
        self.n_members = n_members
        self.max_context = max_context
        self.device = device
        self.n_estimators = n_estimators
        self.paired = paired
        self.random_state = random_state

    def fit(self, X, y):
        from tabicl import TabICLClassifier

        self.members_ = []
        for idx in subsample_indices(len(X), self.max_context, self.n_members,
                                     self.paired, False, self.random_state):
            clf = TabICLClassifier(
                device=self.device, n_estimators=self.n_estimators,
                random_state=self.random_state, n_jobs=1,
            )
            clf.fit(X[idx], y[idx])
            self.members_.append(clf)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members_], axis=0)
        return np.column_stack([1.0 - p, p])


# ------------------------------------------------------------ evaluation

def paired_bootstrap(preds_a, preds_b, n_boot=N_BOOT, seed=0):
    """ΔAUC (a − b) with a 95% interval, resampling matches within each fold.

    `harness.evaluate(return_preds=True)` emits one row per validation MATCH
    (the val slice is never mirrored), so a row here is already one
    observation. Resampling is stratified by fold so every year keeps its
    weight, and AUC is pooled across folds the way the MEAN row is not - the
    interval is about the pooled ranking, which is what a shipped model does.
    """
    from sklearn.metrics import roc_auc_score

    merged = preds_a.merge(preds_b, on=["year", "idx"], suffixes=("_a", "_b"))
    if len(merged) != len(preds_a):
        raise ValueError("prediction frames do not line up match-for-match")
    if not np.array_equal(merged["y_a"].values, merged["y_b"].values):
        raise ValueError("labels disagree between the two runs")

    y = merged["y_a"].values
    pa, pb = merged["p_a"].values, merged["p_b"].values
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)

    folds = [np.asarray(i) for i in merged.groupby("year").indices.values()]
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(n_boot):
        rows = np.concatenate([f[rng.integers(0, len(f), len(f))] for f in folds])
        yb = y[rows]
        if yb.min() != yb.max():
            deltas.append(roc_auc_score(yb, pa[rows]) - roc_auc_score(yb, pb[rows]))
    deltas = np.array(deltas)
    return {
        "delta_auc": float(point),
        "ci_low":    float(np.percentile(deltas, 2.5)),
        "ci_high":   float(np.percentile(deltas, 97.5)),
        "n_matches": int(len(y)),
    }


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def run(name, make_model, df, cont_cols, use_player_ids, years):
    t0 = time.time()
    table, preds = evaluate(df, make_model, cont_cols, use_player_ids=use_player_ids,
                            years=years, return_preds=True)
    mean = table.loc["MEAN"]
    print(f"  {name:<34} AUC {mean['auc']:.4f}  logloss {mean['logloss']:.4f}  "
          f"acc {mean['acc']:.4f}   [{time.time() - t0:.0f}s]", flush=True)

    # Per-match predictions are kept so blends and cross-comparisons can be
    # scored later without paying for another five folds of forward passes.
    # Variants are run as concurrent processes and every one of them re-scores
    # the baseline, so the destination is namespaced by --out; two processes
    # writing one `lgbm...csv` would interleave.
    os.makedirs(PRED_DIR, exist_ok=True)
    preds.to_csv(os.path.join(PRED_DIR, f"{PRED_TAG}__{slug(name)}.csv"), index=False)
    return table, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="one fold (the latest year) instead of the full rolling run")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the foundation models; mps OOMs above "
                         "roughly 5k context, so cpu is the default")
    ap.add_argument("--members", type=int, default=4,
                    help="TabPFN subsample-ensemble size")
    ap.add_argument("--context", type=int, default=10000,
                    help="TabPFN rows per member (its pretraining limit is 10k)")
    ap.add_argument("--icl-members", type=int, default=4,
                    help="TabICL subsample-ensemble size")
    ap.add_argument("--icl-context", type=int, default=4000,
                    help="TabICL rows per member. Lower than TabPFN's because "
                         "TabICL's memory grows steeply with context - 10k "
                         "peaked at 7.4 GB here and drove the machine into "
                         "swap, 4k peaks at 3 GB. It is cheap enough in time "
                         "to buy the coverage back with more members.")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of run names")
    ap.add_argument("--out", default="tabfm.json", help="filename under results/")
    args = ap.parse_args()

    global PRED_TAG
    PRED_TAG = slug(os.path.splitext(args.out)[0])

    years = EVAL_YEARS[-1:] if args.probe else EVAL_YEARS
    df = load_frame()
    cont = feature_cols(df)
    print(f"{len(df):,} completed matches, {len(cont)} continuous features, "
          f"folds {years}\n")

    runs = {
        # Baselines: the shipped candidates on their shipped params.
        "lgbm (shipped baseline)":
            (lambda: factory("lgbm", tuned_params("lgbm"))(), True),
        "xgb (tuned)":
            (lambda: factory("xgb", tuned_params("xgb"))(), True),

        # TabPFN: player IDs in and out. A 700-way identifier coded as an
        # integer is a hostile input to a model that reads columns as numeric,
        # so it gets asked both ways rather than assumed.
        "TabPFN (subsample ens.)":
            (lambda: TabPFNEnsemble(args.members, args.context, args.device), True),
        "TabPFN (no player IDs)":
            (lambda: TabPFNEnsemble(args.members, args.context, args.device), False),
        "TabPFN (recency subsample)":
            (lambda: TabPFNEnsemble(args.members, args.context, args.device,
                                    recency=True), True),

        # TabICL, at TabPFN's context budget - see TabICLEnsemble for why the
        # full-context run was abandoned.
        "TabICL (subsample ens.)":
            (lambda: TabICLEnsemble(args.icl_members, args.icl_context,
                                    args.device), True),
        "TabICL (no player IDs)":
            (lambda: TabICLEnsemble(args.icl_members, args.icl_context,
                                    args.device), False),
    }
    if args.only:
        # The baseline always survives the filter: it is 20 seconds of work and
        # every verdict below is a paired bootstrap against it, so a run without
        # it can only report a bare AUC with no error bar.
        wanted = [w.strip() for w in args.only.split(",")]
        runs = {k: v for k, v in runs.items()
                if k == BASELINE or any(w.lower() in k.lower() for w in wanted)}

    tables, preds, failed = {}, {}, {}
    for name, (make_model, use_ids) in runs.items():
        try:
            tables[name], preds[name] = run(name, make_model, df, cont, use_ids, years)
        except Exception as exc:                      # a missing package, an OOM
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"  {name:<34} FAILED  {failed[name][:90]}", flush=True)

    baseline = BASELINE
    summary = {"folds": list(years), "n_matches": int(len(df)),
               "means": {k: {m: float(t.loc["MEAN", m])
                             for m in ("auc", "logloss", "brier", "acc")}
                         for k, t in tables.items()},
               "per_fold": {k: t.drop(index="MEAN")["auc"].to_dict()
                            for k, t in tables.items()},
               "failed": failed}

    if baseline in preds:
        print(f"\nPaired bootstrap over matches, vs {baseline}:")
        summary["vs_baseline"] = {}
        for name in tables:
            if name == baseline:
                continue
            b = paired_bootstrap(preds[name], preds[baseline])
            summary["vs_baseline"][name] = b
            verdict = ("BETTER" if b["ci_low"] > 0 else
                       "WORSE" if b["ci_high"] < 0 else "no difference")
            print(f"  {name:<34} {b['delta_auc']:+.4f}  "
                  f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  {verdict}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, args.out)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
