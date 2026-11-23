"""Chooses the rule for the final test from the ablation runs (P1) only.

    python scripts/select_rule.py            prints the result
    python scripts/select_rule.py --write    also writes configs/followup.yaml, which scripts/run_main.py reads for F1

Reads data/ablation (pack your own runs with scripts/pack_records.py first) and, as references, the SAC and IER
runs of the same seeds from the analysis of IER (data/analysis/curves.csv.gz).

The rule, fixed before any P1 run:
  1. For each rule and task, the mean over seeds 0 to 9 of the paired AUC difference against IER.
  2. A rule is eligible if this mean is at least -0.02 on both tasks and its mean over the two tasks is above zero.
  3. The eligible rule with the largest mean over the two tasks is chosen. A tie goes to the rule with the smaller
     share of replayed steps.
  4. If no rule is eligible, chosen_arm is none and no final test runs.
AUC is the normalized AUC of iers/metrics.py with the 1M-step normalizer of the analysis of IER.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TASKS = ["HalfCheetah-v4", "walker-walk"]
RULES = ["IER_CAP", "IER_STOP", "IER_MARGIN", "IER_PREFIX"]
SEEDS = list(range(10))
TOLERANCE = -0.02
BUDGET = 1_000_000


def auc(steps: np.ndarray, returns: np.ndarray, normalizer: float) -> float:
    return float(np.trapezoid(returns, steps) / (steps[-1] - steps[0]) / normalizer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    norm = {(x["task"], x["budget"]): x["value"] for x in json.loads((DATA / "analysis" / "normalizers.json").read_text())}
    cur = pd.read_csv(DATA / "analysis" / "curves.csv.gz")
    ev = pd.read_csv(DATA / "ablation" / "eval.csv.gz")
    tr = pd.read_csv(DATA / "ablation" / "train.csv.gz")
    table = {}
    for task in TASKS:
        n = norm[(task, BUDGET)]
        ref = {}
        for arm in ("BASE", "IER"):
            ref[arm] = []
            for s in SEEDS:
                g = cur[cur.run_id == f"A1/{task}__default__sac__{arm}__seed{s}"].groupby("step")["return"].mean()
                ref[arm].append(auc(g.index.to_numpy(float), g.to_numpy(float), n))
        table[task] = {}
        for rule in RULES:
            aucs, shares = [], []
            for s in SEEDS:
                g = ev[(ev.task == task) & (ev.arm == rule) & (ev.seed == s)].groupby("total_steps")["return"].mean()
                if g.index.max() < BUDGET - 20_000:
                    raise SystemExit(f"{task} {rule} seed {s} ends at step {g.index.max()}, before the budget")
                aucs.append(auc(g.index.to_numpy(float), g.to_numpy(float), n))
                x = tr[(tr.task == task) & (tr.arm == rule) & (tr.seed == s)]
                shares.append(x.replay_steps.sum() / x.length.sum())
            table[task][rule] = {"minus_IER": float(np.mean(np.array(aucs) - np.array(ref["IER"]))),
                                 "minus_BASE": float(np.mean(np.array(aucs) - np.array(ref["BASE"]))),
                                 "replay_share": float(np.mean(shares))}
    summary = {}
    for rule in RULES:
        per = [table[t][rule]["minus_IER"] for t in TASKS]
        summary[rule] = {"mean_minus_IER": float(np.mean(per)),
                         "replay_share": float(np.mean([table[t][rule]["replay_share"] for t in TASKS])),
                         "eligible": bool(min(per) >= TOLERANCE and np.mean(per) > 0)}
    eligible = [r for r in RULES if summary[r]["eligible"]]
    chosen = (sorted(eligible, key=lambda r: (-round(summary[r]["mean_minus_IER"], 9), summary[r]["replay_share"]))[0]
              if eligible else "none")
    print("| Rule | HalfCheetah minus IER | walker-walk minus IER | Mean minus IER | Replay share | Eligible |")
    print("|---|---|---|---|---|---|")
    for rule in RULES:
        print(f"| {rule} | {table['HalfCheetah-v4'][rule]['minus_IER']:+.3f} | {table['walker-walk'][rule]['minus_IER']:+.3f} | "
              f"{summary[rule]['mean_minus_IER']:+.3f} | {summary[rule]['replay_share']:.3f} | {summary[rule]['eligible']} |")
    print(f"\nChosen: {chosen}")
    if args.write:
        (ROOT / "configs" / "followup.yaml").write_text(
            "# Written by scripts/select_rule.py from the ablation runs (P1) only.\n" f"chosen_arm: {chosen}\n", encoding="utf-8")
        print("wrote configs/followup.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
