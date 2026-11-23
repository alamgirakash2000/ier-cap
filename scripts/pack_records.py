"""Packs the run records into the compressed tables under data/.

    python scripts/pack_records.py                       packs results/raw/P1 and results/raw/F1
    python scripts/pack_records.py --analysis PATH       also copies the summary records of the analysis of IER

Every run folder under results/raw/<item>/ holds eval.csv (one row per test episode) and train.csv (one row per
training episode). This script stacks them into one table per item and file, with the task, arm and seed of each
run, so that the paper scripts need neither the run folders nor the training code:

    data/ablation/eval.csv.gz, data/ablation/train.csv.gz        item P1, seeds 0 to 9
    data/final_test/eval.csv.gz, data/final_test/train.csv.gz    item F1, seeds 20 to 39

With --analysis, PATH is the root of the companion repository of the analysis of IER. The script copies the
summary tables that the paper uses from its results/aggregated folder, and the training records of the IER runs
on HalfCheetah and walker-walk (item A1) for Fig. 2a, into data/analysis/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ITEMS = {"P1": "ablation", "F1": "final_test"}
ANALYSIS_FILES = ["curves.csv", "runs.csv", "repeats.csv", "contrasts.csv", "gains.csv", "normalizers.json"]


def parse_run(name: str) -> dict:
    task, condition, learner, arm, seed = name.split("__")
    return {"task": task, "condition": condition, "learner": learner, "arm": arm, "seed": int(seed.removeprefix("seed"))}


def stack(folder: Path, file: str) -> pd.DataFrame:
    parts = []
    for run in sorted(p for p in folder.iterdir() if p.is_dir()):
        table = pd.read_csv(run / file)
        meta = parse_run(run.name)
        for key in reversed(list(meta)):
            table.insert(0, key, meta[key])
        parts.append(table)
    if not parts:
        raise SystemExit(f"No run folders in {folder}")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--analysis", type=Path, default=None)
    args = parser.parse_args()
    for item, name in ITEMS.items():
        folder = args.raw / item
        if not folder.exists():
            print(f"skip {item}: {folder} does not exist")
            continue
        out = ROOT / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        for file in ("eval.csv", "train.csv"):
            table = stack(folder, file)
            table.to_csv(out / f"{file}.gz", index=False, compression={"method": "gzip", "mtime": 0})
            print(f"{item} {file}: {table[['task', 'arm', 'seed']].drop_duplicates().shape[0]} runs, {len(table)} rows")
    if args.analysis is not None:
        agg = args.analysis / "results" / "aggregated"
        out = ROOT / "data" / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        for file in ANALYSIS_FILES:
            if file.endswith(".csv"):
                pd.read_csv(agg / file).to_csv(out / f"{file}.gz", index=False,
                                               compression={"method": "gzip", "mtime": 0})
            else:
                shutil.copyfile(agg / file, out / file)
        a1 = args.analysis / "results" / "raw" / "A1"
        parts = []
        for task in ("HalfCheetah-v4", "walker-walk"):
            for seed in range(20):
                run = a1 / f"{task}__default__sac__IER__seed{seed}"
                table = pd.read_csv(run / "train.csv")
                table.insert(0, "seed", seed)
                table.insert(0, "task", task)
                parts.append(table)
        pd.concat(parts, ignore_index=True).to_csv(out / "a1_ier_train.csv.gz", index=False,
                                                   compression={"method": "gzip", "mtime": 0})
        print("analysis tables copied to data/analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
