"""The experiments of the study, in priority order.

    python scripts/run_main.py --item A1                          trains whatever of A1 is unfinished
    python scripts/run_main.py --items B1 C1 --tasks walker-walk  one grid over part of several items
    python scripts/run_main.py --list                             every item with its run count and env steps
    python scripts/run_main.py --item A1 --tasks walker-walk --out check/raw   retrain into a fresh folder

Reads the main seed count and the CrossQ budget from configs/prereg.yaml. A cell and seed belong to the first
item that lists them, so no run is ever trained twice. With A2 at 300k, the CrossQ BASE `default` cells of B2
are those of A2.

`--tasks` and `--share` split the work between GPUs by task, so that every arm compared within a task can run
on the same GPU type. Groups hold at most `--group-size` seeds, 20 by default (configs/hardware.yaml), and
shrink further if the free GPU memory requires it.

IER-CAP study:
    P1  ablation of four rules that limit the replays of IER, SAC, 1M steps, seeds 0 to 9 of HalfCheetah and
        walker-walk
    F1  final test of the chosen rule against BASE (SAC alone) and IER, SAC, 1M steps, new seeds 20 to 39, on
        HalfCheetah, walker-walk and the new tasks cheetah-run and cartpole-swingup. The chosen arm is read from
        configs/followup.yaml, written by scripts/select_rule.py from the P1 results only. Without that file,
        F1 is empty.

Items A1 to E1 are the runs of the earlier analysis of IER. They stay runnable here.

Exit codes as run_grid: 0 all finished, 1 a group failed (its runs keep a crash record), 75 stopped early.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_env  # noqa: E402

require_env()

import yaml  # noqa: E402

from iers import ROOT  # noqa: E402
from iers.specs import load_tasks, make_spec  # noqa: E402

PREREG_CONFIG = ROOT / "configs" / "prereg.yaml"
FOUR = ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]
THREE = ["HalfCheetah-v4", "walker-walk", "finger-turn_hard"]
LONG, SHORT = 1_000_000, 300_000
ORDER = ["A1", "B1", "C1", "R1", "A2", "D1", "B2", "B3"]
# E1 (random-episode replay) is exploratory.
EXPLORATORY = ["E1"]
FOLLOWUP_ITEMS = ["P1", "F1"]
ORDER = ORDER + EXPLORATORY + FOLLOWUP_ITEMS
KNOWN_TASKS = FOUR + ["cheetah-run", "cartpole-swingup"]
CANDIDATES = ["IER_CAP", "IER_STOP", "IER_MARGIN", "IER_PREFIX"]
ITEM_SEEDS = {"P1": list(range(10)), "F1": list(range(20, 40))}
FOLLOWUP_CONFIG = ROOT / "configs" / "followup.yaml"


def chosen_candidate() -> str | None:
    """The arm chosen from P1, or None before the choice is made."""
    if not FOLLOWUP_CONFIG.exists():
        return None
    arm = (yaml.safe_load(FOLLOWUP_CONFIG.read_text(encoding="utf-8")) or {}).get("chosen_arm")
    if arm in (None, "none"):
        return None  # the screen found no eligible candidate, so no confirmation runs
    if arm not in CANDIDATES:
        raise SystemExit(f"configs/followup.yaml names {arm!r}, which is not one of {CANDIDATES}")
    return arm


def item_cells(crossq_budget: int) -> dict[str, list[tuple]]:
    """(task, condition, learner, arm, budget) per item."""
    def grid(tasks, conditions, learner, arms, budget):
        return [(t, c, learner, a, budget) for t in tasks for c in conditions for a in arms]

    return {
        "A1": grid(FOUR, ["default"], "sac", ["BASE", "IER"], LONG),
        "B1": grid(THREE, ["default", "noise5", "dyn20"], "sac", ["BASE", "IER3"], SHORT),
        "C1": grid(THREE, ["default", "noise5"], "sac", ["DUP"], SHORT),
        "A2": grid(FOUR, ["default"], "crossq", ["BASE", "IER"], crossq_budget),
        "R1": grid(FOUR, ["default"], "sac", ["IER_REL"], LONG),
        "D1": grid(FOUR, ["default"], "sac", ["IER3"], LONG),
        "B2": grid(THREE, ["default", "noise5"], "crossq", ["BASE", "IER3"], SHORT),
        "B3": grid(THREE, ["noise10"], "sac", ["BASE", "IER3"], SHORT),
        "E1": grid(["walker-walk"], ["default"], "sac", ["IER3_RANDOM"], SHORT),
        "P1": grid(["HalfCheetah-v4", "walker-walk"], ["default"], "sac", CANDIDATES, LONG),
        "F1": (grid(["HalfCheetah-v4", "walker-walk", "cheetah-run", "cartpole-swingup"], ["default"], "sac",
                    ["BASE", "IER", chosen_candidate()], LONG) if chosen_candidate() else []),
    }


def item_specs(seeds: list[int], crossq_budget: int) -> dict[str, list]:
    tasks = load_tasks()
    taken_by: dict[tuple, str] = {}
    out: dict[str, list] = {}
    for item in ORDER:
        out[item] = []
        for task, condition, learner, arm, budget in item_cells(crossq_budget)[item]:
            for seed in ITEM_SEEDS.get(item, seeds):
                key = (task, condition, learner, arm, budget, seed)
                if key in taken_by:
                    continue
                taken_by[key] = item
                # DUP takes RN = 3, since it is compared with IER3.
                out[item].append(make_spec(item, task, condition, learner, arm, seed, budget, tasks[task]["rn"],
                                           rn=3 if arm == "DUP" else None))
    return out


def load_prereg() -> dict:
    if not PREREG_CONFIG.exists():
        raise SystemExit("configs/prereg.yaml does not exist. It holds the main seed count and the CrossQ budget.")
    return yaml.safe_load(PREREG_CONFIG.read_text(encoding="utf-8"))


def select(specs: dict[str, list], items: list[str], tasks: list[str] | None) -> list:
    """The specs of the given items, in priority order, on the given tasks only."""
    unknown = sorted(set(tasks or []) - set(KNOWN_TASKS))
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Known: {KNOWN_TASKS}")
    chosen = [s for item in ORDER if item in items for s in specs[item]]
    return [s for s in chosen if tasks is None or s.task in tasks]


def select_shares(specs: dict[str, list], shares: list[str]) -> list:
    """The union of several ITEMS:TASKS shares, for example 'A1,R1:Hopper-v4' and 'B1,C1:finger-turn_hard'."""
    chosen = []
    for share in shares:
        items, _, tasks = share.partition(":")
        if not tasks or set(items.split(",")) - set(ORDER):
            raise SystemExit(f"A share is ITEMS:TASKS with items from {ORDER}, got {share!r}")
        chosen += [s for s in select(specs, items.split(","), tasks.split(",")) if s not in chosen]
    return chosen


def longest_first(chosen: list) -> list:
    """Groups start in this order, so 1M-step groups go first and no long group starts last. Scheduling only."""
    return sorted(chosen, key=lambda s: -s.budget)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item", choices=ORDER)
    parser.add_argument("--items", nargs="+", choices=ORDER)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--share", action="append", default=[], metavar="ITEMS:TASKS",
                        help="repeatable, for example A1,R1:Hopper-v4. One grid over the union of the shares")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="only these main seeds, to split one item across machines")
    parser.add_argument("--arms", nargs="+", default=None, help="only these arms, to split one item across machines")
    parser.add_argument("--group-size", type=int, default=None, help="seeds per GPU process, default from hardware.yaml")
    parser.add_argument("--processes", type=int, default=None, help="GPU processes at once, default from hardware.yaml")
    parser.add_argument("--name", default=None, help="grid name for logs and job files")
    parser.add_argument("--out", type=Path, default=None,
                        help="folder for the run results, default results/raw. Use a fresh folder to retrain runs "
                             "whose saved results are already in results/raw")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    config = load_prereg()
    seeds = list(range(int(config["main_seeds"])))
    specs = item_specs(seeds, int(config["crossq_budget"]))
    items = args.items or ([args.item] if args.item else [])
    if args.list or not (items or args.share):
        for item in ORDER:
            chosen = select(specs, [item], args.tasks)
            steps = sum(s.budget for s in chosen)
            print(f"{item}: {len(chosen)} runs, {steps / 1e6:.1f}M env steps")
        return 0
    chosen = longest_first(select_shares(specs, args.share) if args.share else select(specs, items, args.tasks))
    if args.seeds is not None:
        chosen = [s for s in chosen if s.seed in set(args.seeds)]
    if args.arms is not None:
        chosen = [s for s in chosen if s.arm in set(args.arms)]
    if args.share:
        items = [item for item in ORDER if any(s.experiment == item for s in chosen)]
        args.tasks = sorted({s.task for s in chosen}, key=KNOWN_TASKS.index)
    if not chosen:
        print("Nothing selected.")
        return 0
    learners = {s.learner for s in chosen}
    if len(learners) != 1:
        raise SystemExit(f"One grid trains one learner, got {sorted(learners)}. Run the items separately.")
    from iers.grid import run_grid

    name = args.name or "_".join(items + ([t.split("-")[0].lower() for t in args.tasks] if args.tasks else []))
    kwargs = {} if args.out is None else {"results_root": args.out, "ckpt_root": args.out.parent / "checkpoints"}
    return run_grid(name, chosen, group_size=args.group_size, processes=args.processes, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
