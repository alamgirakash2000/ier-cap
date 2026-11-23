"""Builds every figure and table of the paper from the records in data/.

    python paper/make_paper.py            writes paper/out/figures, paper/out/tables and paper/out/numbers.json

Runs on a CPU in about two minutes. Each bootstrap uses a fixed random seed, so the output is the same on
every run. Figure and table names follow the paper:

    Fig. 2    figures/Fig2.pdf     replay share, retention and the gain of IER (analysis of IER)
    Fig. 3    figures/Fig3.pdf     learning curves and replay share of the final test
    Fig. 4    figures/Fig4.pdf     mean AUC against the budget
    Fig. A1   figures/FigA1.pdf    paired differences of the final test
    Table 2   tables/tab_main.tex
    Table 3   tables/tab_paired.tex
    Table A1  tables/tab_app_contrasts.tex
    Table A2  tables/tab_app_windows.tex
    Table A3  tables/tab_ablation.tex
    Table C1  tables/tab_app_effect.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402

import style as C  # noqa: E402
from iers.stats import paired_bootstrap  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "paper" / "out"
FIG, TAB = OUT / "figures", OUT / "tables"

TASKS = ["HalfCheetah-v4", "walker-walk", "cheetah-run", "cartpole-swingup"]
DEV_TASKS = ["HalfCheetah-v4", "walker-walk"]
ARMS = ["BASE", "IER", "IER_CAP"]
RULES = ["IER_STOP", "IER_MARGIN", "IER_PREFIX", "IER_CAP"]
TEST_SEEDS = list(range(20, 40))
DEV_SEEDS = list(range(10))
BUDGET = 1_000_000
WINDOWS = [100_000, 200_000, 300_000, 500_000, 1_000_000]
TL = {"HalfCheetah-v4": "HalfCheetah", "walker-walk": "walker-walk", "cheetah-run": "cheetah-run",
      "cartpole-swingup": "cartpole-swingup", "Hopper-v4": "Hopper", "finger-turn_hard": "finger-turn_hard"}
AL = {"BASE": "SAC", "IER": "IER", "IER_CAP": "IER-CAP"}
OI = C.OI
COL = {"BASE": OI["black"], "IER": OI["blue"], "IER_CAP": OI["vermillion"]}
CONTRASTS = [("IER_CAP", "IER"), ("IER_CAP", "BASE"), ("IER", "BASE")]


# ---------------------------------------------------------------- metrics (as in iers/metrics.py)

def curve(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Steps and mean test return of each test of one run."""
    g = table.groupby("total_steps", sort=True)["return"].mean()
    return g.index.to_numpy(np.float64), g.to_numpy(np.float64)


def final_return(returns: np.ndarray) -> float:
    return float(np.mean(returns[-5:]))


def auc(steps: np.ndarray, returns: np.ndarray, normalizer: float) -> float:
    return float(np.trapezoid(returns, steps) / (steps[-1] - steps[0]) / normalizer)


def sign(x: float) -> str:
    v = f"{x:+.3f}"
    return v.replace("-", "$-$") if x < 0 else v


def plain(x: float, d: int = 3) -> str:
    v = f"{x:.{d}f}"
    return v.replace("-", "$-$")


# ---------------------------------------------------------------- tests

def sign_flip_p(values: np.ndarray) -> float:
    """Exact two-sided sign-flip test of a zero mean, over all 2^n sign patterns."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    observed = abs(values.mean())
    total, hits, chunk = 2**n, 0, 2**16
    bits = np.arange(n)
    for start in range(0, total, chunk):
        codes = np.arange(start, min(total, start + chunk))[:, None]
        signs = 1.0 - 2.0 * ((codes >> bits) & 1)
        hits += int(np.sum(np.abs(signs @ values) / n >= observed - 1e-12))
    return hits / total


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    m, out, running = len(order), {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[key]))
        out[key] = running
    return out


# ---------------------------------------------------------------- final test

def final_test():
    ev = pd.read_csv(DATA / "final_test" / "eval.csv.gz")
    tr = pd.read_csv(DATA / "final_test" / "train.csv.gz")
    runs, norm, curves = {}, {}, {}
    for t in TASKS:
        for a in ARMS:
            for s in TEST_SEEDS:
                st, r = curve(ev[(ev.task == t) & (ev.arm == a) & (ev.seed == s)])
                curves[(t, a, s)] = (st, r)
        norm[t] = float(np.mean([final_return(curves[(t, "BASE", s)][1]) for s in TEST_SEEDS]))
    grid = np.arange(10_000, 1_000_001, 10_000)
    for t in TASKS:
        for a in ARMS:
            for s in TEST_SEEDS:
                st, r = curves[(t, a, s)]
                x = tr[(tr.task == t) & (tr.arm == a) & (tr.seed == s)]
                tot = x["total_steps"].to_numpy()
                rep = x["replay_steps"].cumsum().to_numpy()
                win = {}
                for lo, hi in [(0, 100e3), (100e3, 300e3), (300e3, 1e6)]:
                    m = (tot > lo) & (tot <= hi)
                    win[f"{int(lo / 1e3)}-{int(hi / 1e3)}k"] = float(x.loc[m, "replay_steps"].sum() / x.loc[m, "length"].sum())
                runs[(t, a, s)] = dict(
                    auc={W: auc(st[st <= W], r[st <= W], norm[t]) for W in WINDOWS},
                    final=final_return(r), share=float(x["replay_steps"].sum() / x["length"].sum()), win=win,
                    run_share=np.interp(grid, tot, rep / tot), max_share=float(np.max((rep / tot)[tot >= 20_000])))
    return runs, norm, curves


def cell(runs, t, a, key, W=BUDGET):
    v = np.array([runs[(t, a, s)][key][W] if key == "auc" else runs[(t, a, s)][key] for s in TEST_SEEDS])
    return v


def paired(runs, W=BUDGET):
    out = {}
    for t in TASKS:
        for a, b in CONTRASTS:
            d = cell(runs, t, a, "auc", W) - cell(runs, t, b, "auc", W)
            out[(t, a, b)] = (paired_bootstrap(d, seed=0), d)
    return out


def pooled(runs, W):
    out = {}
    for a, b in CONTRASTS:
        rng = np.random.default_rng(0)
        boot = np.zeros(10_000)
        for t in TASKS:
            d = cell(runs, t, a, "auc", W) - cell(runs, t, b, "auc", W)
            boot += d[rng.integers(0, 20, size=(10_000, 20))].mean(1)
        boot /= len(TASKS)
        m = float(np.mean([(cell(runs, t, a, "auc", W) - cell(runs, t, b, "auc", W)).mean() for t in TASKS]))
        out[(a, b)] = (m, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return out


def prob_improvement(runs):
    """Chance that a random run of one method beats a random run of the other on the same task, mean over tasks,
    with a bootstrap interval that resamples the runs of each method."""
    allc = {t: {a: cell(runs, t, a, "auc") for a in ARMS} for t in TASKS}

    def poi(X, Y):
        return float(np.mean([np.mean((X[t][:, None] > Y[t][None, :]) + 0.5 * (X[t][:, None] == Y[t][None, :]))
                              for t in TASKS]))
    rng = np.random.default_rng(0)
    res = {}
    for a, b in [("IER_CAP", "BASE"), ("IER", "BASE"), ("IER_CAP", "IER")]:
        m = poi({t: allc[t][a] for t in TASKS}, {t: allc[t][b] for t in TASKS})
        bs = []
        for _ in range(4000):
            X = {t: allc[t][a][rng.integers(0, 20, 20)] for t in TASKS}
            Y = {t: allc[t][b][rng.integers(0, 20, 20)] for t in TASKS}
            bs.append(poi(X, Y))
        res[(a, b)] = (round(m, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))
    return res


# ---------------------------------------------------------------- ablation

def ablation():
    norm = {(x["task"], x["budget"]): x["value"] for x in json.loads((DATA / "analysis" / "normalizers.json").read_text())}
    cur = pd.read_csv(DATA / "analysis" / "curves.csv.gz")
    runs = pd.read_csv(DATA / "analysis" / "runs.csv.gz")
    ev = pd.read_csv(DATA / "ablation" / "eval.csv.gz")
    tr = pd.read_csv(DATA / "ablation" / "train.csv.gz")

    def ref_auc(exp, arm, t, s):
        d = cur[cur.run_id == f"{exp}/{t}__default__sac__{arm}__seed{s}"]
        g = d.groupby("step")["return"].mean()
        return auc(g.index.to_numpy(float), g.to_numpy(float), norm[(t, BUDGET)])

    abl, raw = {}, {}
    for t in DEV_TASKS:
        A = {"BASE": [ref_auc("A1", "BASE", t, s) for s in DEV_SEEDS], "IER": [ref_auc("A1", "IER", t, s) for s in DEV_SEEDS],
             "IER3": [ref_auc("D1", "IER3", t, s) for s in DEV_SEEDS],
             "IER_REL": [ref_auc("R1", "IER_REL", t, s) for s in DEV_SEEDS]}
        share = {"BASE": 0.0}
        for k, exp in [("IER", "A1"), ("IER3", "D1"), ("IER_REL", "R1")]:
            share[k] = float(runs[(runs.experiment == exp) & (runs.arm == k) & (runs.task == t) & (runs.seed < 10)
                                  & (runs.learner == "sac")].repetition_share.mean())
        final = {}
        for k in RULES:
            A[k], v, f = [], [], []
            for s in DEV_SEEDS:
                st, r = curve(ev[(ev.task == t) & (ev.arm == k) & (ev.seed == s)])
                A[k].append(auc(st, r, norm[(t, BUDGET)]))
                f.append(final_return(r))
                x = tr[(tr.task == t) & (tr.arm == k) & (tr.seed == s)]
                v.append(x.replay_steps.sum() / x.length.sum())
            share[k], final[k] = float(np.mean(v)), float(np.mean(f))
        abl[t] = {k: dict(auc=float(np.mean(v)), share=share.get(k, 0.0)) for k, v in A.items()}
        for k in RULES:
            abl[t][k]["final"] = final[k]
            abl[t][k]["minus_IER"] = float(np.mean(np.array(A[k]) - np.array(A["IER"])))
            abl[t][k]["minus_BASE"] = float(np.mean(np.array(A[k]) - np.array(A["BASE"])))
        raw[t] = A
    rng = np.random.default_rng(0)
    change = {}
    for k in ["BASE", "IER3", "IER_REL", "IER_STOP", "IER_MARGIN", "IER_PREFIX", "IER_CAP"]:
        per = [np.array(raw[t][k]) - np.array(raw[t]["IER"]) for t in DEV_TASKS]
        boot = np.zeros(10_000)
        for p in per:
            boot += p[rng.integers(0, 10, size=(10_000, 10))].mean(1)
        boot /= 2
        change[k] = (float(np.mean([p.mean() for p in per])), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return abl, change


def selection(abl):
    """The selection rule fixed before the ablation runs, as in scripts/select_rule.py."""
    summary = {}
    for k in RULES:
        per = [abl[t][k]["minus_IER"] for t in DEV_TASKS]
        summary[k] = dict(mean_minus_IER=float(np.mean(per)), eligible=bool(min(per) >= -0.02 and np.mean(per) > 0),
                          share=float(np.mean([abl[t][k]["share"] for t in DEV_TASKS])))
    eligible = [k for k in RULES if summary[k]["eligible"]]
    chosen = sorted(eligible, key=lambda k: (-round(summary[k]["mean_minus_IER"], 9), summary[k]["share"]))[0] if eligible else "none"
    return chosen, summary


# ---------------------------------------------------------------- tables

def rows(label, cs, last=False):
    out = [label + " & " + " & ".join(c[0] for c in cs) + r" \\",
           " & " + " & ".join(c[1] for c in cs) + r" \\"]
    if not last:
        out.append(r"\addlinespace[2pt]")
    return out


def interval_cell(m, lo, hi):
    mm = r"\textbf{" + sign(m) + "}" if (lo > 0 or hi < 0) else sign(m)
    return mm, r"{\footnotesize " + f"[{sign(lo)}, {sign(hi)}]" + "}"


def table_main(runs):
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Normalized AUC} & & \\ \cmidrule(lr){3-4}",
             r"Task & Method & 1M steps & First 300k steps & Final return & Replay share \\", r"\midrule"]

    def bold(v, best):
        return r"\textbf{" + v + "}" if v == best else v
    stats = {}
    for t in TASKS:
        for a in ARMS:
            A1, A3, F = cell(runs, t, a, "auc"), cell(runs, t, a, "auc", 300_000), cell(runs, t, a, "final")
            stats[(t, a)] = dict(auc=A1.mean(), auc_sd=A1.std(ddof=1), auc300=A3.mean(), auc300_sd=A3.std(ddof=1),
                                 final=F.mean(), final_sd=F.std(ddof=1), share=cell(runs, t, a, "share").mean())
    for t in TASKS:
        b1 = max(f"{stats[(t, a)]['auc']:.3f}" for a in ARMS)
        b3 = max(f"{stats[(t, a)]['auc300']:.3f}" for a in ARMS)
        bf = max(f"{stats[(t, a)]['final']:.0f}" for a in ARMS)
        for j, a in enumerate(ARMS):
            e = stats[(t, a)]
            share = 0.0 if a == "BASE" else e["share"]
            v1, v3, vf = f"{e['auc']:.3f}", f"{e['auc300']:.3f}", f"{e['final']:.0f}"
            name = TL[t] if j == 0 else ""
            lines.append(f"{name} & {AL[a]} & {bold(v1, b1)} $\\pm$ {e['auc_sd']:.3f} & "
                         f"{bold(v3, b3)} $\\pm$ {e['auc300_sd']:.3f} & "
                         f"{bold(vf, bf)} $\\pm$ {e['final_sd']:.0f} & {share:.3f} \\\\")
        lines.append(r"\midrule")
    m1 = {a: sum(stats[(t, a)]["auc"] for t in TASKS) / 4 for a in ARMS}
    m3 = {a: sum(stats[(t, a)]["auc300"] for t in TASKS) / 4 for a in ARMS}
    ms = {a: (0.0 if a == "BASE" else sum(stats[(t, a)]["share"] for t in TASKS) / 4) for a in ARMS}
    b1 = max(f"{v:.3f}" for v in m1.values())
    b3 = max(f"{v:.3f}" for v in m3.values())
    for j, a in enumerate(ARMS):
        v1, v3, name = f"{m1[a]:.3f}", f"{m3[a]:.3f}", "Mean" if j == 0 else ""
        lines.append(f"{name} & {AL[a]} & {bold(v1, b1)} & {bold(v3, b3)} & & {ms[a]:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_main.tex").write_text("\n".join(lines) + "\n")
    return stats, m1, m3, ms


def table_paired(pr, pool, poi):
    keys = [("IER_CAP", "BASE"), ("IER", "BASE"), ("IER_CAP", "IER")]
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Task & IER-CAP minus SAC & IER minus SAC & IER-CAP minus IER \\", r"\midrule"]
    for t in TASKS:
        cs = [interval_cell(pr[(t, a, b)][0].mean, pr[(t, a, b)][0].low, pr[(t, a, b)][0].high) for a, b in keys]
        L += rows(TL[t], cs, last=(t == TASKS[-1]))
    L.append(r"\midrule")
    L += rows("Four tasks", [interval_cell(*pool[k]) for k in keys])
    cs = []
    for k in keys:
        m, lo, hi = poi[k]
        mm = r"\textbf{" + f"{m:.2f}" + "}" if (lo > 0.5 or hi < 0.5) else f"{m:.2f}"
        cs.append((mm, r"{\footnotesize " + f"[{lo:.2f}, {hi:.2f}]" + "}"))
    L += rows("Prob.\\ of improvement", cs, last=True)
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_paired.tex").write_text("\n".join(L) + "\n")


def table_contrasts(pr, pool):
    p = {f"{t}: {a} minus {b}": sign_flip_p(pr[(t, a, b)][1]) for t in TASKS for a, b in CONTRASTS}
    h = holm(p)
    name = {"IER_CAP": "IER-CAP", "IER": "IER", "BASE": "SAC"}
    L = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrcrrr@{}}", r"\toprule",
         r"Task & Contrast & Mean & 95\% interval & Wins & $p$ & Holm $p$ \\", r"\midrule"]
    for t in TASKS:
        for i, (a, b) in enumerate(CONTRASTS):
            r = pr[(t, a, b)][0]
            m = r"\textbf{" + sign(r.mean) + "}" if r.clear else sign(r.mean)
            key = f"{t}: {a} minus {b}"
            L.append(f"{TL[t] if i == 0 else ''} & {name[a]} $-$ {name[b]} & {m} & [{sign(r.low)}, {sign(r.high)}] & "
                     f"{r.win_rate:.2f} & {p[key]:.3f} & {h[key]:.3f} \\\\")
        L.append(r"\addlinespace[2pt]")
    L.append(r"\midrule")
    m, lo, hi = pool[("IER_CAP", "IER")]
    L.append(f"Four tasks & IER-CAP $-$ IER & {sign(m)} & [{sign(lo)}, {sign(hi)}] & & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_app_contrasts.tex").write_text("\n".join(L) + "\n")
    return p, h


def table_windows(pools):
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Window & IER-CAP minus IER & IER-CAP minus SAC & IER minus SAC \\", r"\midrule"]
    names = {100_000: "First 100k steps", 200_000: "First 200k steps", 300_000: "First 300k steps",
             500_000: "First 500k steps", 1_000_000: "Full budget of 1M steps"}
    for W in WINDOWS:
        L += rows(names[W], [interval_cell(*pools[W][k]) for k in CONTRASTS], last=(W == BUDGET))
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_app_windows.tex").write_text("\n".join(L) + "\n")


def table_ablation(abl, change):
    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Normalized AUC} & Mean change & Replay \\",
         r"\cmidrule(lr){3-4}",
         r"Variant & Limit on the replays & HalfCheetah & walker-walk & vs.\ IER & share \\", r"\midrule"]

    def row(name, what, k):
        hc, ww = abl["HalfCheetah-v4"][k], abl["walker-walk"][k]
        share = (hc["share"] + ww["share"]) / 2
        if k == "IER":
            return f"{name} & {what} & {hc['auc']:.3f} & {ww['auc']:.3f} & reference & {share:.3f} \\\\"
        m, lo, hi = change[k]
        return (f"{name} & {what} & {hc['auc']:.3f} & {ww['auc']:.3f} & {sign(m)} & {share:.3f} \\\\\n"
                + r" & & & & {\footnotesize " + f"[{sign(lo)}, {sign(hi)}]" + r"} & \\")
    L.append(row("SAC", "no replays", "BASE"))
    L.append(row("IER", "no limit", "IER"))
    L.append(r"\midrule")
    L.append(row("IER-REL", "released schedule", "IER_REL"))
    L.append(row("IER3", "repeat count of 3", "IER3"))
    L.append(r"\midrule")
    for k, nm, w in [("IER_STOP", "IER-STOP", "return of the last replay"), ("IER_MARGIN", "IER-MARGIN", "size of the new best"),
                     ("IER_PREFIX", "IER-PREFIX", "length of a replay"), ("IER_CAP", r"\textbf{IER-CAP}", "share of steps so far")]:
        L.append(row(nm, w, k))
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_ablation.tex").write_text("\n".join(L) + "\n")


def table_effect():
    g = pd.read_csv(DATA / "analysis" / "gains.csv.gz")
    c = pd.read_csv(DATA / "analysis" / "contrasts.csv.gz")
    runs = pd.read_csv(DATA / "analysis" / "runs.csv.gz")
    order = ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]
    sel = lambda d, label, t: d[(d.label == label) & (d.task == t) & (d.learner == "sac") & (d.condition == "default")
                                & (d.budget == BUDGET)].iloc[0]
    a1 = runs[(runs.experiment == "A1") & (runs.learner == "sac") & (runs.condition == "default")]
    L = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & HalfCheetah & Hopper & walker-walk & finger-turn\_hard \\", r"\midrule"]
    for arm, name in [("BASE", "BASE"), ("IER", "IER")]:
        L.append(f"Final return, {name} & " + " & ".join(f"{a1[(a1.task == t) & (a1.arm == arm)].final_return.mean():.0f}" for t in order) + r" \\")
    spec = [("IER minus BASE", g, "IER $-$ BASE"), ("IER_REL minus BASE", g, "IER-REL $-$ BASE"),
            ("IER3 minus BASE", g, "IER3 $-$ BASE"), ("IER minus IER_REL", c, "IER $-$ IER-REL"), ("IER3 minus IER", c, "IER3 $-$ IER")]
    for label, d, name in spec:
        vals = [sel(d, label, t) for t in order]
        top = [r"\textbf{" + plain(v.estimate) + "}" if (v.low > 0 or v.high < 0) else plain(v.estimate) for v in vals]
        low = [r"{\footnotesize [" + plain(v.low) + ", " + plain(v.high) + "]}" for v in vals]
        L.append(f"{name} & " + " & ".join(top) + r" \\")
        L.append(" & " + " & ".join(low) + r" \\")
    L += [r"\botrule", r"\end{tabular}"]
    (TAB / "tab_app_effect.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- figures

def fig2():
    tr = pd.read_csv(DATA / "analysis" / "a1_ier_train.csv.gz")
    rep = pd.read_csv(DATA / "analysis" / "repeats.csv.gz")
    con = pd.read_csv(DATA / "analysis" / "contrasts.csv.gz")
    r = rep[(rep.run_id.str.startswith("A1/")) & (rep.arm == "IER") & (rep.learner == "sac") & (rep.condition == "default")]
    fig, ax = C.strip(1.95, [1.05, 0.95, 1.35], legend=False, gaps=[0.55, 0.62], top=0.30, bottom=0.52)
    wins = [(0, 100e3), (100e3, 300e3), (300e3, 1e6)]
    labels = ["0 to\n100k", "100k to\n300k", "300k to\n1M"]
    x, w = np.arange(3), 0.36
    shares = {}
    for j, t in enumerate(["HalfCheetah-v4", "walker-walk"]):
        v = []
        for lo, hi in wins:
            per = []
            for s in range(20):
                d = tr[(tr.task == t) & (tr.seed == s)]
                m = (d.total_steps > lo) & (d.total_steps <= hi)
                per.append(d.loc[m, "replay_steps"].sum() / d.loc[m, "length"].sum())
            v.append(float(np.mean(per)))
        shares[t] = v
        ax[0].bar(x + (j - 0.5) * w, v, w, color=C.TASK_STYLE[t]["color"], label=TL[t])
    ax[0].set_xticks(x, labels, fontsize=7)
    ax[0].set_ylim(0, 0.75)
    ax[0].set_ylabel("Replay share")
    ax[0].legend(loc="upper right", fontsize=7, handlelength=1.0)
    C.panel_label(ax[0], "a")
    edges = [0, 25e3, 50e3, 100e3, 200e3, 400e3, 1e6]
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    for t in ["Hopper-v4", "walker-walk", "HalfCheetah-v4"]:
        med = []
        for i in range(len(edges) - 1):
            m = r[(r.task == t) & (r.total_steps > edges[i]) & (r.total_steps <= edges[i + 1])]
            per = m.groupby("seed").ratio.median()
            med.append(per.median() if len(per) >= 5 else np.nan)
        st = C.TASK_STYLE[t]
        ax[1].plot(mids, np.clip(med, -0.2, 1.1), color=st["color"], marker=st["marker"], label=TL[t])
    ax[1].set_xscale("log")
    ax[1].set_xlim(10e3, 1e6)
    ax[1].xaxis.set_major_formatter(mt.FuncFormatter(C.steps_formatter))
    ax[1].xaxis.set_minor_formatter(mt.NullFormatter())
    ax[1].set_ylim(-0.2, 1.1)
    ax[1].set_ylabel("Retention")
    ax[1].set_xlabel("Environment steps", labelpad=1)
    ax[1].legend(loc="upper right", fontsize=7, handlelength=1.4)
    C.panel_label(ax[1], "b")
    order = ["HalfCheetah-v4", "Hopper-v4", "walker-walk", "finger-turn_hard"]
    v = []
    for t in order:
        row = con[(con.label == "IER minus BASE") & (con.task == t) & (con.learner == "sac") & (con.condition == "default")
                  & (con.budget == BUDGET)].iloc[0]
        v.append((row.estimate, row.low, row.high))
    v = np.array(v)
    ax[2].bar(np.arange(4), v[:, 0], 0.6, color=[C.TASK_STYLE[t]["color"] for t in order])
    ax[2].errorbar(np.arange(4), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="black",
                   elinewidth=0.7, capsize=2)
    ax[2].axhline(0, color="black", lw=0.6)
    ax[2].set_xticks(np.arange(4), ["Half-\nCheetah", "Hopper", "walker-\nwalk", "finger-\nturn"], fontsize=7)
    ax[2].set_ylabel("AUC of IER minus SAC")
    C.panel_label(ax[2], "c")
    fig.savefig(FIG / "Fig2.pdf")
    plt.close(fig)
    return shares


def fig3(runs, curves):
    rng = np.random.default_rng(0)
    fig, axs = C.grid(2, 4, 3.55, legend=True, wgap=0.40, hgap=0.42, left=0.50, top=0.42)
    grid = np.arange(10_000, 1_000_001, 10_000)
    for i, t in enumerate(TASKS):
        ax = axs[i]
        for a in ARMS:
            M = np.array([curves[(t, a, s)][1] for s in TEST_SEEDS])
            steps = curves[(t, a, TEST_SEEDS[0])][0]
            idx = rng.integers(0, M.shape[0], size=(2000, M.shape[0]))
            means = M[idx].mean(axis=1)
            lo, hi = np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)
            ax.fill_between(steps, lo, hi, color=COL[a], alpha=C.BAND_ALPHA, lw=0)
            ax.plot(steps, M.mean(0), color=COL[a], lw=1.0, label=AL[a])
        ax.xaxis.set_major_locator(mt.FixedLocator([0, 5e5, 1e6]))
        ax.xaxis.set_major_formatter(mt.FuncFormatter(C.steps_formatter))
        ax.set_xlim(0, 1e6)
        ax.yaxis.set_major_locator(mt.MaxNLocator(4))
        if i == 0:
            ax.set_ylabel("Test return")
        C.panel_label(ax, "abcd"[i], TL[t])
        ax2 = axs[4 + i]
        ax2.axhline(0.25, color=C.GRAY, ls="--", lw=0.8)
        for a in ["IER", "IER_CAP"]:
            ax2.plot(grid, np.mean([runs[(t, a, s)]["run_share"] for s in TEST_SEEDS], axis=0), color=COL[a], lw=1.0)
        ax2.set_ylim(0, 0.9)
        ax2.set_xlim(0, 1e6)
        ax2.xaxis.set_major_locator(mt.FixedLocator([0, 5e5, 1e6]))
        ax2.xaxis.set_major_formatter(mt.FuncFormatter(C.steps_formatter))
        ax2.yaxis.set_major_locator(mt.FixedLocator([0, 0.25, 0.5, 0.75]))
        if i == 0:
            ax2.set_ylabel("Replay share so far")
        ax2.set_xlabel("Environment steps")
        C.panel_label(ax2, "efgh"[i], TL[t])
    h = [plt.Line2D([], [], color=COL[a], lw=1.2) for a in ARMS] + [plt.Line2D([], [], color=C.GRAY, ls="--", lw=0.9)]
    C.top_legend(fig, h, ["SAC", "IER", "IER-CAP", r"budget $\beta = 0.25$"], ncol=4)
    fig.savefig(FIG / "Fig3.pdf")
    plt.close(fig)


def fig4(runs):
    ts = dict(C.TASK_STYLE)
    ts["cheetah-run"] = {"color": OI["blue"], "marker": "D"}
    ts["cartpole-swingup"] = {"color": OI["reddish_purple"], "marker": "v"}
    fig, ax = C.strip(2.05, [1, 1], legend=True, gaps=[0.55], left=0.52, bottom=0.46, top=0.42)
    xs, arms = [0, 1, 2], ["BASE", "IER_CAP", "IER"]
    means = {}
    for k, (W, title) in enumerate([(BUDGET, "Full budget of 1M steps"), (300_000, "First 300k steps")]):
        a = ax[k]
        for t in TASKS:
            m, lo, hi = [], [], []
            for arm in arms:
                x = cell(runs, t, arm, "auc", W)
                rng = np.random.default_rng(1)
                bm = x[rng.integers(0, 20, size=(4000, 20))].mean(1)
                m.append(x.mean())
                lo.append(np.percentile(bm, 2.5))
                hi.append(np.percentile(bm, 97.5))
            m, lo, hi = np.array(m), np.array(lo), np.array(hi)
            means[(W, t)] = dict(zip(arms, m.tolist()))
            st = ts[t]
            a.errorbar(xs, m, yerr=[m - lo, hi - m], color=st["color"], marker=st["marker"], lw=1.0, elinewidth=0.7,
                       capsize=1.8, ms=3.5)
        a.set_xticks(xs, ["0\nSAC", "0.25\nIER-CAP", "1\nIER"], fontsize=7)
        a.set_xlim(-0.3, 2.3)
        a.set_xlabel(r"Budget $\beta$", labelpad=1)
        if k == 0:
            a.set_ylabel("Normalized AUC")
        a.yaxis.set_major_locator(mt.MaxNLocator(5))
        C.panel_label(a, "ab"[k], title)
    h = [plt.Line2D([], [], color=ts[t]["color"], marker=ts[t]["marker"], lw=1.0, ms=3.5) for t in TASKS]
    C.top_legend(fig, h, [TL[t] for t in TASKS], ncol=4)
    fig.savefig(FIG / "Fig4.pdf")
    plt.close(fig)
    return means


def fig_a1(per):
    fig, ax = C.strip(2.15, [1, 1], legend=True, gaps=[0.35], left=0.95, bottom=0.42, top=0.42)
    con = [(("IER_CAP", "IER"), "IER-CAP minus IER", OI["vermillion"], "o"),
           (("IER_CAP", "BASE"), "IER-CAP minus SAC", OI["reddish_purple"], "s"),
           (("IER", "BASE"), "IER minus SAC", OI["blue"], "^")]
    for k, (W, title) in enumerate([(BUDGET, "Full budget of 1M steps"), (300_000, "First 300k steps")]):
        a = ax[k]
        for i, t in enumerate(TASKS):
            for j, (key, _, col, mk) in enumerate(con):
                r = per[W][(t, *key)][0]
                y = -i + (1 - j) * 0.22
                a.plot([r.low, r.high], [y, y], color=col, lw=1.0)
                a.plot(r.mean, y, marker=mk, ms=4, color=col, mfc=col if r.clear else "white", mew=1.0, ls="none")
        a.axvline(0, color="black", lw=0.6)
        a.set_yticks([-i for i in range(4)], [TL[t] for t in TASKS] if k == 0 else [""] * 4, fontsize=7)
        a.set_ylim(-3.5, 0.5)
        a.grid(axis="x", color=C.GRID, lw=0.5)
        a.grid(axis="y", visible=False)
        a.set_xlabel("Paired difference in normalized AUC", labelpad=1)
        a.xaxis.set_major_locator(mt.MultipleLocator(0.05))
        C.panel_label(a, "ab"[k], title)
    lims = [min(x.get_xlim()[0] for x in ax), max(x.get_xlim()[1] for x in ax)]
    for a in ax:
        a.set_xlim(*lims)
    h = [plt.Line2D([], [], color=c, marker=m, lw=1.0, ms=4) for _, _, c, m in con]
    C.top_legend(fig, h, [l for _, l, _, _ in con], ncol=3)
    fig.savefig(FIG / "FigA1.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    C.apply_style()
    runs, norm, curves = final_test()
    per = {W: paired(runs, W) for W in WINDOWS}
    pools = {W: pooled(runs, W) for W in WINDOWS}
    poi = prob_improvement(runs)
    stats, m1, m3, ms = table_main(runs)
    table_paired(per[BUDGET], pools[BUDGET], poi)
    p, h = table_contrasts(per[BUDGET], pools[BUDGET])
    table_windows(pools)
    abl, change = ablation()
    table_ablation(abl, change)
    chosen, summary = selection(abl)
    table_effect()
    shares = fig2()
    fig3(runs, curves)
    means = fig4(runs)
    fig_a1(per)

    numbers = {
        "normalizer": norm,
        "replay_share": {t: {a: float(cell(runs, t, a, "share").mean()) for a in ["IER", "IER_CAP"]} for t in TASKS},
        "fewer_replayed_steps_pct": {t: 100 * (1 - cell(runs, t, "IER_CAP", "share").mean() / cell(runs, t, "IER", "share").mean())
                                     for t in TASKS},
        "replay_share_by_window": {t: {a: {k: float(np.mean([runs[(t, a, s)]["win"][k] for s in TEST_SEEDS]))
                                           for k in runs[(t, a, TEST_SEEDS[0])]["win"]} for a in ["IER", "IER_CAP"]} for t in TASKS},
        "max_share_ier_cap_after_20k": max(runs[(t, "IER_CAP", s)]["max_share"] for t in TASKS for s in TEST_SEEDS),
        "mean_auc_1M": m1, "mean_auc_300k": m3,
        "pct_auc_1M_vs_SAC": {a: 100 * (m1[a] / m1["BASE"] - 1) for a in ARMS},
        "pct_auc_300k_ier_cap_vs_ier": {t: 100 * (stats[(t, "IER_CAP")]["auc300"] / stats[(t, "IER")]["auc300"] - 1) for t in TASKS},
        "pct_auc_300k_ier_cap_vs_ier_mean": 100 * (m3["IER_CAP"] / m3["IER"] - 1),
        "pooled": {str(W): {f"{a} minus {b}": pools[W][(a, b)] for a, b in CONTRASTS} for W in WINDOWS},
        "prob_improvement": {f"{a} over {b}": v for (a, b), v in poi.items()},
        "sign_flip_p": p, "holm_p": h,
        "ablation": {"chosen": chosen, "summary": summary, "mean_change_vs_IER": change},
        "analysis_replay_share_by_window": shares,
        "auc_against_budget": {f"{W}|{t}": v for (W, t), v in means.items()},
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
