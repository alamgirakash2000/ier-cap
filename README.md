# IER-CAP: A Replay Budget for Instant Episode Repetition

This repository holds the code, the run records and the figure scripts for the paper
"IER-CAP: A Replay Budget for Instant Episode Repetition in Off-Policy Reinforcement Learning".

Instant episode repetition (IER) runs the actions of the best episode so far again, open-loop, in the next few
episodes. IER-CAP keeps this trigger rule but starts a replay only while the replayed steps so far stay below a
fixed share β of all steps so far. A replay that may not start becomes an ordinary policy episode. The rule needs
one counter and one comparison per episode, it leaves the learner unchanged, and the share of replayed steps
never exceeds β by more than one episode. With β = 0 the rule gives back the learner alone, and with β = 1 it
gives back IER.

## Results at a glance

Soft actor-critic (SAC) with 1M steps per run, new seeds 20 to 39, on HalfCheetah, walker-walk, cheetah-run and
cartpole-swingup. Values are means over the four tasks. The normalized AUC divides the area under the learning
curve by the final return of SAC on the same task.

| | SAC | IER | IER-CAP |
|---|---|---|---|
| Normalized AUC, 1M steps | 0.837 | 0.850 | **0.866** |
| Normalized AUC, first 300k steps | 0.587 | 0.561 | **0.614** |
| Share of steps that run stored actions | 0 | 0.197 | 0.126 |

IER-CAP replays 22% to 41% fewer steps than IER and raises the AUC of SAC by 3.4% over 1M steps. Over the first
300k steps its AUC is 9.3% above that of IER on average and up to 19.8% above it on a single task.

## The rule

The rule sits in `iers/repetition.py`. At the start of each episode, with `t` the steps taken so far and `M` the
steps of them that ran stored actions:

```python
may_replay = (t == 0) or (M < beta * t)
```

If the check fails while IER would replay, the episode runs the policy and still counts as one of the repeated
episodes. The check is made once per episode, so a replay that starts always runs to its end. The setting of
IER-CAP and of the three other rules of the ablation is in `ARM_VARIANT` in the same file. IER-CAP uses β = 0.25.

## Repository layout

```
iers/              training code: SAC, the repetition controller with IER and IER-CAP, batched training
third_party/ier/   the released code of IER, unmodified, whose networks our SAC uses
configs/           tasks, repeat counts, seeds, hardware and the rule chosen for the final test
scripts/           training, selection of the rule and packing of the run records
paper/             make_paper.py builds every figure and table of the paper from data/
data/              the run records behind the paper
paper_assets/      fonts of the figures
```

## Setup

```bash
conda env create -f environment.yml
conda activate iercap
python scripts/check_env.py
```

The environment installs the exact package versions of every run (`requirements-lock.txt`). Training needs an
NVIDIA GPU with a driver for CUDA 12.8 or newer. The paper scripts run on a CPU.

## Figures and tables of the paper

```bash
python paper/make_paper.py
```

This writes `paper/out/figures`, `paper/out/tables` and `paper/out/numbers.json` in under a minute. Every
bootstrap uses a fixed seed, so the output matches the paper exactly.

| Paper | File |
|---|---|
| Fig. 2 | `figures/Fig2.pdf` |
| Fig. 3 | `figures/Fig3.pdf` |
| Fig. 4 | `figures/Fig4.pdf` |
| Fig. A1 | `figures/FigA1.pdf` |
| Table 2 | `tables/tab_main.tex` |
| Table 3 | `tables/tab_paired.tex` |
| Table A1 | `tables/tab_app_contrasts.tex` |
| Table A2 | `tables/tab_app_windows.tex` |
| Table A3 | `tables/tab_ablation.tex` |
| Table C1 | `tables/tab_app_effect.tex` |

Fig. 1 is a schematic. Fig. C1 and Tables B1, C2 and C3 come from the analysis of IER, whose code is described
below.

## Training the runs again

The study adds two sets of runs to the analysis of IER. Both use SAC and 1M steps per run.

| Item | Runs | What it does |
|---|---|---|
| P1 | 80 | ablation of four rules that limit the replays, seeds 0 to 9 of HalfCheetah and walker-walk |
| F1 | 240 | final test of SAC, IER and the chosen rule, new seeds 20 to 39 of four tasks |

```bash
python scripts/run_main.py --list              # every item with its run count
python scripts/run_main.py --item P1           # ablation
python scripts/pack_records.py                 # packs results/raw into data/
python scripts/select_rule.py --write          # applies the selection rule and writes configs/followup.yaml
python scripts/run_main.py --item F1           # final test of the chosen rule
python scripts/pack_records.py
python paper/make_paper.py
```

Runs are written to `results/raw/<item>/`. Finished runs are skipped and stopped runs resume from their
checkpoint. `--tasks`, `--seeds` and `--arms` split an item across machines, and `--out` trains into a fresh
folder. One group of 20 seeds trains in one process with stacked networks and takes about 5 hours for 1M steps on
an RTX 3090. Note that `pack_records.py` replaces `data/ablation` and `data/final_test` with the runs in
`results/raw`.

The selection rule was fixed before any P1 run. It keeps a rule whose mean paired AUC difference against IER is at
least -0.02 on both tasks and above zero over the two tasks, and it picks the kept rule with the largest mean.
IER-PREFIX and IER-CAP pass, and IER-CAP has the larger mean.

## Data

| File | One row per |
|---|---|
| `data/final_test/eval.csv.gz` | test episode of F1, 10 episodes every 10k steps |
| `data/final_test/train.csv.gz` | training episode of F1, with its length, return, kind and replayed steps |
| `data/ablation/eval.csv.gz` | test episode of P1 |
| `data/ablation/train.csv.gz` | training episode of P1 |
| `data/analysis/curves.csv.gz` | test of a run of the analysis of IER, as the mean return |
| `data/analysis/runs.csv.gz` | run of the analysis of IER, with its final return and replay statistics |
| `data/analysis/repeats.csv.gz` | replay in the analysis of IER, with its return and the stored return |
| `data/analysis/contrasts.csv.gz`, `gains.csv.gz` | paired difference between two arms of the analysis |
| `data/analysis/a1_ier_train.csv.gz` | training episode of IER on HalfCheetah and walker-walk, seeds 0 to 19 |
| `data/analysis/normalizers.json` | task and budget, with the final return of SAC used to normalize the AUC |

Every run is named by its task, condition, learner, arm and seed. The arms are BASE (SAC alone), IER, IER_CAP and
the other rules of the ablation, IER_STOP, IER_MARGIN and IER_PREFIX. The kind of a training episode is explore
(the random phase), policy or replay.

## The analysis of IER

Section 4 and Appendix C of the paper rest on an earlier analysis of IER with 1280 runs. Its code, the record of
every run, the replay check and the random-episode test are in the repository
[ier-stress](https://github.com/alamgirakash2000/ier-stress). This repository carries the summary tables of that
analysis that the paper uses, and `scripts/run_main.py` can still train its runs (items A1 to E1).

## Third-party code

`third_party/ier` is the released code of IER by its authors, unmodified, under the MIT license in
`third_party/ier/LICENSE`. The fonts in `paper_assets/fonts` are Latin Modern under the GUST Font License.
