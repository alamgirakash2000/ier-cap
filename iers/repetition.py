"""The repetition controller for the arms BASE, IER, IER3 and DUP.

Two schedules are available, because the paper and the released code differ.

`algorithm1` follows Algorithm 1 of the paper. This is the default for the IER arms:
    R_max starts at 0. When an episode ends with a return above R_max, the actions are
    stored and the counter is set to RN. Otherwise a positive counter drops by one. While
    the counter is positive, episodes replay the stored actions. A replay that beats
    R_max therefore re-triggers.

`released_code` follows train_loops/ier/episode_reward_loop.py in the released repository:
    A trigger sets the counter to RN but does not start repeating, so the next episode is a
    policy episode. RN - 1 replays follow. No episode can trigger until the counter is back
    at 0, which includes replays and the policy episode after the trigger.

Both schedules hand over to the policy when a replayed episode outlasts the stored actions
(the paper does not say what happens here, this is what the released code does). Exploration
actions take priority over replay in both.

IER-CAP and the three other rules of the ablation, `algorithm1` only. Each rule is switched on by one argument
and off by default, so every other arm runs exactly as before:
    cap         a replay starts only while replayed steps so far are below cap times all steps so far.
                A replay that may not start is a policy episode and still uses up one of the RN episodes.
    stop_below  after a replay that keeps less than this share of the stored return and sets no new best,
                the remaining replays of that stored episode are cancelled.
    margin      a new best starts replays only if it beats the mean return of the last `margin_window`
                policy episodes by margin times the absolute value of that mean. Otherwise R_max still
                rises, but the stored actions and the counter are left as they are. With fewer than three
                policy episodes so far, every new best starts replays, as in IER.
    prefix      a replay runs only the first `prefix` stored actions, then the policy acts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

EXPLORE, REPLAY, POLICY = "explore", "replay", "policy"


class Mode(IntEnum):
    NONE = 0
    REPEAT = 1
    DUPLICATE = 2


@dataclass
class EndResult:
    triggered: bool = False
    dup_copies: int = 0  # how many copies of the stored episode the runner must insert


class RepetitionController:
    def __init__(self, mode: Mode, rn: int, exploration_steps: int, schedule: str = "algorithm1",
                 pick: str = "best", pick_seed: int | None = None, cap: float | None = None,
                 stop_below: float | None = None, margin: float | None = None, margin_window: int = 10,
                 prefix: int | None = None):
        if schedule not in ("algorithm1", "released_code"):
            raise ValueError(f"Unknown schedule: {schedule}")
        if pick not in ("best", "random") or (pick == "random" and (schedule != "algorithm1" or pick_seed is None)):
            raise ValueError(f"pick={pick!r} needs the algorithm1 schedule and a pick_seed")
        self.mode = mode
        self.rn = rn
        self.exploration_steps = exploration_steps
        self.schedule = schedule
        self.r_max = 0.0
        self.counter = 0
        self.repeat_episode = False  # used only by the released_code schedule
        self.stored_actions: list | None = None
        # E1, IER3_RANDOM (exploratory): a trigger replays a uniformly drawn past episode, the triggering one
        # included, instead of the new best. The draw has its own stream, a seventh child of the run seed, so the six
        # streams of iers/runner.py are untouched.
        self.pick = pick
        self.past: list[tuple[np.ndarray, float]] = []
        self.stored_index = -1
        self.stored_return = 0.0
        self.pick_rng = (np.random.default_rng(np.random.SeedSequence(pick_seed).spawn(7)[6])
                         if pick == "random" else None)
        # Rules of the ablation. None switches a rule off.
        self.cap, self.stop_below, self.margin, self.prefix = cap, stop_below, margin, prefix
        self.margin_window = margin_window
        self.variant = any(v is not None for v in (cap, stop_below, margin, prefix))
        if self.variant and (schedule != "algorithm1" or mode != Mode.REPEAT or pick != "best"):
            raise ValueError("The ablation rules need the algorithm1 schedule, repetition and the best episode.")
        self.replay_steps = 0  # replayed steps so far, for cap
        self.cap_open = True  # whether this episode may replay, decided at its first step, for cap
        self.ep_replayed = False  # whether this episode replayed any step
        self.ep_explored = False  # whether this episode took any exploration step
        self.policy_returns: list[float] = []  # returns of the latest policy episodes, for margin

    def source(self, t: int, ep_t: int):
        """Return (source, stored action or None) for global step t and 1-based episode step ep_t."""
        if self.variant:
            return self._source_variant(t, ep_t)
        if t < self.exploration_steps:
            return EXPLORE, None
        if self.mode != Mode.REPEAT:
            return POLICY, None
        if self.schedule == "algorithm1":
            if self.counter > 0 and ep_t <= len(self.stored_actions):
                return REPLAY, self.stored_actions[ep_t - 1]
            return POLICY, None
        if self.counter > 0 and self.repeat_episode:
            action = self.stored_actions[ep_t - 1]
            if ep_t >= len(self.stored_actions):
                self.repeat_episode = False
            return REPLAY, action
        return POLICY, None

    def _source_variant(self, t: int, ep_t: int):
        """source() for the ablation rules: the algorithm1 rule, plus cap and prefix, plus episode flags."""
        if ep_t == 1:
            self.ep_replayed = self.ep_explored = False
            self.cap_open = self.cap is None or t == 0 or self.replay_steps < self.cap * t
        if t < self.exploration_steps:
            self.ep_explored = True
            return EXPLORE, None
        limit = len(self.stored_actions) if self.stored_actions is not None else 0
        if self.prefix is not None:
            limit = min(limit, self.prefix)
        if self.counter > 0 and self.cap_open and ep_t <= limit:
            self.ep_replayed = True
            self.replay_steps += 1
            return REPLAY, self.stored_actions[ep_t - 1]
        return POLICY, None

    def _end_variant(self, episode_return: float, actions: list, steps_done: int) -> EndResult:
        """on_episode_end() for the ablation rules. Without margin and stop_below it equals algorithm1."""
        is_policy = not self.ep_replayed and not self.ep_explored
        if episode_return > self.r_max:
            passes = True
            if self.margin is not None:
                recent = self.policy_returns[-self.margin_window:]
                if len(recent) >= 3:
                    mean = float(np.mean(recent))
                    passes = episode_return - mean >= self.margin * abs(mean)
            self.r_max = episode_return
            if passes:
                self.stored_actions = actions
                self.stored_index, self.stored_return = -1, float(episode_return)
                self.counter = self.rn
                result = EndResult(triggered=True)
            else:
                if self.counter > 0:
                    self.counter -= 1
                result = EndResult()
        else:
            if self.counter > 0:
                failed = (self.stop_below is not None and self.ep_replayed and self.stored_return > 0
                          and episode_return < self.stop_below * self.stored_return)
                self.counter = 0 if failed else self.counter - 1
            result = EndResult()
        if is_policy:
            self.policy_returns.append(float(episode_return))
            del self.policy_returns[:-max(self.margin_window, 1)]
        return result

    def on_episode_end(self, episode_return: float, actions: list, steps_done: int) -> EndResult:
        """Update after an episode. steps_done is the number of environment steps taken so far."""
        if self.mode == Mode.NONE:
            return EndResult()
        if self.variant:
            return self._end_variant(episode_return, actions, steps_done)
        if self.schedule == "released_code":
            return self._end_released(episode_return, actions, steps_done)
        if self.pick == "random":
            # One array per episode keeps checkpoints small. Stacking keeps every value exactly.
            self.past.append((np.stack(actions) if len(actions) else np.empty((0, 0)), float(episode_return)))
        if episode_return > self.r_max:
            self.r_max = episode_return
            self.stored_actions = actions
            self.stored_index, self.stored_return = -1, float(episode_return)
            if self.pick == "random":
                self.stored_index = int(self.pick_rng.integers(len(self.past)))
                drawn, self.stored_return = self.past[self.stored_index]
                self.stored_actions = list(drawn)
            if self.mode == Mode.REPEAT:
                self.counter = self.rn
                return EndResult(triggered=True)
            # DUP inserts when IER would start repeating. During exploration IER would not
            # repeat, because exploration actions take priority.
            copies = self.rn if steps_done >= self.exploration_steps else 0
            return EndResult(triggered=True, dup_copies=copies)
        if self.counter > 0:
            self.counter -= 1
        return EndResult()

    def _end_released(self, episode_return: float, actions: list, steps_done: int) -> EndResult:
        if self.counter == 1:
            self.repeat_episode = False
            self.counter -= 1
        elif self.counter > 1:
            self.counter -= 1
            self.repeat_episode = True
        elif episode_return > 0 and episode_return > self.r_max:
            self.r_max = episode_return
            self.stored_actions = actions
            if self.mode == Mode.REPEAT:
                self.counter = self.rn
                return EndResult(triggered=True)
            copies = self.rn if steps_done >= self.exploration_steps else 0
            return EndResult(triggered=True, dup_copies=copies)
        return EndResult()

    def state_dict(self) -> dict:
        state = {
            "r_max": self.r_max,
            "counter": self.counter,
            "repeat_episode": self.repeat_episode,
            "stored_actions": self.stored_actions,
        }
        if self.pick == "random":
            state.update(past=self.past, stored_index=self.stored_index, stored_return=self.stored_return,
                         pick_rng=self.pick_rng.bit_generator.state)
        if self.variant:
            state.update(stored_return=self.stored_return, replay_steps=self.replay_steps, cap_open=self.cap_open,
                         ep_replayed=self.ep_replayed, ep_explored=self.ep_explored,
                         policy_returns=list(self.policy_returns))
        return state

    def load_state_dict(self, state: dict) -> None:
        self.r_max = state["r_max"]
        self.counter = state["counter"]
        self.repeat_episode = state["repeat_episode"]
        self.stored_actions = state["stored_actions"]
        if self.pick == "random":
            self.past, self.stored_index = state["past"], state["stored_index"]
            self.stored_return = state["stored_return"]
            self.pick_rng.bit_generator.state = state["pick_rng"]
        if self.variant:
            self.stored_return, self.replay_steps = state["stored_return"], state["replay_steps"]
            self.cap_open, self.ep_replayed, self.ep_explored = state["cap_open"], state["ep_replayed"], state["ep_explored"]
            self.policy_returns = list(state["policy_returns"])


ARM_TO_MODE = {
    "BASE": Mode.NONE,
    "IER": Mode.REPEAT,
    "IER3": Mode.REPEAT,
    "IER_REL": Mode.REPEAT,
    "IER3_REL": Mode.REPEAT,
    "DUP": Mode.DUPLICATE,
    "IER3_RANDOM": Mode.REPEAT,  # E1, exploratory
    # IER-CAP and the other rules of the ablation (see the docstring above), with the per-task RN.
    "IER_CAP": Mode.REPEAT,
    "IER_STOP": Mode.REPEAT,
    "IER_MARGIN": Mode.REPEAT,
    "IER_PREFIX": Mode.REPEAT,
}

# One setting per rule, fixed before any of their runs. IER-CAP uses a budget of 0.25.
ARM_VARIANT = {
    "IER_CAP": {"cap": 0.25},
    "IER_STOP": {"stop_below": 0.5},
    "IER_MARGIN": {"margin": 0.1, "margin_window": 10},
    "IER_PREFIX": {"prefix": 200},
}


def arm_variant(arm: str) -> dict:
    """Keyword arguments of RepetitionController for an ablation rule, empty for every other arm."""
    return dict(ARM_VARIANT.get(arm, {}))


def arm_pick(arm: str) -> str:
    """Which stored episode a trigger replays: the new best, or for IER3_RANDOM a uniformly drawn past one."""
    return "random" if arm == "IER3_RANDOM" else "best"


def arm_schedule(arm: str) -> str:
    """The sensitivity arms use the released-code schedule. Every other arm uses Algorithm 1."""
    return "released_code" if arm in ("IER_REL", "IER3_REL") else "algorithm1"


def arm_rn(arm: str, task_rn: int) -> int:
    """RN for an arm: the per-task paper value, fixed at 3 for IER3 and IER3_REL, and 0 for BASE.

    DUP takes the RN of the arm it is compared with. That is the per-task RN by default. Comparisons
    with IER3 pass rn=3 to make_spec, so that the only difference between the two arms is the arm's own
    change.
    """
    return {"BASE": 0, "IER": task_rn, "IER3": 3, "IER_REL": task_rn, "IER3_REL": 3, "DUP": task_rn,
            "IER3_RANDOM": 3, "IER_CAP": task_rn, "IER_STOP": task_rn, "IER_MARGIN": task_rn,
            "IER_PREFIX": task_rn}[arm]
