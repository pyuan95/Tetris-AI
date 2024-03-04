import numpy as np
from typing import *
from nbTetris import Tetris, T, fillT, equalT
from time import time
from numba import jit
from numba import types
from numba.typed import Dict, List
from policy import Policy, Estimator, Rollout

INIT_VALUE = 1e9


@jit(nopython=True)
def get_best_action(
    cpuct: float,
    priors: np.ndarray,
    visits: np.ndarray,
    qvalues: np.ndarray,
) -> int:
    ucb = qvalues * (1 + cpuct * priors / np.sqrt(1 + visits))
    return np.argmax(ucb)


@jit(nopython=True)
def update(value: float, action: int, visits: np.ndarray, qvalues: np.ndarray):
    n = visits[action]
    visits[action] += 1
    qvalues[action] = n / (n + 1) * qvalues[action] + value / (n + 1)


@jit(nopython=True)
def MCTSselect(t: T, s: T, cpuct, tree):
    fillT(s, t)
    h = s.hash()
    path = []
    prev_score = s.getScore()
    while h in tree and not s.end:
        priors, visits, q_values = tree[h]
        action = get_best_action(cpuct, priors, visits, q_values)
        s.play(action)
        h = s.hash()

        current_score = s.getScore()
        path.append((action, visits, q_values, current_score - prev_score))
        prev_score = current_score
    return h, path, s.end


class MCTSPolicy(Policy):
    def __init__(
        self,
        network: Estimator,
        cpuct: float,
        num_sims=100,
        time_allowed=float("inf"),
        td_alpha=0.01,
        gamma=0.999,
        value_leaves_only=True,
        n_replays=3,
        n_epochs=1,
    ):
        self.net = network  # should take state -> value, policy
        self.num_sims = num_sims
        self.time_allowed = time_allowed
        self.cpuct = cpuct
        self.memory = np.zeros(
            [num_sims * 2 + 10, Tetris.num_actions], dtype=np.float32
        )
        self.memory_idx = 0
        self.td_alpha = td_alpha
        self.gamma = gamma
        self.value_leaves_only = value_leaves_only
        self.n_replays = n_replays
        self.n_epochs = n_epochs

    def __call__(self, state: Tetris, temp=1.0, eps=0.0, return_tree=False):
        self.__reset_memory()

        if temp >= 10 or eps >= 0.90:
            return np.ones(Tetris.num_actions) / Tetris.num_actions

        # state hash to Tuple[priors: np.array, visits: np.array, q-values: np.array]
        tree = tree = Dict.empty(
            key_type=types.int64,
            value_type=types.Tuple(
                (types.float32[:], types.float32[:], types.float32[:])
            ),
        )
        state.tetris.tot_actions = 0
        s = state.clone()
        for _ in range(self.num_sims):
            h, path, isterminal = MCTSselect(state.tetris, s.tetris, self.cpuct, tree)
            if isterminal:
                value = 0
            else:
                value, policy = self.net(s)
                # set initial Q to extremely high number; will get overwritten with first
                # real estimate immediately.
                tree[h] = (
                    policy,
                    self.__allocate_memory(0),
                    self.__allocate_memory(INIT_VALUE),
                )
            # backprop
            for action, vs, qs, rew in reversed(path):
                if not self.value_leaves_only:
                    value = rew + self.gamma * value
                update(value, action, vs, qs)
        # search is done, return action distribution
        _, visits, _ = tree[hash(state)]
        if return_tree:
            return tree
        else:
            # colder temp -> best action dominates
            # select random action with probability epsilon
            visits = (visits / np.max(visits)) ** (1 / temp)
            return (1 - eps) * (visits / np.sum(visits)) + eps / Tetris.num_actions

    def train(self, rollouts: List[Rollout], batch_size=1024, **kwargs) -> None:
        losses_over_replays = []
        value_over_replays = []
        stddev_value_over_replays = []
        for _ in range(self.n_replays):
            states = []
            values = []
            actions = []
            policies = []

            for r in rollouts:
                # using td; discard last state if truncated
                vs, _ = self.net.batch_call(r.states)
                vs, target = (
                    (vs[:-1], vs[1:]) if r.truncated else (vs, np.append(vs[1:], 0.0))
                )

                vs = (1 - self.td_alpha) * vs + self.td_alpha * (
                    r.rewards + self.gamma * target
                )
                values.extend(vs)
                actions.extend(r.actions[:-1] if r.truncated else r.actions)
                policies.extend(r.policies[:-1] if r.truncated else r.policies)
                states.extend(r.states[:-1] if r.truncated else r.states)
            values = np.array(values)
            actions = np.array(actions)
            policies = np.array(policies)
            losses = []
            for _ in range(self.n_epochs):
                for i in range(0, len(states), batch_size):
                    end = min(i + batch_size, len(states))
                    self.net.accumulate_gradient(
                        states[i:end],
                        actions[i:end],
                        policies[i:end],
                        values[i:end],
                    )
                    losses.append(self.net.apply_gradient())
            losses_over_replays.append(np.mean(losses))
            value_over_replays.append(np.mean(values))
            stddev_value_over_replays.append(np.std(values))

        # if "log" in kwargs:
        #     print(losses)
        #     kwargs["log"]("MSE over replays: {0}".format(losses_over_replays))
        #     kwargs["log"]("avg value over replays: {0}".format(value_over_replays))
        #     kwargs["log"]("stddev value: {0}".format(stddev_value_over_replays))

    def save(self, file_path: str) -> None:
        self.net.save(file_path)

    def load(self, file_path: str) -> None:
        self.net.load(file_path)

    def __reset_memory(self):
        """
        resets internal memory
        num_nodes: number of nodes to reserve memory for
        """
        self.memory_idx = 0

    def __allocate_memory(self, init) -> np.ndarray:
        r = self.memory[self.memory_idx]
        r[:] = init
        self.memory_idx += 1
        return r
