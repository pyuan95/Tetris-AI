from policy import Policy, Rollout
from nbTetris import *
import numpy as np
import pickle
from typing import Callable, Optional, Tuple
from torch.multiprocessing import Pool
import random
import time
import subprocess
import os
import tqdm

MAX_EXAMPLES_PER_ITER = 200000


def print_rollout(rollout_path):
    r = load_rollout(rollout_path)
    for state in r.states:
        state.printState()
        print()


def make_exp_decreasing_fn(start, period, asymptote):
    def f(it):
        """
        simple exponentially decreasing function
        """
        return max(asymptote, start * (0.5 ** (it / period)))

    return f


def train_policy(
    board_size,
    actions_per_drop,
    policy_factory: Callable[[], Policy],
    save_dir: str,
    iters: int,
    rollouts_per_iter: int,
    num_processes,
    temp_fn,
    eps_fn,
    truncate_length,
    batch_size,
    save_freq,
    save_historical,
    log,
    rollout_prefix,
):
    log("=" * 20 + "Started Training" + "=" * 20)
    rollout_lengths = [1]
    policy_save_path = os.path.join(save_dir, "policy")

    # make initial policy
    if not os.path.exists(policy_save_path):
        policy = policy_factory()
        policy.save(policy_save_path)
        if save_historical:
            policy.save(os.path.join(save_dir, "policy_iter0"))

    # start generating rollouts
    def get_rollout_file():
        files = [
            os.path.join(save_dir, f)
            for f in os.listdir(save_dir)
            if f.startswith(rollout_prefix)
        ]
        return files[0] if files else None

    for iter_num in range(iters):
        params_outf = "./params_out"
        with open(params_outf, "wb") as params_out:
            pickle.dump(
                {
                    "policy_factory": policy_factory,
                    "policy_save_path": policy_save_path,
                    "name": rollout_prefix,
                    "board_size": board_size,
                    "actions_per_drop": actions_per_drop,
                    "directory": save_dir,
                    "temp": temp_fn(iter_num),
                    "eps": eps_fn(iter_num),
                    "truncate_length": truncate_length,
                },
                params_out,
            )
        if iter_num == 0:
            # start the rollout generation processes
            for _ in range(num_processes):
                subprocess.Popen(["python3", "generate_rollouts.py", params_outf])

        rollouts = []
        n_examples = 0
        rewards = []
        rollout_lengths = []
        lines_cleared = []
        with tqdm.tqdm(range(rollouts_per_iter)) as pbar:
            for _ in pbar:
                # stupid wait until we see some rollouts
                while not (rollout_file := get_rollout_file()):
                    time.sleep(1.0)
                # train on rollouts
                policy = policy_factory()
                policy.load(policy_save_path)

                r = load_rollout(rollout_file)
                os.remove(rollout_file)
                rollouts.append(r)
                rewards.append(np.sum(r.rewards))
                rollout_lengths.append(len(r.states))
                lines_cleared.append(r.states[-1].getLines())
                n_examples += len(r.states)
                policy.train([r], batch_size=batch_size, log=log)
                policy.save(policy_save_path)

                pbar.set_postfix(
                    {
                        "atr": np.mean(
                            [r.states[-1].score for r in rollouts]
                        ),
                        "atl": np.mean(lines_cleared),
                    }
                )
        log(
            "iter {0}: trained on {1} rollouts, {2} examples".format(
                iter_num, len(rollouts), sum(rollout_lengths)
            )
        )
        log("average rollout length: {0}".format(np.mean(rollout_lengths)))
        log(
            "average total reward: {0}".format(
                np.mean([r.states[-1].score for r in rollouts])
            )
        )
        log("average total lines cleared {0}".format(np.mean(lines_cleared)))
        log(
            "average reward per timestep: {0}".format(
                np.sum([np.sum([s.score for s in r.states]) for r in rollouts])
                / np.sum(rollout_lengths)
            )
        )
        log("stddev rollout length: {0}".format(np.std(rollout_lengths)))
        log("stddev total reward: {0}".format(np.std(rewards)))
        log("stddev total lines cleared {0}\n".format(np.std(lines_cleared)))
        if iter_num % save_freq == 0 and save_historical:
            policy_historical_save_path = os.path.join(
                save_dir, "policy_iter{0}".format(iter_num + 1)
            )
            policy.save(policy_historical_save_path)


def save_jitclass(jc):
    typ = jc._numba_type_
    fields = typ.struct

    data = {"name": typ.classname, "struct": {k: getattr(jc, k) for k in fields}}
    return data


def load_jitclass(data):
    cls = globals()[data["name"]]
    instance = cls()
    for k in data["struct"]:
        setattr(instance, k, data["struct"][k])
    return instance


def generate_rollouts_helper(
    directory: str,
    name: str,
    policy_factory: Callable[[], Policy],
    policy_save_path: str,
    board_size: Tuple[int, int],
    actions_per_drop: int,
    temp: float,
    eps: float,
    truncate_length: int,
    n_rollouts: int,
):
    """
    Populates [dir] with rollouts until it contains [n_rollouts] rollouts
    """

    def count_rollouts(directory):
        files = os.listdir(directory)
        return len([f for f in files if f.startswith(name)])

    policy = policy_factory()
    policy.load(policy_save_path)
    while count_rollouts(directory) < n_rollouts:
        fname = name + "_" + str(random.randint(1e20, 1e21))
        t = Tetris(boardsize=board_size, actions_per_drop=actions_per_drop)
        generate_rollout(
            policy, os.path.join(directory, fname), t, temp, eps, truncate_length
        )


def generate_rollout(
    policy: Policy,
    save_path: Optional[str],
    t: Tetris,
    temp: float,
    eps: float,
    truncate_length: int,
):
    states = []
    actions = []
    rewards = []
    policies = []
    prev_score = t.getScore()
    it = 0
    while not t.end and it < truncate_length:
        output_policy = policy(t, temp=temp, eps=eps)
        action = np.random.choice(np.arange(Tetris.num_actions), p=output_policy)
        states.append(t.clone())
        actions.append(action)
        policies.append(output_policy)
        t.play(action)
        cur_score = t.getScore()
        rewards.append(cur_score - prev_score)
        prev_score = cur_score
        it += 1
    truncated = not t.end

    rollout = Rollout(
        states, np.array(policies), np.array(actions), np.array(rewards), truncated
    )

    if save_path:
        save_rollout(rollout, save_path)
    else:
        return rollout


def save_rollout(rollout: Rollout, save_path: str):

    def convertT(t: Tetris):
        t = t.clone()
        t.tetris = save_jitclass(t.tetris)
        t.tetris["struct"]["block"] = save_jitclass(t.tetris["struct"]["block"])
        t.tetris["struct"]["board"] = save_jitclass(t.tetris["struct"]["board"])
        return t

    with open(save_path, "wb") as f:
        np.save(f, rollout.actions)
        np.save(f, rollout.rewards)
        np.save(f, rollout.policies)
        pickle.dump([convertT(s) for s in rollout.states], f)
        pickle.dump(rollout.truncated, f)


def load_rollout(save_path: str) -> Rollout:
    def unconvertT(tp):
        tp.tetris["struct"]["block"] = load_jitclass(tp.tetris["struct"]["block"])
        tp.tetris["struct"]["board"] = load_jitclass(tp.tetris["struct"]["board"])
        tp.tetris = load_jitclass(tp.tetris)
        return tp

    with open(save_path, "rb") as f:
        actions = np.load(f, allow_pickle=True)
        rewards = np.load(f, allow_pickle=True)
        policies = np.load(f, allow_pickle=True)
        states = [unconvertT(t) for t in pickle.load(f)]
        trunc = pickle.load(f)
        rollout = Rollout(states, policies, actions, rewards, trunc)
        return rollout
