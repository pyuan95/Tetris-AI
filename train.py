from policy import Policy, Rollout
from nbTetris import Tetris, Block, Board, T
import numpy as np
import pickle
from typing import Callable, List, Dict, Optional, Tuple
from multiprocessing import Pool
import random
import os
import gc

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
    log,  # log function
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

    pool = Pool(num_processes)
    for iter_num in range(iters):
        # generate rollouts
        rollout_prefix = "rollout_iter{0}_".format(iter_num)
        n_rollouts = min(
            rollouts_per_iter, MAX_EXAMPLES_PER_ITER / (np.mean(rollout_lengths) + 0.1)
        )  # if our rollouts are huge, then let's only try to make around MAX_EXAMPLES_PER_ITER examples.
        n_rollouts = max(n_rollouts, 1)  # make at least one rollout
        generate_rollouts_multiprocessed(
            save_dir,
            rollout_prefix,
            policy_factory,
            policy_save_path,
            board_size,
            actions_per_drop,
            temp_fn(iter_num),
            eps_fn(iter_num),
            truncate_length,
            n_rollouts,
            num_processes,
            pool,
        )

        # train on rollouts
        policy = policy_factory()
        policy.load(policy_save_path)
        rollout_files = [
            os.path.join(save_dir, f)
            for f in os.listdir(save_dir)
            if f.startswith(rollout_prefix)
        ]
        rollouts = []
        n_examples = 0
        rewards = []
        rollout_lengths = []
        lines_cleared = []
        for rollout_file in rollout_files:
            r = load_rollout(rollout_file)
            os.remove(rollout_file)
            rollouts.append(r)
            rewards.append(np.sum(r.rewards))
            rollout_lengths.append(len(r.states))
            lines_cleared.append(r.states[-1].getLines())
            n_examples += len(r.states)
            if n_examples >= MAX_EXAMPLES_PER_ITER:  # so we don't run out of memory
                policy.train(rollouts, batch_size=batch_size, log=log)
                n_examples = 0
                rollouts.clear()
        if len(rollouts) > 0:
            policy.train(rollouts, batch_size=batch_size, log=log)
            rollouts.clear()
        log(
            "iter {0}: trained on {1} rollouts, {2} examples".format(
                iter_num, len(rollout_files), sum(rollout_lengths)
            )
        )
        log("average rollout length: {0}".format(np.mean(rollout_lengths)))
        log(
            "average total reward: {0}".format(
                np.sum([np.sum([s.score for s in r.states]) for r in rollouts])
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
        policy.save(policy_save_path)
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


def generate_rollouts_multiprocessed(
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
    n_processes: int,
    p: Pool,
):
    p.starmap(
        generate_rollouts_helper,
        [
            (
                directory,
                name,
                policy_factory,
                policy_save_path,
                board_size,
                actions_per_drop,
                temp,
                eps,
                truncate_length,
                max(1, n_rollouts - n_processes),
            )
            for _ in range(n_processes)
        ],
    )


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
