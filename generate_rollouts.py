from policy import Policy, Rollout
from nbTetris import Tetris
import numpy as np
import pickle
from typing import Optional
from torch.multiprocessing import Pool
import random
import os
import pickle
import sys
from parameters import *
from time import sleep


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


def count_rollouts():
    return len(
        [
            os.path.join(save_dir, f)
            for f in os.listdir(save_dir)
            if f.startswith(rollout_prefix)
        ]
    )


if __name__ == "__main__":
    while 1:
        if count_rollouts() > 200:
            # if we are making more rollouts than the trainer can consume, chill for a bit
            sleep(10)
        with open(sys.argv[1], "rb") as args_f:
            # unpack args
            args = pickle.load(args_f)
            policy_save_path = args["policy_save_path"]
            policy_factory = args["policy_factory"]
            name = args["name"]

            # generate the rollout
            policy = policy_factory()
            while 1:
                # saved policy might be edited while we try to read
                # this is a hack obviously lol
                try:
                    policy.load(policy_save_path)
                    break
                except:
                    pass
            fname = name + "_" + str(random.randint(1e20, 1e21))
            t = Tetris(
                boardsize=args["board_size"], actions_per_drop=args["actions_per_drop"]
            )
            generate_rollout(
                policy,
                os.path.join(args["directory"], fname),
                t,
                args["temp"],
                args["eps"],
                args["truncate_length"],
            )
