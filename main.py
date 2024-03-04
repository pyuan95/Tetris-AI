from train import train_policy, make_exp_decreasing_fn, generate_rollout, save_rollout
from nbTetris import Tetris
import MCTSPolicy
import estimators
import os
import time
import numpy as np
import json
import pickle
from datetime import datetime
import shutil

from parameters import *


def policy_factory():
    if use_cnn:
        network = estimators.get_cnn_estimator(
            inp_shape, ker_size, l1_filters, hidden_size, pool_size, lr, device
        )
    else:
        network = estimators.get_dnn_estimator(
            inp_shape,
            initial_hidden,
            value_hidden_sizes,
            policy_hidden_sizes,
            lr,
            device,
        )
    return MCTSPolicy.MCTSPolicy(
        network,
        cpuct,
        num_sims,
        time_allowed,
        td_alpha,
        gamma,
        value_leaves_only,
        n_replays,
        n_epochs,
    )


if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    par_save_dir = save_dir
    save_dir = os.path.join(save_dir, str(time.time()))
    os.mkdir(save_dir)

    with open(os.path.join(save_dir, "params.txt"), "w") as file:
        file.write(json.dumps(params))

    with open(os.path.join(save_dir, "policy_factory"), "wb") as file:
        pickle.dump(policy_factory, file)

    if os.path.exists(os.path.join(par_save_dir, "init_policy")):
        shutil.copyfile(
            os.path.join(par_save_dir, "init_policy"),
            os.path.join(save_dir, "policy"),
        )

    def log(msg):
        print(msg)
        with open(os.path.join(save_dir, "log.txt"), "a+") as f:
            timestamp = str(datetime.today()) + ": "
            f.write(timestamp + msg + "\n")

    dummy_policy = policy_factory()
    dummy_tetris = Tetris(board_size, actions_per_drop)
    dummy_policy(dummy_tetris)
    start = time.time()
    n = 3
    for _ in range(n):
        dummy_policy(dummy_tetris)
    end = time.time()
    print("policy speed test: {0} moves/sec".format(n / (end - start)))

    train_policy(
        board_size,
        actions_per_drop,
        policy_factory,
        save_dir,
        iters,
        rollouts_per_iter,
        num_processes,
        temp_fn,
        eps_fn,
        truncate_length,
        batch_size,
        save_freq,
        True,
        log,
        rollout_prefix,
    )
