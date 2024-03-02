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

# game settings
board_size = (22, 10)
actions_per_drop = 2

# MCTS Policy settings
cpuct = 0.5
num_sims = 400
time_allowed = float("inf")
td_alpha = 0.9
gamma = 0.995
value_leaves_only = False
n_replays = 4
n_epochs = 10

use_cnn = False

if use_cnn:
    # CNN estimator settings
    inp_shape = board_size
    ker_size = 3
    l1_filters = 4
    hidden_size = 64
    pool_size = 2
    lr = 0.001
else:
    # DNN estimator settings
    lr = 0.0001
    inp_shape = board_size
    initial_hidden = 2048
    value_hidden_sizes = [64, 64]
    policy_hidden_sizes = [64, 64]

# train settings
device = "cpu"
save_dir = "./saved"
iters = 5000
rollouts_per_iter = 100
num_processes = 8
truncate_length = 20000
batch_size = 32
save_freq = 1
temp_fn_params = (1, 10, 0.25)
eps_fn_params = (0.20, 20, 0.14)
temp_fn = make_exp_decreasing_fn(*temp_fn_params)
eps_fn = make_exp_decreasing_fn(*eps_fn_params)
warm_start = False
warm_start_iters = 20

params = {
    "board_size": board_size,
    "actions_per_drop": actions_per_drop,
    "cpuct": cpuct,
    "num_sims": num_sims,
    "time_allowed": time_allowed,
    "td_alpha": td_alpha,
    "gamma": gamma,
    "value_leaves_only": value_leaves_only,
    "n_replays": n_replays,
    "use_cnn": use_cnn,
    "temp_fn_params": temp_fn_params,
    "eps_fn_params": eps_fn_params,
    "warm_start": warm_start,
    "warm_start_iters": warm_start_iters,
}

if use_cnn:
    params["inp_shape"] = inp_shape
    params["ker_size"] = ker_size
    params["l1_filters"] = l1_filters
    params["hidden_size"] = hidden_size
    params["pool_size"] = pool_size
    params["lr"] = lr
else:
    params["lr"] = lr
    params["inp_shape"] = inp_shape
    params["initial_hidden"] = initial_hidden
    params["value_hidden_sizes"] = value_hidden_sizes
    params["policy_hidden_sizes"] = policy_hidden_sizes


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
    if warm_start:
        log("warm start: for {0} iters".format(warm_start_iters))
        temp_fn_warm_start = make_exp_decreasing_fn(100, 100, 50)
        eps_fn_warm_start = make_exp_decreasing_fn(1, 100, 1)
        train_policy(
            board_size,
            actions_per_drop,
            policy_factory,
            save_dir,
            warm_start_iters,
            200,
            12,
            temp_fn_warm_start,
            eps_fn_warm_start,
            truncate_length,
            batch_size,
            save_freq,
            False,
            log,
        )
        shutil.copyfile(
            os.path.join(save_dir, "policy"),
            os.path.join(save_dir, "policy_warmstarted"),
        )
        log("=" * 20 + "\n" + "finished warm start!\n" + "=" * 20 + "\n\n\n")

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
    )
