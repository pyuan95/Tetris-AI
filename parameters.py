from train import make_exp_decreasing_fn
import MCTSPolicy
import estimators
import os

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
n_replays = 1
n_epochs = 1

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
    lr = 0.0002
    inp_shape = board_size
    initial_hidden = 2048
    value_hidden_sizes = [64, 64]
    policy_hidden_sizes = [64, 64]

# train settings
device = "cpu"
save_dir = "./saved"
iters = 5000
rollouts_per_iter = 50
num_processes = 2
truncate_length = 20000
batch_size = 64
save_freq = 1
temp_fn_params = (1, 10, 0.25)
eps_fn_params = (0.20, 20, 0.14)
temp_fn = make_exp_decreasing_fn(*temp_fn_params)
eps_fn = make_exp_decreasing_fn(*eps_fn_params)
warm_start = False
warm_start_iters = 20
rollout_prefix = "rollout_"

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
