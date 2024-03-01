import policy
import MCTSPolicy
from nbTetris import Tetris
import numpy as np
from time import time
from numba import jit
import train
import estimators
import torch


def test_mcts_policy_value():
    t = Tetris(actions_per_drop=1)
    tclone = t.clone()
    c = t.clone()
    c.play(4)
    c.play(4)
    c.play(3)
    specialh = hash(c)

    class DummyEstimator(policy.Estimator):
        def __call__(self, state):
            policy = np.ones([Tetris.num_actions]) / Tetris.num_actions
            policy = policy.astype(np.float32)
            if hash(state) == specialh:
                return 500000, policy
            return 10000, policy

    m = MCTSPolicy.MCTSPolicy(DummyEstimator(), 4, 1000)
    tree = m(t, return_tree=True)
    argmax_path = []
    h = hash(t)
    while h in tree:
        _, visits, qs = tree[h]
        action = np.argmax(visits)
        argmax_path.append((action, visits[action], qs[action]))
        t.play(action)
        h = hash(t)
    # print(argmax_path)  # should be 4, 3, 4, perhaps permuted in some order
    # print(tree[hash(tclone)])
    assert sorted([z[0] for z in argmax_path[:3]]) == [3, 4, 4]

    t = tclone.clone()
    p = m(t.clone(), return_tree=False)
    p1 = m(t.clone(), temp=4)
    a = argmax_path[0][0]

    assert np.argmax(p) == a
    assert np.argmax(p1) == a
    assert p[a] > p1[a]


def test_mcts_policy_rewards():
    t = Tetris(actions_per_drop=1)

    class DummyEstimator(policy.Estimator):
        def __call__(self, state):
            policy = np.ones([Tetris.num_actions]) / Tetris.num_actions
            policy = policy.astype(np.float32)
            return 0, policy

    m = MCTSPolicy.MCTSPolicy(DummyEstimator(), 4, 1000)
    tree = m(t, return_tree=True)
    argmax_path = []
    h = hash(t)
    while h in tree:
        _, visits, qs = tree[h]
        action = np.argmax(visits)
        argmax_path.append((action, visits[action], qs[action]))
        t.play(action)
        h = hash(t)
    assert argmax_path[0][0] == 5  # dropping gives the most rewards
    if argmax_path[-1][1] == 0:
        # last node never visited; fix dummy Q value
        tmp = argmax_path[-1]
        argmax_path[-1] = (tmp[0], tmp[1], 0)
    assert sorted(argmax_path, key=lambda x: x[2], reverse=True) == argmax_path


def speedtest():
    t = Tetris(actions_per_drop=1)
    c = t.clone()
    c.play(2)
    c.play(5)
    h = hash(c)

    class DummyEstimator(policy.Estimator):
        def __call__(self, state):
            policy = np.ones([Tetris.num_actions]) / Tetris.num_actions
            policy = policy.astype(np.float32)
            if hash(state) == h:
                return 100000, policy
            return 10000, policy

    sims_per_trial = 300
    num_trials = 300
    m = MCTSPolicy.MCTSPolicy(DummyEstimator(), 2, sims_per_trial)
    _ = m(t, return_tree=False)  # skip warmup costs
    start = time()
    results = []
    for i in range(num_trials):
        b = m(t, return_tree=False)
    end = time()
    print("speed: {0} sims/sec".format(sims_per_trial * num_trials / (end - start)))


def test_rollout():
    def dummy_policy(state, temp=1.0):
        return np.ones(Tetris.num_actions) / Tetris.num_actions

    t = Tetris()
    x = train.generate_rollout(dummy_policy, None, t, lambda x: 1, 100000)
    assert not x.truncated
    assert (
        np.abs(np.mean(x.policies) * Tetris.num_actions - 1) <= 0.00001
    )  # should be close to zero
    assert [
        len(x.states),
        x.policies.shape[0],
        x.actions.shape[0],
        x.rewards.shape[0],
    ] == [len(x.states)] * 4
    assert sum(x.rewards) >= 10  # have to get rewards for landing some blocks...
    assert isinstance(x.states[0], Tetris)


class dummy_policy:
    def __call__(self, state):
        return np.ones(Tetris.num_actions) / Tetris.num_actions

    def load(self):
        return

    def save(self):
        return


def dummy_policy_factory():
    return dummy_policy()


def test_rollouts():
    train.generate_rollouts_multiprocessed(
        "./testing",
        "test",
        dummy_policy_factory,
        "dummy path",
        (22, 10),
        3,
        train.simple_temp,
        100000,
        100,
        5,
    )


def test_save_load():
    r1 = policy.Rollout(
        [Tetris() for _ in range(100)],
        np.random.randn(100),
        np.random.randn(100),
        np.random.randn(100),
        False,
    )
    train.save_rollout(r1, "testing.pkl")
    r2 = train.load_rollout("testing.pkl")
    assert r1.states == r2.states
    assert (r1.actions == r2.actions).all()
    assert (r1.rewards == r2.rewards).all()
    assert (r1.policies == r2.policies).all()
    assert r1.truncated == r2.truncated


def speedtest_net_cnn():
    x = estimators.CNN((22, 10), 3, 16, 16, 2)
    x = torch.jit.script(x)
    x.load_state_dict(x.state_dict())
    n = 10000
    i = np.random.randn(220 * n).reshape(n, 1, 22, 10)
    start = time()
    for j in range(n):
        with torch.no_grad():
            tmp = torch.tensor(i[j], dtype=torch.float32)
            z = x(tmp)
    end = time()
    print("trials per sec:", n / (end - start))


def speedtest_net_dnn():
    x = estimators.DNN((22, 10), 16)
    x = torch.jit.script(x)
    x.load_state_dict(x.state_dict())
    n = 10000
    i = np.random.randn(220 * n).reshape(n, 22, 10)
    start = time()
    for j in range(n):
        with torch.no_grad():
            tmp = torch.tensor(i[j], dtype=torch.float32)
            z = x(tmp)
    end = time()
    print("trials per sec:", n / (end - start))


if __name__ == "__main__":
    # test_rollout()
    # test_rollouts()
    # test_mcts_policy_value()
    # test_mcts_policy_rewards()
    # speedtest()
    # import cProfile
    # cProfile.run('speedtest()')
    # test_save_load()
    speedtest_net_dnn()
    print("tests passed!")
