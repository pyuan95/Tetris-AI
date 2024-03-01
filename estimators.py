import numpy as np
from typing import Tuple, List
from nbTetris import Tetris
from policy import Estimator
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


class CNN(nn.Module):
    def __init__(self, inp_shape, ker_size, l1_filters, hidden_size, pool_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, l1_filters, ker_size, padding="same")
        self.pool = nn.MaxPool2d(pool_size, pool_size)

        # getting the shape...
        with torch.no_grad():
            x = torch.zeros((1,) + inp_shape)
            x = self.conv1(x)
            x = torch.stack(torch.mean(x, dim=1), torch.max(x, dim=1))
            shp = x.flatten().shape[0]
            shp = int(shp)

        self.policy_hidden = nn.Linear(shp, hidden_size)
        self.policy = nn.Linear(hidden_size, Tetris.num_actions)
        self.value_hidden = nn.Linear(shp, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x, batched: bool = False):
        if batched:
            x = torch.unsqueeze(x, 1)
        else:
            x = torch.unsqueeze(x, 0)

        x = F.relu(self.conv1(x))

        if batched:
            x = torch.flatten(x, 1)  # flatten everything except batch
        else:
            x = torch.flatten(x)

        policy = F.relu(self.policy_hidden(x.detach()))
        policy = F.softmax(self.policy(policy), dim=-1)
        value = self.value(F.relu(self.value_hidden(x)))
        return value, policy


class DNN(nn.Module):
    def __init__(
        self, inp_shape, value_hidden_sizes: List[int], policy_hidden_sizes: List[int]
    ):
        super().__init__()
        value_hidden_sizes = [int(np.prod(inp_shape))] + value_hidden_sizes
        policy_hidden_sizes = [int(np.prod(inp_shape))] + policy_hidden_sizes
        self.value_layers = nn.ModuleList(
            [
                nn.Linear(value_hidden_sizes[i], value_hidden_sizes[i + 1])
                for i in range(len(value_hidden_sizes) - 1)
            ]
        )
        self.policy_layers = nn.ModuleList(
            [
                nn.Linear(policy_hidden_sizes[i], policy_hidden_sizes[i + 1])
                for i in range(len(policy_hidden_sizes) - 1)
            ]
        )
        self.value = nn.Linear(value_hidden_sizes[-1], 1)
        self.policy = nn.Linear(policy_hidden_sizes[-1], Tetris.num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, x, batched: bool = False):
        if batched:
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x)
        p = x
        v = x
        for l in self.value_layers:
            v = self.relu(l(v))
        for l in self.policy_layers:
            p = self.relu(l(p))
        value = self.value(v)
        policy = self.softmax(self.policy(p))
        return value, policy


class EstimatorImpl(Estimator):
    """
    Helper class: can be made into CNN/DNN estimator
    """

    def __init__(self, network, lr, device):
        self.net = network
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.example_cnt = 0
        self.losses = []

    def __call__(self, state: Tetris) -> Tuple[float, np.ndarray]:
        tmp = torch.tensor(state.getState(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            value, policy = self.net(tmp, batched=False)
            value = float(value)
            policy = policy.cpu().numpy()
            return value, policy

    def batch_call(self, states: List[Tetris]) -> Tuple[np.ndarray, np.ndarray]:
        states = np.array([s.getState() for s in states])
        tmp = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            value, policy = self.net(tmp, batched=True)
            return value.cpu().numpy().reshape(-1), policy.cpu().numpy()

    def accumulate_gradient(
        self,
        states: List[Tetris],
        actions: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ):
        states = np.array([s.getState() for s in states])
        tmp = torch.tensor(states, dtype=torch.float32).to(self.device)
        pred_values, pred_policies = self.net.forward(tmp, batched=True)
        policies = (
            torch.tensor(policies, dtype=torch.float32)
            .reshape(pred_policies.shape)
            .to(self.device)
        )
        values = (
            torch.tensor(values, dtype=torch.float32)
            .reshape(pred_values.shape)
            .to(self.device)
        )
        self.losses.append(
            F.mse_loss(pred_policies, policies) + F.mse_loss(pred_values, values)
        )
        self.example_cnt += len(states)
        if self.device == "cuda":
            del values
            del policies
            del pred_values
            del pred_policies
            del tmp
            del states
            gc.collect()
            torch.cuda.empty_cache()

    def apply_gradient(self):
        self.optimizer.zero_grad()
        loss = torch.mean(torch.stack(self.losses))
        loss.backward()
        self.optimizer.step()
        self.losses = []
        return float(loss)

    def save(self, file_path):
        torch.save(
            {
                "net": self.net.state_dict(),
                "opt": self.optimizer.state_dict(),
                "example_cnt": self.example_cnt,
            },
            file_path,
        )

    def load(self, file_path):
        data = torch.load(file_path)
        self.net.load_state_dict(data["net"])
        self.net.eval()
        self.optimizer.load_state_dict(data["opt"])
        self.example_cnt = data["example_cnt"]


def get_cnn_estimator(
    inp_shape, ker_size, l1_filters, hidden_size, pool_size, lr, device
):
    net = CNN(inp_shape, ker_size, l1_filters, hidden_size, pool_size).to(device)
    return EstimatorImpl(torch.jit.script(net), lr, device)


def get_dnn_estimator(inp_shape, value_hidden_sizes, policy_hidden_sizes, lr, device):
    net = DNN(inp_shape, value_hidden_sizes, policy_hidden_sizes).to(device)
    return EstimatorImpl(torch.jit.script(net), lr, device)
