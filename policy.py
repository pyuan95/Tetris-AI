import numpy as np
from typing import *
from nbTetris import Tetris

# states, actions, rewards, all have the same shape
Rollout = NamedTuple(
    "Rollout",
    [
        ("states", List[Tetris]),
        ("policies", np.ndarray),
        ("actions", np.ndarray),
        ("rewards", np.ndarray),
        ("truncated", bool),
    ],
)


class Estimator:
    def __call__(self, state: Tetris) -> Tuple[float, np.ndarray]:
        """
        returns the estimated value and policy, given a state
        used for flexibility with different models/frameworks
        """
        raise NotImplementedError()

    def batch_call(self, states: List[Tetris]) -> Tuple[np.ndarray, np.ndarray]:
        """
        same as __call__, except batched
        """
        raise NotImplementedError()

    def accumulate_gradient(
        self,
        states: List[Tetris],
        actions: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ):
        """
        accumulates gradients for the estimator on the given states, actions, values
        actions.shape = (N,), actions.dtype = int32
        values.shape = (N,), values.dtype = float32
        """
        raise NotImplementedError()

    def apply_gradient(self) -> float:
        """
        applies the accumulated gradient for the estimator, and returns the loss
        """
        raise NotImplementedError()

    def save(self, file_path: str):
        """
        saves the estimator
        [file_path] is the path to a file, not a dir
        """
        raise NotImplementedError()

    def load(self, file_path: str):
        """
        loads the estimator
        """
        raise NotImplementedError()


class Policy:
    def __call__(self, state: Tetris, temp=1.0, eps=0.0, **kwargs) -> np.ndarray:
        """
        takes in a state; returns a distribution over the actions
        action must be an int in [0, 6]
        """
        raise NotImplementedError()

    def train(self, rollouts: List[Rollout], batch_size=1024) -> None:
        """
        Trains the policy on a batch of rollouts
        """
        raise NotImplementedError()

    def save(self, file_path: str) -> None:
        """
        Saves the model to [file_path]
        """
        raise NotImplementedError()

    def load(self, file_path: str) -> None:
        """
        Loads the model to [file_path]
        """
        raise NotImplementedError()


class DQNPolicy(Policy):
    # TODO
    pass


class CrossEntropyPolicy(Policy):
    # TODO
    pass
