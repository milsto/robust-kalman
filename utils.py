"""
Utilities for robust Kalman implementation and testing.
"""
import numpy as np


class HuberScore:
    """Robust Huber score function."""
    def __init__(self, delta=1.5):
        self._delta = delta

    def evaluate(self, z):
        if abs(z) >= self._delta:
            return self._delta * abs(z) - pow(self._delta, 2) / 2.0
        else:
            return pow(z, 2) / 2.0

    def derivative(self, z):
        raise NotImplemented


class VariablesHistory:
    """Utility to easily track variable history for plotting."""
    def __init__(self):
        self._variables_history = dict()

    def __getitem__(self, item):
        return self._variables_history[item]

    def update(self, variable_name, value):
        if variable_name not in self._variables_history:
            self._variables_history[variable_name] = list()

        self._variables_history[variable_name].append(value)


class WindowStatisticsEstimator:
    """Windowed estimations of first and second moment of a random process."""
    def __init__(self, win_size=25):
        self._win_size = win_size
        self._buffer = np.zeros((self._win_size,), np.float32)
        self._head = 0

    def update(self, value):
        self._buffer[self._head] = value
        self._head = (self._head + 1) % self._win_size

    def mean(self):
        return np.mean(self._buffer)

    def variance(self):
        return np.var(self._buffer)