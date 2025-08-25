import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
from abc import ABC, abstractmethod
from collections import deque

# --- Section 5.1: Interfaces ---

class CostModel(ABC):
    """Abstract base class for a cost model."""
    @abstractmethod
    def calculate_cost(self, execution_report: dict) -> float:
        pass

class ExecutionEngine(ABC):
    """Abstract base class for an execution engine."""
    @abstractmethod
    def execute(self, action: int, context: np.ndarray) -> dict:
        pass

# --- Section 5.2: MarketEnvironment ---

class MarketEnvironment(ExecutionEngine):
    """
    Simulates a non-stationary market environment for order routing.
    """
    def __init__(self, n_actions: int, n_features: int, drift_mode: str = 'abrupt',
                 change_prob: float = 0.001, drift_std: float = 0.1):
        self.n_actions = n_actions
        self.n_features = n_features
        self.drift_mode = drift_mode
        self.change_prob = change_prob
        self.drift_std = drift_std
        
        self.true_coeffs = np.random.randn(n_actions, n_features)
        self.noise_std = 0.1

    def _update_true_costs(self):
        """Updates the true cost model to simulate non-stationarity."""
        if self.drift_mode == 'abrupt':
            if np.random.rand() < self.change_prob:
                best_action = np.random.randint(self.n_actions)
                self.true_coeffs = np.random.randn(self.n_actions, self.n_features) * 0.5
                self.true_coeffs[best_action] *= -2 
        elif self.drift_mode == 'gradual':
            self.true_coeffs += np.random.randn(self.n_actions, self.n_features) * self.drift_std

    def get_context(self) -> np.ndarray:
        """Generates a new context vector."""
        return np.random.randn(self.n_features)

    def execute(self, action: int, context: np.ndarray) -> dict:
        """Executes an action and returns an execution report (cost)."""
        self._update_true_costs()
        true_costs = self.true_coeffs @ context
        observed_cost = true_costs[action] + np.random.randn() * self.noise_std
        
        return {
            'action': action,
            'cost': observed_cost,
            'all_true_costs': true_costs 
        }

# --- Section 5.3: MirrorDescentRouter (Exponentiated Gradient) ---

class MirrorDescentRouter:
    """Implements an Exponentiated Gradient router."""
    def __init__(self, n_actions: int, c: float = 1.0):
        self.n_actions = n_actions
        self.c = c
        self.weights = np.ones(n_actions)
        self.t = 0

    def choose_action_probs(self, context: np.ndarray = None) -> np.ndarray:
        return self.weights / np.sum(self.weights)

    def update(self, chosen_action: int, loss: float, probs: np.ndarray):
        self.t += 1
        eta_t = self.c / np.sqrt(self.t)
        estimated_loss = loss / probs[chosen_action]
        self.weights[chosen_action] *= np.exp(-eta_t * estimated_loss)
        
        if np.sum(self.weights) < 1e-9:
            self.weights = np.ones(self.n_actions)

# --- Section 5.4: ContextualBanditRouter (Thompson Sampling) ---

class ThompsonSamplingRouter:
    """Implements a Sliding-Window Thompson Sampling router."""
    def __init__(self, n_actions: int, n_features: int, window_size: int = 100, alpha: float = 1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.window_size = window_size
        self.alpha = alpha
        
        self.history = {a: {'X': deque(maxlen=window_size), 'y': deque(maxlen=window_size)} for a in range(n_actions)}
        self.B = {a: np.eye(n_features) * self.alpha for a in range(n_actions)}
        self.mu = {a: np.zeros(n_features) for a in range(n_actions)}
        self.f = {a: np.zeros(n_features) for a in range(n_actions)}

    def choose_action(self, context: np.ndarray) -> int:
        sampled_coeffs = np.array([
            np.random.multivariate_normal(self.mu[a], np.linalg.inv(self.B[a]))
            for a in range(self.n_actions)
        ])
        expected_rewards = sampled_coeffs @ context
        return np.argmax(expected_rewards)

    def update(self, chosen_action: int, context: np.ndarray, cost: float):
        reward = -cost
        self.history[chosen_action]['X'].append(context)
        self.history[chosen_action]['y'].append(reward)

        X_hist = np.array(self.history[chosen_action]['X'])
        y_hist = np.array(self.history[chosen_action]['y'])
        
        if len(X_hist) > 0:
            self.B[chosen_action] = self.alpha * np.eye(self.n_features) + X_hist.T @ X_hist
            self.f[chosen_action] = X_hist.T @ y_hist
            self.mu[chosen_action] = np.linalg.solve(self.B[chosen_action], self.f[chosen_action])
