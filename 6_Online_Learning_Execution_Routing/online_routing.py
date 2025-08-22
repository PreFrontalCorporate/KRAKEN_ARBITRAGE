import matplotlib.pyplot as plt
import unittest
from abc import ABC, abstractmethod
from scipy.stats import invgamma, multivariate_normal
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
    This class acts as both the environment and the execution engine for simplicity.
    """
    def __init__(self, n_actions: int, n_features: int, drift_mode: str = 'abrupt',
                 change_prob: float = 0.001, drift_std: float = 0.1):
        """
        Initializes the market environment.

        Args:
            n_actions (int): Number of possible routing actions (K).
            n_features (int): Dimensionality of the context vector (d).
            drift_mode (str): 'abrupt' for sudden changes, 'gradual' for slow drift.
            change_prob (float): Probability of an abrupt change at any step.
            drift_std (float): Standard deviation for the random walk in gradual drift.
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.drift_mode = drift_mode
        self.change_prob = change_prob
        self.drift_std = drift_std

        # True underlying cost model parameters (hidden from the agent)
        self.true_coeffs = np.random.randn(n_actions, n_features)
        self.noise_std = 0.1
        self.t = 0 # Initialize time step

    def _update_true_costs(self):
        """Updates the true cost model to simulate non-stationarity."""
        if self.drift_mode == 'abrupt':
            if np.random.rand() < self.change_prob:
                # A random action becomes significantly better/worse
                self.true_coeffs = np.random.randn(self.n_actions, self.n_features) * 0.5
                best_action = np.random.randint(self.n_actions)
                self.true_coeffs[best_action] *= -np.random.choice([-2, 2]) # Randomly better or worse

        elif self.drift_mode == 'gradual':
            self.true_coeffs += np.random.randn(self.n_actions, self.n_features) * self.drift_std

    def get_context(self) -> np.ndarray:
        """Generates a new context vector."""
        return np.random.randn(self.n_features)

    def execute(self, action: int, context: np.ndarray) -> dict:
        """
        Executes an action and returns an execution report (cost).
        This method also advances the environment's state.
        """
        self._update_true_costs()
        self.t += 1

        # Calculate true costs for all actions given the context
        true_costs = self.true_coeffs @ context

        # Observed cost for the chosen action has some noise
        observed_cost = true_costs[action] + np.random.randn() * self.noise_std

        execution_report = {
            'action': action,
            'cost': observed_cost,
            'all_true_costs': true_costs # For regret calculation
        }
        return execution_report

# --- Section 5.3: MirrorDescentRouter (Exponentiated Gradient) ---

class MirrorDescentRouter:
    """
    Implements an Exponentiated Gradient router for online decision making.
    This is a special case of Mirror Descent with the negative entropy regularizer.
    """
    def __init__(self, n_actions: int, c: float = 1.0):
        self.n_actions = n_actions
        self.c = c  # Constant for time-varying learning rate
        self.weights = np.ones(n_actions)
        self.t = 0

    def choose_action_probs(self, context: np.ndarray = None) -> np.ndarray:
        """Returns the probability distribution over actions."""
        return self.weights / np.sum(self.weights)

    def update(self, chosen_action: int, loss: float, probs: np.ndarray):
        """Updates weights using the Exponentiated Gradient rule."""
        self.t += 1
        eta_t = self.c / np.sqrt(self.t)

        # Importance-weighted loss estimate (only needed for bandit feedback)
        estimated_loss = loss / probs[chosen_action]

        # Multiplicative update
        self.weights[chosen_action] *= np.exp(-eta_t * estimated_loss)

        # Numerical stability: prevent weights from becoming too small or large
        self.weights = np.maximum(1e-10, self.weights) # Lower bound
        self.weights /= np.sum(self.weights) # Normalize


# --- Section 5.4: ContextualBanditRouter (Sliding-Window Thompson Sampling) ---

class ThompsonSamplingRouter:
    """
    Implements a Sliding-Window Thompson Sampling router with a linear reward model.
    Assumes rewards are negative costs.
    """
    def __init__(self, n_actions: int, n_features: int, window_size: int = 100, alpha: float = 1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.window_size = window_size
        self.alpha = alpha  # Controls prior variance

        # Store recent history for each arm using deques for efficient sliding window
        self.history = {a: {'X': deque(maxlen=window_size), 'y': deque(maxlen=window_size)} for a in range(n_actions)}

        # Posterior parameters for each arm's coefficients beta_a
        # Prior: beta_a ~ N(0, alpha^-1 * I)
        self.B = {a: np.eye(n_features) * self.alpha for a in range(n_actions)}
        self.mu = {a: np.zeros(n_features) for a in range(n_actions)}
        self.f = {a: np.zeros(n_features) for a in range(n_actions)}

    def choose_action(self, context: np.ndarray) -> int:
        """Choose an action by sampling from the posterior and maximizing."""
        sampled_coeffs = np.array([
            np.random.multivariate_normal(self.mu[a], np.linalg.inv(self.B[a]))
            for a in range(self.n_actions)
        ])

        expected_rewards = sampled_coeffs @ context
        return np.argmax(expected_rewards)

    def update(self, chosen_action: int, context: np.ndarray, cost: float):
        """Update the posterior for the chosen action using data in the sliding window."""
        reward = -cost

        # Add new observation to history (deque handles sliding window automatically)
        self.history[chosen_action]['X'].append(context)
        self.history[chosen_action]['y'].append(reward)

        # Re-calculate posterior from scratch using only data in the window
        X_hist = np.array(list(self.history[chosen_action]['X']))
        y_hist = np.array(list(self.history[chosen_action]['y']))

        if len(X_hist) > 0:
            self.B[chosen_action] = self.alpha * np.eye(self.n_features) + X_hist.T @ X_hist
            self.f[chosen_action] = X_hist.T @ y_hist
            self.mu[chosen_action] = np.linalg.solve(self.B[chosen_action], self.f[chosen_action])

# --- Section 6: Experimental Analysis and Simulation ---

def run_simulation(env, router, T):
    """Runs a simulation for T steps and returns cumulative regret and action history."""
    cumulative_regret = 0
    regret_history = []
    action_history = []
    optimal_action_probs = [] # For MD router

    for t in range(T):
        context = env.get_context()

        if isinstance(router, MirrorDescentRouter):
            probs = router.choose_action_probs(context)
            action = np.random.choice(env.n_actions, p=probs)
            optimal_action_probs.append(probs)
        else: # ThompsonSamplingRouter
            action = router.choose_action(context)

        report = env.execute(action, context)
        cost = report['cost']

        # Calculate instantaneous regret
        optimal_cost = np.min(report['all_true_costs'])
        inst_regret = cost - optimal_cost
        cumulative_regret += inst_regret

        regret_history.append(cumulative_regret)
        action_history.append(action)

        # Update router
        if isinstance(router, MirrorDescentRouter):
            router.update(action, cost, probs)
        else:
            router.update(action, context, cost)

    return regret_history, action_history, optimal_action_probs

def plot_results(regret_histories, labels):
    """Plots cumulative regret for different algorithms."""
    plt.figure(figsize=(12, 8))
    for regret_history, label in zip(regret_histories, labels):
        plt.plot(regret_history, label=label)
    plt.title('Cumulative Regret Over Time')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_action_probabilities(probs_history, env, change_point):
    """Plots the evolution of action probabilities for the MD router."""
    probs_df = pd.DataFrame(probs_history, columns=[f'Action {i}' for i in range(env.n_actions)])
    plt.figure(figsize=(12, 8))
    probs_df.plot(ax=plt.gca())
    plt.axvline(x=change_point, color='r', linestyle='--', label='Regime Change')
    plt.title('Action Probabilities (Mirror Descent) vs. Time')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main execution block for simulation ---
if __name__ == '__main__':
    # --- 6.1 Simulation Setup ---
    T = 10000
    N_ACTIONS = 5
    N_FEATURES = 10

    # --- 6.2 Performance Metric: Cumulative Regret Plot ---
    print("Running simulation for cumulative regret...")
    env_gradual = MarketEnvironment(N_ACTIONS, N_FEATURES, drift_mode='gradual')
    md_router = MirrorDescentRouter(N_ACTIONS)
    ts_router = ThompsonSamplingRouter(N_ACTIONS, N_FEATURES, window_size=200)

    regret_md, _, _ = run_simulation(env_gradual, md_router, T)
    regret_ts, _, _ = run_simulation(env_gradual, ts_router, T)

    plot_results([regret_md, regret_ts], ['Mirror Descent', 'Thompson Sampling'])

    # --- 6.3 Demonstrating Adaptability ---
    print("\nRunning simulation to demonstrate adaptability...")
    CHANGE_POINT = T // 2
    # Create an environment that has a single, predictable change
    class AbruptChangeEnv(MarketEnvironment):
        def _update_true_costs(self):
            if self.t == CHANGE_POINT:
                print(f"*** Regime change triggered at t={self.t} ***")
                self.true_coeffs = np.random.randn(self.n_actions, self.n_features) * 0.5
                self.true_coeffs[0] *= -5 # Make action 0 the new best

    env_abrupt = AbruptChangeEnv(N_ACTIONS, N_FEATURES)
    # Make last action best initially
    env_abrupt.true_coeffs[-1] *= -5
    md_router_adapt = MirrorDescentRouter(N_ACTIONS)
    _, _, probs_hist = run_simulation(env_abrupt, md_router_adapt, T)

    plot_action_probabilities(probs_hist, env_abrupt, CHANGE_POINT)


# --- Section 5.5: Unit Testing ---

class TestOnlineAlgorithms(unittest.TestCase):
    # ... (tests from previous response, improved for clarity and coverage)
    def test_simplex_projection(self):
        """Test that MirrorDescentRouter always outputs a valid probability distribution."""
        router = MirrorDescentRouter(n_actions=5)
        for _ in range(100):
            probs = router.choose_action_probs()
            self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
            self.assertTrue(np.all(probs >= 0))
            router.update(np.random.choice(5), np.random.rand(), probs) # Update with random action and loss

    def test_weight_update(self):
        """Test that the weight for a consistently low-loss arm grows."""
        router = MirrorDescentRouter(n_actions=3)
        initial_probs = router.choose_action_probs()

        for _ in range(100):
            probs = router.choose_action_probs()
            router.update(0, 0.1, probs) # Action 0 has low loss
            router.update(1, 0.9, probs) # Actions 1 and 2 have high loss
            router.update(2, 0.9, probs)

        final_probs = router.choose_action_probs()
        self.assertGreater(final_probs[0], initial_probs[0])
        self.assertLess(final_probs[1], initial_probs[1])
        self.assertLess(final_probs[2], initial_probs[2])

    def test_regret_scaling(self):
        """Test that cumulative regret scales sublinearly, roughly as O(sqrt(T))."""
        # Use a fixed adversarial environment for reproducibility
        class FixedAdversary(MarketEnvironment):
            def __init__(self, n_actions, n_features):
                super().__init__(n_actions, n_features)
                self.costs_sequence = [np.random.uniform(0, 1, n_actions) for _ in range(400)]
                self.t = 0
            def execute(self, action, context):
                true_costs = self.costs_sequence[self.t % len(self.costs_sequence)]
                self.t += 1
                return {'cost': true_costs[action], 'all_true_costs': true_costs}
            def get_context(self): return None # Context not needed for this test

        env = FixedAdversary(3, 1)
        regret_T, _, _ = run_simulation(env, MirrorDescentRouter(3), T=100)
        env.t = 0 # Reset environment time
        regret_4T, _, _ = run_simulation(env, MirrorDescentRouter(3), T=400)

        final_regret_T = regret_T[-1]
        final_regret_4T = regret_4T[-1]

        # Expect regret at 4T to be roughly 2x regret at T, not 4x.
        ratio = final_regret_4T / final_regret_T if final_regret_T > 0 else 0
        self.assertLess(ratio, 3.0, "Regret should scale sublinearly, O(sqrt(T))")
        self.assertGreater(ratio, 1.5, "Regret should still be growing")

    def test_adaptation_to_shock(self):
        """Test if the TS router adapts after an abrupt environmental shock."""
        class ShockEnv(MarketEnvironment):
            def __init__(self, n_actions, n_features, shock_time):
                super().__init__(n_actions, n_features)
                self.shock_time = shock_time
                self.t = 0
                # Arm 0 is best before shock
                self.true_coeffs[0] *= -10
                self.true_coeffs[1] *= 10
            def _update_true_costs(self):
                self.t += 1
                if self.t == self.shock_time:
                    # Arm 1 becomes best after shock
                    self.true_coeffs[0], self.true_coeffs[1] = self.true_coeffs[1], self.true_coeffs[0]

        T = 400
        SHOCK_TIME = 200
        env = ShockEnv(2, 1, SHOCK_TIME)
        router = ThompsonSamplingRouter(2, 1, window_size=50)

        _, actions, _ = run_simulation(env, router, T)

        # Phase 1: Before shock, should mostly choose arm 0
        actions_phase1 = actions[:SHOCK_TIME]
        self.assertGreater(actions_phase1.count(0), actions_phase1.count(1) * 2)

        # Phase 2: After shock, should learn to choose arm 1
        actions_phase2 = actions[SHOCK_TIME:]
        # Check adaptation in the second half of phase 2
        actions_phase2_late = actions[SHOCK_TIME + SHOCK_TIME//2:]
        self.assertGreater(actions_phase2_late.count(1), actions_phase2_late.count(0) * 2)


if __name__ == '__main__':
    # This block will now run the unit tests after the simulation
    print("\nRunning unit tests...")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestOnlineAlgorithms))
    runner = unittest.TextTestRunner()
    runner.run(suite)