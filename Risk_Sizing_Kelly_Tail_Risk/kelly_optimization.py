import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.stats import genpareto, t

# --- Module 1: Classical and Fractional Kelly ---
class KellySizer:
    def calculate_classical_kelly(self, p, b, a=1.0):
        """Calculates the classical Kelly criterion fraction for a binary bet."""
        q = 1 - p
        if (p * b - q * a) <= 0:
            return 0.0
        return (p * b - q * a) / (a * b)

    def run_kelly_simulation(self, f, returns, n_simulations, n_trials, initial_wealth=100):
        """Runs a Monte Carlo simulation for a given betting fraction."""
        wealth_paths = np.zeros((n_simulations, n_trials + 1))
        wealth_paths[:, 0] = initial_wealth
        for i in range(n_simulations):
            sim_returns = np.random.choice(returns, size=n_trials, replace=True)
            for t in range(n_trials):
                wealth_paths[i, t+1] = wealth_paths[i, t] * (1 + f * sim_returns[t])
                if wealth_paths[i, t+1] <= 0:
                    wealth_paths[i, t+1:] = 0
                    break
        return wealth_paths

# --- Module 2: Drawdown-Constrained Kelly Optimizer ---
def optimize_drawdown_constrained_kelly(return_scenarios, alpha, beta):
    """Solves the drawdown-constrained Kelly problem using CVXPY."""
    num_scenarios, num_assets = return_scenarios.shape
    pi = np.ones(num_scenarios) / num_scenarios
    b = cp.Variable(num_assets)
    lambda_param = np.log(beta) / np.log(alpha)
    
    log_growth = pi @ cp.log(return_scenarios @ b)
    objective = cp.Maximize(log_growth)
    
    constraints = [
        cp.sum(b) == 1,
        b >= 0,
        pi @ cp.power(return_scenarios @ b, -lambda_param) <= 1
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: Optimizer failed. Status: {problem.status}")
        return np.zeros(num_assets)
        
    return b.value

# --- Module 3: EVT-based Tail Risk Calculator ---
class EVTModel:
    def calculate_evt_cvar(self, losses, confidence_level=0.95, threshold_quantile=0.95):
        """Calculates VaR and CVaR using the Peaks-Over-Threshold EVT method."""
        losses = pd.Series(losses).dropna()
        u = losses.quantile(threshold_quantile)
        excesses = losses[losses > u] - u
        
        if len(excesses) < 30: # Fallback for insufficient data
            var_hist = losses.quantile(confidence_level)
            cvar_hist = losses[losses > var_hist].mean()
            return {'Method': 'Historical', 'VaR': var_hist, 'CVaR': cvar_hist}

        xi, _, sigma = genpareto.fit(excesses, floc=0)
        N, Nu = len(losses), len(excesses)
        phi_u = Nu / N

        var = u + (sigma / xi) * ((((1 - confidence_level) / phi_u)**(-xi)) - 1)
        cvar = var + (sigma + xi * (var - u)) / (1 - xi)

        return {
            'Method': 'EVT-GPD', 'VaR': var, 'CVaR': cvar,
            'xi (tail_index)': xi, 'sigma (scale)': sigma
        }
