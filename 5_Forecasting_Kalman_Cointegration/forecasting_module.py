#
# Part B: Colab-Ready Python Implementation
#
# This block contains a complete, self-contained implementation for a
# forecasting and regime detection module as specified in the user query.
# It includes a synthetic environment, model implementations, validation tests,
# and visualization.
#

#
# I. Preliminaries and Library Imports
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller

#
# II. Synthetic Market Environment Generator
#
def generate_synthetic_data(n_points=1000, seed=42):
    """
    Generates a synthetic dataset of two correlated assets with a mean-reverting
    spread and a mid-point regime shift in volatility.

    Args:
        n_points (int): Number of data points to generate.
        seed (int): Seed for the random number generator for reproducibility.

    Returns:
        tuple: A tuple containing:
            - p1 (np.ndarray): Price series of asset 1.
            - p2 (np.ndarray): Price series of asset 2.
            - true_state (np.ndarray): The true underlying price of asset 1 without noise.
            - true_spread (np.ndarray): The true underlying mean-reverting spread.
    """
    rng = np.random.default_rng(seed)

    # --- Parameters ---
    # Cointegration and spread parameters
    beta = 1.0  # Cointegrating vector [1, -beta]
    spread_mean = 0.5
    spread_reversion_speed = 0.1  # Theta in Ornstein-Uhlenbeck
    spread_vol = 0.05

    # Common component (random walk) parameters
    common_vol_regime1 = 0.1
    common_vol_regime2 = 0.3  # Increased volatility after regime shift
    regime_shift_point = n_points // 2

    # Measurement noise
    measurement_noise_std = 0.02

    # --- Data Generation ---
    # 1. Generate the mean-reverting spread (Ornstein-Uhlenbeck process)
    true_spread = np.zeros(n_points)
    true_spread[0] = spread_mean
    for t in range(1, n_points):
        dt = 1
        d_spread = spread_reversion_speed * (spread_mean - true_spread[t-1]) * dt \
                   + spread_vol * rng.normal() * np.sqrt(dt)
        true_spread[t] = true_spread[t-1] + d_spread

    # 2. Generate the common random walk component with a regime shift
    common_component = np.zeros(n_points)
    innovations = np.zeros(n_points)
    innovations[:regime_shift_point] = rng.normal(0, common_vol_regime1, regime_shift_point)
    innovations[regime_shift_point:] = rng.normal(0, common_vol_regime2, n_points - regime_shift_point)
    common_component = np.cumsum(innovations) + 100 # Start price at 100

    # 3. Construct the two asset prices
    p1_true = common_component + 0.5 * true_spread
    p2_true = p1_true - true_spread

    # 4. Add measurement noise to get observed prices
    p1_observed = p1_true + rng.normal(0, measurement_noise_std, n_points)
    p2_observed = p2_true + rng.normal(0, measurement_noise_std, n_points)


    return p1_observed, p2_observed, p1_true, true_spread

#
# III. Model Implementations
#
class KalmanFilter:
    """
    A numerically stable implementation of a 1D Kalman Filter for a
    constant velocity model.
    State vector x = [position, velocity]^T
    """
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        self.dt = dt
        # State Transition Matrix (Phi)
        self.Phi = np.array([[1, dt], [0, 1]])
        # Observation Matrix (H)
        self.H = np.array([[1, 0]])
        # Process Noise Covariance (Q)
        # Using a standard process noise model for CV model
        G = np.array([[0.5 * dt**2], [dt]])
        self.Q = G @ G.T * process_noise_std**2
        # Measurement Noise Covariance (R)
        self.R = np.array([[measurement_noise_std**2]])
        # Initial state and covariance
        self.x_hat = np.zeros((2, 1))
        self.P = np.eye(2) * 500  # Large initial uncertainty

    def predict(self):
        # Time Update (Predict)
        self.x_hat = self.Phi @ self.x_hat
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q

    def update(self, z):
        # Measurement Update (Correct)
        innovation = z - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ innovation

        # Joseph form for numerical stability
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x_hat, innovation.item()

def estimate_ecm(p1, p2):
    """
    Performs cointegration analysis and estimates a VECM.
    """
    print("--- Starting Cointegration and ECM Analysis ---")
    df = pd.DataFrame({'p1': p1, 'p2': p2})

    # 1. Test for unit roots (I(1)) in individual series
    print("ADF Test for p1:", adfuller(df['p1'])[1]) # Print p-value
    print("ADF Test for p2:", adfuller(df['p2'])[1]) # Print p-value

    # 2. Johansen Cointegration Test
    # det_order is -1 for no deterministic terms, 0 for constant, 1 for trend
    # k_ar_diff is number of lags in differences
    johansen_result = coint_johansen(df, det_order=0, k_ar_diff=1)
    trace_stat = johansen_result.lr1
    trace_crit = johansen_result.cvt[:, 1] # 95% critical values

    print(f"Trace Statistic: {trace_stat[0]}") # Print statistic for r=0
    print(f"Critical Values (90%, 95%, 99%): {trace_crit}")

    if trace_stat[0] > trace_crit[0]: # Test for r=0 vs r>0 at 95%
        print("Result: Cointegration detected (rank=1 or more).")
        coint_rank = 1
    else:
        print("Result: No cointegration detected.")
        return None, None

    # 3. Estimate VECM
    model = VECM(df, k_ar_diff=1, coint_rank=coint_rank, deterministic='ci')
    vecm_res = model.fit()

    # Extract cointegrating vector (beta) and speed of adjustment (alpha)
    beta = np.append([1.0], -vecm_res.beta[0,0]) # Correct sign for beta
    alpha = vecm_res.alpha

    print("\n--- VECM Estimation Results ---")
    print(f"Estimated Cointegrating Vector (beta): {beta}")
    print(f"Estimated Speed of Adjustment (alpha): {alpha}")

    return vecm_res, df

def cusum_detector(innovations, threshold=5.0, drift=0.1):
    """
    Simple CUSUM detector for mean shifts away from zero.
    """
    s_pos = 0
    s_neg = 0
    detections = []
    for i, val in enumerate(innovations):
        s_pos = max(0, s_pos + val - drift)
        s_neg = max(0, s_neg - val - drift)
        if s_pos > threshold or s_neg > threshold:
            detections.append(i)
            # Reset after detection to find subsequent changes
            s_pos = 0
            s_neg = 0
    return detections

#
# IV. Main Execution and Testing Block
#
if __name__ == '__main__':
    # --- 1. Generate Synthetic Data ---
    # This is where you would plug in your real tick data for p1 and p2.
    # Ensure they are numpy arrays.
    p1_obs, p2_obs, p1_true, spread_true = generate_synthetic_data(n_points=1000, seed=42)

    # --- 2. Test Kalman Filter ---
    print("--- Testing Kalman Filter ---")
    kf = KalmanFilter(dt=1, process_noise_std=0.05, measurement_noise_std=0.02)
    estimates = []
    innovations = []
    for z in p1_obs:
        est, inn = kf.update(z)
        kf.predict()
        estimates.append(est.copy())
        innovations.append(inn)

    estimates = np.array(estimates).squeeze()

    # Performance Metric: MSE
    mse_kalman = np.mean((estimates[:, 0] - p1_true)**2)
    mse_naive = np.mean((p1_obs[1:] - p1_true[:-1])**2) # Naive: predict previous obs
    print(f"Kalman Filter MSE: {mse_kalman:.6f}")
    print(f"Naive Predictor MSE: {mse_naive:.6f}")
    assert mse_kalman < mse_naive, "Kalman filter should outperform naive predictor"
    print("Kalman filter successfully reduced MSE vs. naive predictor.")

    # --- 3. Test ECM ---
    vecm_res, df = estimate_ecm(p1_obs, p2_obs)
    if vecm_res:
        # Check if alpha for p1 has the correct (negative) sign for reversion
        assert vecm_res.alpha[0,0] < 0, "Alpha for p1 should be negative" # Access alpha correctly
        print("\nECM correctly predicts spread reversion (alpha for p1 is negative).")

        # Generate a 5-step forecast with confidence intervals
        forecast, lower, upper = vecm_res.predict(steps=5, alpha=0.05)
        print("\n5-step ahead forecast for p1 and p2:")
        print(pd.DataFrame(forecast, columns=['p1_forecast', 'p2_forecast']))

    # --- 4. Test CUSUM Change-Point Detector ---
    print("\n--- Testing CUSUM Detector ---")
    regime_shift_point = len(p1_obs) // 2
    detections = cusum_detector(innovations, threshold=0.5, drift=0.01)
    print(f"True regime shift at index: {regime_shift_point}")
    print(f"CUSUM detected changes at indices: {detections}")

    # Check if a detection occurred near the true shift point
    detected_correctly = any(abs(d - regime_shift_point) < 50 for d in detections)
    assert detected_correctly, "CUSUM failed to detect the regime shift"
    print("CUSUM successfully detected the regime shift.")

    # --- 5. Visualization ---
    fig, ax = plt.subplots(figsize=(14, 7))

    time_axis = np.arange(len(p1_true))

    ax.plot(time_axis, p1_true, linestyle='--', label='True State (p1_true)')
    ax.plot(time_axis, p1_obs, alpha=0.5, marker='.', linestyle='none', label='Noisy Observations (p1_obs)')
    ax.plot(time_axis, estimates[:, 0], linewidth=2, label='Kalman Filter Estimate')

    # Highlight the true regime shift
    ax.axvline(regime_shift_point, linestyle=':', linewidth=2, label='True Regime Shift')

    # Highlight detected change points
    for d in detections:
        ax.axvline(d, linestyle='-.', linewidth=1.5, label=f'CUSUM Detection at {d}' if d == detections[0] else "") # Avoid duplicate labels

    ax.set_title('Kalman Filter State Estimation with Regime Shift Detection')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(linestyle=':')
    plt.tight_layout()
    plt.show()