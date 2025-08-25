import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import adfuller, coint
import statsmodels.api as sm
import unittest

class ForecastingModule:
    """
    A module for short-horizon forecasting and regime detection using
    Kalman filters, cointegration, and change-point detection.
    """
    def __init__(self):
        # Seed for reproducibility in all methods that use randomness
        np.random.seed(42)

    def generate_synthetic_data(self, n_points=500, regime_change_point=250,
                                mean1=5, mean2=-5):
        """
        Generates a synthetic dataset of two cointegrated series with a regime shift.
        """
        x1 = np.random.normal(0, 1, n_points).cumsum() + 100
        noise = np.random.normal(0, 0.5, n_points)
        spread = np.zeros(n_points)

        # First regime: mean-reverting spread around mean1
        for i in range(1, regime_change_point):
            spread[i] = 0.8 * (spread[i-1]) + np.random.normal(0, 0.2)

        # Second regime: mean-reverting spread around mean2
        spread[regime_change_point] = 0 # Reset spread at change point
        for i in range(regime_change_point + 1, n_points):
             spread[i] = 0.8 * (spread[i-1]) + np.random.normal(0, 0.2)

        # Combine to create the second series
        x2 = x1.copy()
        x2[:regime_change_point] += mean1 + spread[:regime_change_point] + noise[:regime_change_point]
        x2[regime_change_point:] += mean2 + spread[regime_change_point:] + noise[regime_change_point:]

        return x1, x2

    def kalman_filter(self, observations):
        """
        A simple 1D Kalman filter to smooth a noisy signal.
        """
        n = len(observations)
        x_hat = np.zeros(n)      # A posteriori estimate of state
        P = np.zeros(n)          # A posteriori error covariance
        x_hat_minus = np.zeros(n) # A priori estimate of state
        P_minus = np.zeros(n)    # A priori error covariance
        K = np.zeros(n)          # Kalman Gain

        A, H, Q, R = 1, 1, 1e-4, 0.1 # Model parameters

        # Initial guesses
        x_hat[0] = observations[0]
        P[0] = 1.0

        for k in range(1, n):
            # Predict
            x_hat_minus[k] = A * x_hat[k-1]
            P_minus[k] = A * P[k-1] * A + Q
            # Update
            K[k] = P_minus[k] * H / (H * P_minus[k] * H + R)
            x_hat[k] = x_hat_minus[k] + K[k] * (observations[k] - H * x_hat_minus[k])
            P[k] = (1 - K[k] * H) * P_minus[k]

        return x_hat

    def run_cointegration_test(self, y, x):
        """Performs an Engle-Granger cointegration test."""
        score, p_value, _ = coint(y, x)
        return p_value

    def estimate_ecm(self, y, x):
        """Estimates a simple Error-Correction Model."""
        spread = y - x
        delta_y = np.diff(y)
        delta_x = np.diff(x)
        spread_lagged = sm.add_constant(spread[:-1])

        # Model: delta_y = const + alpha * spread_lagged + beta * delta_x
        X = np.column_stack([spread_lagged, delta_x])
        model = sm.OLS(delta_y, X)
        results = model.fit()
        return results

    def cusum_change_point_detector(self, series, threshold_std_multiplier=5):
        """A simple CUSUM change point detector."""
        n = len(series)
        s_plus, s_minus = np.zeros(n), np.zeros(n)
        detections = []

        mean, std = np.mean(series), np.std(series)
        threshold = threshold_std_multiplier * std

        for t in range(1, n):
            s_plus[t] = max(0, s_plus[t-1] + (series[t] - mean))
            s_minus[t] = max(0, s_minus[t-1] - (series[t] - mean))
            if s_plus[t] > threshold or s_minus[t] > threshold:
                detections.append(t)
                # Reset after detection to find subsequent changes
                s_plus[t], s_minus[t] = 0, 0

        return detections
