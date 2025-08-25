import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import unittest

# --- Module Interfaces (ABCs) ---

class CEXFeeModule(ABC):
    @abstractmethod
    def get_fee(self, trade_value: float, order_type: str, volume_30d: float) -> float:
        """Calculates the trading fee for a CEX."""
        pass

class DEXFeeModule(ABC):
    @abstractmethod
    def get_fee(self, trade_value: float) -> float:
        """Calculates the protocol fee for a DEX."""
        pass

class GasEstimator(ABC):
    @abstractmethod
    def estimate_gas_cost(self, g_units: int, p_native: float) -> float:
        """Estimates the total gas cost for a transaction."""
        pass

class FundingRateSimulator(ABC):
    @abstractmethod
    def simulate_holding_cost(self, notional_value: float, periods: int) -> float:
        """Simulates the total funding cost over a holding period."""
        pass

class SlippageFunction(ABC):
    @abstractmethod
    def calculate_slippage(self, trade_size_in: float) -> tuple[float, float]:
        """Calculates the output amount and slippage cost for a trade."""
        pass

# --- Concrete Implementations ---

class TieredCEXFeeModule(CEXFeeModule):
    """Implements a tiered maker/taker fee schedule for a CEX."""
    def __init__(self, fee_tiers: dict):
        # fee_tiers = {volume_threshold: {'maker': fee, 'taker': fee}}
        self.fee_tiers = sorted(fee_tiers.items())

    def get_fee(self, trade_value: float, order_type: str, volume_30d: float) -> float:
        if order_type not in ['maker', 'taker']:
            raise ValueError("Order type must be 'maker' or 'taker'")
        
        applicable_tier = self.fee_tiers[0][1] # Default to lowest tier
        for threshold, rates in self.fee_tiers:
            if volume_30d >= threshold:
                applicable_tier = rates
            else:
                break
        
        fee_rate = applicable_tier[order_type]
        return trade_value * fee_rate

class ConstantDEXFeeModule(DEXFeeModule):
    """Implements a constant percentage protocol fee for a DEX."""
    def __init__(self, fee_rate: float):
        self.fee_rate = fee_rate

    def get_fee(self, trade_value: float) -> float:
        return trade_value * self.fee_rate

class HistoricalGasEstimator(GasEstimator):
    """Estimates gas cost based on historical average base fee."""
    def __init__(self, historical_base_fees_gwei: list, priority_fee_gwei: float = 2.0):
        self.avg_base_fee = np.mean(historical_base_fees_gwei)
        self.priority_fee = priority_fee_gwei

    def estimate_gas_cost(self, g_units: int, p_native: float) -> float:
        total_gwei = (self.avg_base_fee + self.priority_fee) * g_units
        return total_gwei * p_native / 1e9  # Convert Gwei to native token

class BootstrapFundingRateSimulator(FundingRateSimulator):
    """Simulates funding costs by bootstrapping historical rates."""
    def __init__(self, historical_rates: list, n_simulations: int = 1000):
        self.historical_rates = np.array(historical_rates)
        self.n_simulations = n_simulations

    def simulate_holding_cost(self, notional_value: float, periods: int) -> float:
        simulated_costs = []
        for _ in range(self.n_simulations):
            sampled_rates = np.random.choice(self.historical_rates, size=periods, replace=True)
            total_cost = notional_value * np.sum(sampled_rates)
            simulated_costs.append(total_cost)
        
        return np.mean(simulated_costs)

class ConstantProductSlippage(SlippageFunction):
    """Calculates slippage for a constant product (x*y=k) AMM."""
    def __init__(self, reserve_x: float, reserve_y: float, protocol_fee_rate: float):
        self.reserve_x = reserve_x
        self.reserve_y = reserve_y
        self.k = reserve_x * reserve_y
        self.gamma = 1 - protocol_fee_rate

    def calculate_slippage(self, trade_size_in: float) -> tuple[float, float]:
        if self.reserve_x <= 0 or self.reserve_y <= 0:
            return 0, float('inf')
            
        initial_price = self.reserve_y / self.reserve_x
        amount_in_with_fee = trade_size_in * self.gamma
        amount_out_y = (self.reserve_y * amount_in_with_fee) / (self.reserve_x + amount_in_with_fee)
        expected_out_y = trade_size_in * initial_price
        slippage_cost_in_y = expected_out_y - amount_out_y
        
        return amount_out_y, slippage_cost_in_y

# --- Main CostModel Class ---
class CostModel:
    """A unified transaction cost model for CEX and DEX trading."""
    def __init__(self, venues_config: dict):
        self.venues = venues_config

    def calculate_trade_cost(self, venue_name: str, trade_details: dict) -> dict:
        if venue_name not in self.venues:
            raise ValueError(f"Venue '{venue_name}' not configured.")
        
        config = self.venues[venue_name]
        costs = {'total': 0, 'fee': 0, 'slippage': 0, 'gas': 0, 'withdrawal': 0}

        if config['type'] == 'CEX':
            costs['fee'] = config['fee_module'].get_fee(
                trade_details['trade_value'],
                trade_details['order_type'],
                trade_details['volume_30d']
            )
            costs['slippage'] = trade_details['trade_value'] * 0.0002
            costs['withdrawal'] = config.get('withdrawal_fee', 0)

        elif config['type'] == 'DEX':
            _, slippage_cost_in_y = config['slippage_module'].calculate_slippage(trade_details['trade_size_in'])
            
            p_y = trade_details['trade_value'] / trade_details['trade_size_in']
            costs['slippage'] = slippage_cost_in_y * p_y

            costs['gas'] = config['gas_estimator'].estimate_gas_cost(
                trade_details['g_units'],
                trade_details['p_native']
            )
        
        costs['total'] = sum(costs.values())
        return costs
