import numpy as np
import math
import networkx as nx
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# --- Part 1: AMM Stepper Implementations ---

class UniswapV2Pool:
    """Implements the exact stepper for a Uniswap v2 constant-product pool."""
    def __init__(self, name, token0, token1, reserve0, reserve1, fee=0.003):
        self.name = name
        self.tokens = (token0, token1)
        self.reserves = {token0: float(reserve0), token1: float(reserve1)}
        self.fee = fee
        self.gamma = 1 - fee

    def __repr__(self):
        return (f"UniswapV2Pool '{self.name}' ({self.tokens[0]}/{self.tokens[1]}, "
                f"Reserves: {self.reserves[self.tokens[0]]:.2f}/{self.reserves[self.tokens[1]]:.2f})")

    def get_amount_out(self, token_in, amount_in):
        token_out = self.tokens[1] if token_in == self.tokens[0] else self.tokens[0]
        reserve_in, reserve_out = self.reserves[token_in], self.reserves[token_out]
        amount_in_with_fee = amount_in * self.gamma
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        return numerator / denominator if denominator != 0 else 0, token_out

class CurveStableswapPool:
    """Implements a simplified stepper for a Curve-like stableswap pool."""
    def __init__(self, name, token0, token1, reserve0, reserve1, A=100, fee=0.0004):
        self.name = name
        self.tokens = (token0, token1)
        self.reserves = {token0: float(reserve0), token1: float(reserve1)}
        self.A, self.n, self.fee, self.gamma = A, 2, fee, 1 - fee
        self.D = self._calculate_D([self.reserves[token0], self.reserves[token1]])

    def __repr__(self):
        return (f"CurvePool '{self.name}' ({self.tokens[0]}/{self.tokens[1]}, "
                f"Reserves: {self.reserves[self.tokens[0]]:.2f}/{self.reserves[self.tokens[1]]:.2f}, A={self.A})")

    def _calculate_D(self, xp, tol=1e-10, max_iter=256):
        S = float(sum(xp))
        if S == 0: return 0.0
        D_prev, D = 0.0, S
        Ann = self.A * self.n**self.n
        for _ in range(max_iter):
            P_prod = xp[0] * xp[1]
            if P_prod == 0: return D
            P = D**(self.n + 1) / (self.n**self.n * P_prod)
            f_D = Ann * S + D - Ann * D - P
            f_prime_D = 1 - Ann - (self.n + 1) * P / D
            if abs(f_prime_D) < 1e-6: return D
            D, D_prev = D - f_D / f_prime_D, D
            if abs(D - D_prev) < tol: return D
        raise ValueError("D calculation did not converge")

    def _get_y(self, i, j, x, D):
        Ann = self.A * self.n**self.n
        def f(y):
            if y <= 0 or x <= 0: return float('inf')
            return Ann * (x + y) + D - Ann * D - (D**(self.n + 1)) / (self.n**self.n * x * y)
        try:
            res = minimize_scalar(lambda y: abs(f(y)), bounds=(1e-6, D * 1.5), method='bounded')
            return res.x if res.success else None
        except (RuntimeError, ValueError): return None

    def get_amount_out(self, token_in, amount_in):
        i, j = self.tokens.index(token_in), 1 - self.tokens.index(token_in)
        token_out = self.tokens[j]
        new_reserve_in = self.reserves[token_in] + amount_in * self.gamma
        new_reserve_out = self._get_y(i, j, new_reserve_in, self.D)
        if new_reserve_out is None: return 0, token_out
        amount_out = self.reserves[token_out] - new_reserve_out
        return amount_out if amount_out > 0 else 0, token_out

# --- Part 2: Graph Construction and Arbitrage Detection ---

def build_dex_graph(pools):
    G = nx.MultiDiGraph()
    for pool in pools:
        t0, t1 = pool.tokens
        rate0, _ = pool.get_amount_out(t0, 1.0)
        if rate0 > 0: G.add_edge(t0, t1, key=pool.name, weight=-math.log(rate0), pool=pool)
        rate1, _ = pool.get_amount_out(t1, 1.0)
        if rate1 > 0: G.add_edge(t1, t0, key=pool.name, weight=-math.log(rate1), pool=pool)
    return G

def find_arbitrage_cycles(graph):
    all_cycles = []
    found_cycles_canonical = set()
    for node in graph.nodes:
        try:
            cycle_nodes = nx.find_negative_cycle(graph, source=node, weight='weight')
            canonical_key = frozenset(cycle_nodes[:-1])
            if canonical_key in found_cycles_canonical:
                continue
            all_cycles.append(cycle_nodes)
            found_cycles_canonical.add(canonical_key)
        except nx.NetworkXError:
            continue
    return all_cycles

# --- Part 3: Optimal Trade Size Calculation ---

def get_pools_from_cycle(full_path, graph):
    pools = []
    for i in range(len(full_path) - 1):
        u, v = full_path[i], full_path[i+1]
        min_weight = float('inf')
        best_pool = None
        for key, edge_data in graph.get_edge_data(u, v).items():
            if edge_data['weight'] < min_weight:
                min_weight = edge_data['weight']
                best_pool = edge_data['pool']
        if best_pool:
            pools.append(best_pool)
        else:
            raise ValueError(f"Could not find a pool for the edge {u}->{v}")
    return pools

def calculate_profit_for_cycle(amount_in, cycle_path, pools):
    amount, token = amount_in, cycle_path[0]
    for pool in pools:
        amount, token = pool.get_amount_out(token, amount)
        if amount <= 1e-9: return -float('inf')
    return amount - amount_in if token == cycle_path[0] else -float('inf')

def find_optimal_trade_size(cycle_path, pools):
    def objective(amount_in):
        return -calculate_profit_for_cycle(amount_in, cycle_path, pools) if amount_in > 0 else 0
    start_token = cycle_path[0]
    max_bound = pools[0].reserves[start_token] * 0.25
    result = minimize_scalar(objective, bounds=(0, max_bound), method='bounded')
    return (result.x, -result.fun) if result.success and -result.fun > 1e-6 else (0, 0)
