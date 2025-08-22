# MEV/Mempool Simulation and Adversarial Economics Laboratory
#
# This Colab-ready Python script implements a discrete-time event simulation
# of a blockchain mempool, block construction, and adversarial MEV extraction.
# It is designed as a foundational toolkit for prototyping and analyzing
# MEV strategies like sandwich attacks.

# --- Core Dependencies ---
import heapq
import time
import random
import numpy as np
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Optional

# --- Configuration Parameters (from Table 1) ---
# This dictionary centralizes all simulation parameters for reproducibility.
CONFIG = {
    # Mempool Dynamics
    "TX_ARRIVAL_RATE_LAMBDA": 5.0,  # tx/sec
    "BUNDLE_ARRIVAL_RATE_LAMBDA": 0.5, # bundles/sec (not used in this simplified model)
    "SIMULATION_DURATION": 120, # seconds

    # Blockchain State
    "BLOCK_INTERVAL": 12, # seconds
    "INITIAL_BASE_FEE": 20, # Gwei
    "BLOCK_GAS_LIMIT": 30_000_000,

    # CFMM Configuration
    "INITIAL_RESERVE_X": 1000.0, # e.g., WETH
    "INITIAL_RESERVE_Y": 2_000_000.0, # e.g., USDC

    # Gas Cost Model
    "GAS_PER_SWAP": 150_000,
    "GAS_PER_BUNDLE": 50_000, # Overhead for bundle execution

    # Adversarial Agents
    "NUM_SANDWICH_BOTS": 1,
    "BOT_LATENCY_MS": 50, # milliseconds

    # Experiment Control
    "DETERMINISTIC_SEED": 42,
}

# --- Setup for Reproducibility ---
def set_seed(seed: int):
    """Sets the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)

set_seed(CONFIG["DETERMINISTIC_SEED"])

# --- Section 4: Architecture of the Mempool Simulator ---

# 4.1. Core Components & Event-Driven Loop

@dataclass(order=True)
class Event:
    """Represents an event in the simulation queue."""
    timestamp: float
    action: str
    data: Any = field(compare=False)

@dataclass
class Transaction:
    """Represents a single blockchain transaction."""
    tx_id: int
    sender: str
    arrival_time: float
    gas_limit: int
    priority_fee_gwei: float # Per gas unit
    # EIP-1559 fields
    max_fee_per_gas: float # Includes base_fee + priority_fee
    # For swaps
    is_swap: bool = False
    swap_details: Dict = field(default_factory=dict)

    @property
    def effective_gas_price(self) -> float:
        """Priority fee per gas, used for sorting."""
        return self.priority_fee_gwei

@dataclass
class Bundle:
    """Represents an atomic bundle of transactions from a searcher."""
    bundle_id: str
    transactions: List[Transaction]
    submit_time: float

    @property
    def total_gas_limit(self) -> int:
        return sum(tx.gas_limit for tx in self.transactions)

    @property
    def effective_gas_price(self) -> float:
        """Calculates the effective gas price for the entire bundle."""
        total_priority_fee = sum(tx.priority_fee_gwei * tx.gas_limit for tx in self.transactions)
        return total_priority_fee / self.total_gas_limit if self.total_gas_limit > 0 else 0


class Clock:
    """Manages simulation time and the event queue."""
    def __init__(self):
        self.current_time = 0.0
        self.event_queue = []

    def schedule_event(self, delay: float, action: str, data: Any = None):
        """Schedules a future event."""
        heapq.heappush(self.event_queue, Event(self.current_time + delay, action, data))

    def next_event(self) -> Optional[Event]:
        """Pops the next event from the queue and advances time."""
        if not self.event_queue:
            return None
        event = heapq.heappop(self.event_queue)
        self.current_time = event.timestamp
        return event

class Mempool:
    """Simulates the transaction mempool."""
    def __init__(self):
        self.pending_txs: Dict[int, Transaction] = {}
        self.pending_bundles: Dict[str, Bundle] = {}

    def add_transaction(self, tx: Transaction):
        print(f" Mempool: Received TX {tx.tx_id} with priority {tx.priority_fee_gwei:.2f} Gwei")
        self.pending_txs[tx.tx_id] = tx

    def add_bundle(self, bundle: Bundle):
        print(f" Mempool: Received Bundle {bundle.bundle_id} from bot.")
        self.pending_bundles[bundle.bundle_id] = bundle

    def get_pending_items(self) -> List[Tuple[float, Any]]:
        """Returns all pending items sorted by effective gas price."""
        items = []
        for tx in self.pending_txs.values():
            items.append((-tx.effective_gas_price, tx)) # Use negative for max-heap behavior
        for bundle in self.pending_bundles.values():
            items.append((-bundle.effective_gas_price, bundle))

        items.sort() # sorts by first element of tuple (negative effective gas price)
        return items

    def clear(self):
        """Clears the mempool after a block is built."""
        self.pending_txs.clear()
        self.pending_bundles.clear()

# ... (Rest of the code from the previous response, including CostModel, CFMM, BlockBuilder, VirtualMachine, TransactionGenerator, SandwichBot, experiments, and unit tests)