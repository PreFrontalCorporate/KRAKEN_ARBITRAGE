import heapq
import time
import random
import numpy as np
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Optional

# --- Data Structures ---
@dataclass(order=True)
class Event:
    timestamp: float
    action: str
    data: Any = field(compare=False)

@dataclass
class Transaction:
    tx_id: int
    sender: str
    arrival_time: float
    gas_limit: int
    priority_fee_gwei: float
    max_fee_per_gas: float
    is_swap: bool = False
    swap_details: Dict = field(default_factory=dict)

    @property
    def effective_gas_price(self) -> float:
        return self.priority_fee_gwei

@dataclass
class Bundle:
    bundle_id: str
    transactions: List[Transaction]
    submit_time: float

    @property
    def total_gas_limit(self) -> int:
        return sum(tx.gas_limit for tx in self.transactions)

    @property
    def effective_gas_price(self) -> float:
        total_priority_fee = sum(tx.priority_fee_gwei * tx.gas_limit for tx in self.transactions)
        return total_priority_fee / self.total_gas_limit if self.total_gas_limit > 0 else 0

# --- Core Components ---
class Mempool:
    """Simulates the transaction mempool."""
    def __init__(self):
        self.pending_txs: Dict[int, Transaction] = {}
        self.pending_bundles: Dict[str, Bundle] = {}

    def add_transaction(self, tx: Transaction):
        self.pending_txs[tx.tx_id] = tx

    def add_bundle(self, bundle: Bundle):
        self.pending_bundles[bundle.bundle_id] = bundle

    def get_pending_items(self) -> List[Tuple[float, Any]]:
        items = []
        for tx in self.pending_txs.values():
            items.append((-tx.effective_gas_price, tx))
        for bundle in self.pending_bundles.values():
            items.append((-bundle.effective_gas_price, bundle))
        items.sort(key=lambda x: x[0])
        return [(item[0], item[1]) for item in items]

    def clear(self):
        self.pending_txs.clear()
        self.pending_bundles.clear()
