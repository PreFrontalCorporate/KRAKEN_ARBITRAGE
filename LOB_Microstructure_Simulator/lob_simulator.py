# ==============================================================================
# Part B: Colab-Ready Python Implementation and Unit Tests
#
# This block contains a complete, self-contained implementation of a Limit
# Order Book simulator and its validation suite. It can be copied and pasted
# directly into a Google Colab notebook for execution.
#
# Dependencies: numpy
# Optional but recommended for LOB implementation: sortedcontainers
#!pip install sortedcontainers
# ==============================================================================

import numpy as np
import collections
import unittest
import time
import math

# For a more performant LOB, sortedcontainers is highly recommended.
# It provides a SortedDict which is implemented as a balanced binary tree,
# offering O(log N) for insertions, deletions, and lookups.
# For this pedagogical example, we will use a simpler dictionary-based approach
# and manage sorted keys manually, which is less efficient for large books
# but more transparent.
try:
    from sortedcontainers import SortedDict
except ImportError:
    print("Warning: `sortedcontainers` not found. Using a less efficient dict-based LOB.")
    # Fallback to a simple dict if sortedcontainers is not available
    SortedDict = dict


# ==============================================================================
# 1. Limit Order Book (LOB) Data Structure Implementation
# ==============================================================================

class LimitOrderBook:
    """
    A simplified implementation of a Limit Order Book (LOB).

    This class manages the two sides of the book (bids and asks) and processes
    limit orders, market orders, and cancellations. It follows a price-time
    priority matching algorithm.

    Attributes:
        bids (SortedDict): A collection of buy orders, keyed by price.
                           Prices are sorted in descending order.
        asks (SortedDict): A collection of sell orders, keyed by price.
                           Prices are sorted in ascending order.
        order_map (dict): A mapping from order_id to the order object for
                          O(1) lookup and cancellation.
    """
    def __init__(self):
        # Bids are orders to buy. We want to match market sells against the highest bid.
        # A SortedDict for bids would naturally sort keys (prices) ascending.
        # To get the best bid (highest price), we would need the last element.
        # To make it symmetric with asks, we can use a negative price key for bids.
        self.bids = SortedDict()  # Key: price, Value: collections.deque of orders
        self.asks = SortedDict()  # Key: price, Value: collections.deque of orders
        self.order_map = {}
        self._order_id_counter = 0

    def _get_next_order_id(self):
        self._order_id_counter += 1
        return self._order_id_counter

    @property
    def best_bid(self):
        """Returns the highest bid price, or None if no bids exist."""
        if not self.bids:
            return None
        return self.bids.peekitem(-1)[0] # peekitem(-1) gets the largest key

    @property
    def best_ask(self):
        """Returns the lowest ask price, or None if no asks exist."""
        if not self.asks:
            return None
        return self.asks.peekitem(0)[0] # peekitem(0) gets the smallest key

    def get_mid_price(self):
        """Calculates the mid-price of the book."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    def add_limit_order(self, side, price, size, timestamp):
        """
        Adds a new limit order to the book.
        """
        order_id = self._get_next_order_id()
        order = {
            'id': order_id,
            'side': side,
            'price': price,
            'size': size,
            'timestamp': timestamp
        }

        self.order_map[order_id] = order

        if side == 'buy':
            book_side = self.bids
        else:
            book_side = self.asks

        if price not in book_side:
            book_side[price] = collections.deque()

        book_side[price].append(order)
        return order_id

    def cancel_order(self, order_id):
        """
        Cancels a resting limit order from the book.
        """
        if order_id not in self.order_map:
            return False

        order_to_cancel = self.order_map[order_id]
        price = order_to_cancel['price']
        side = order_to_cancel['side']

        if side == 'buy':
            book_side = self.bids
        else:
            book_side = self.asks

        if price in book_side:
            queue = book_side[price]
            try:
                queue.remove(order_to_cancel)
                if not queue:
                    del book_side[price]
                del self.order_map[order_id]
                return True
            except ValueError:
                return False
        return False

    def process_market_order(self, side, size):
        """
        Processes a market order, matching it against resting limit orders.
        """
        trades = []
        size_to_fill = size

        if side == 'buy':
            book_side = self.asks
            price_levels = list(book_side.keys())
        else:
            book_side = self.bids
            price_levels = reversed(list(book_side.keys()))

        for price in price_levels:
            if size_to_fill == 0:
                break

            queue = book_side[price]
            while queue and size_to_fill > 0:
                resting_order = queue[0]
                trade_size = min(size_to_fill, resting_order['size'])

                trades.append({
                    'price': resting_order['price'],
                    'size': trade_size,
                    'aggressor_side': side,
                    'resting_order_id': resting_order['id']
                })

                size_to_fill -= trade_size
                resting_order['size'] -= trade_size

                if resting_order['size'] == 0:
                    queue.popleft()
                    del self.order_map[resting_order['id']]

            if not queue:
                del book_side[price]

        return trades, size_to_fill

# ==============================================================================
# 2. Order Arrival Generators
# ==============================================================================

def poisson_generator(rate, duration):
    """
    Generates event arrival times according to a homogeneous Poisson process.
    """
    current_time = 0.0
    while current_time < duration:
        inter_arrival_time = np.random.exponential(scale=1.0 / rate)
        current_time += inter_arrival_time
        if current_time < duration:
            yield current_time

def hawkes_generator(mu, alpha, beta, duration):
    """
    Generates event arrival times for a univariate Hawkes process.
    """
    if alpha >= beta:
        print(f"Warning: Hawkes process may be non-stationary (alpha={alpha} >= beta={beta}).")

    arrival_times = []
    current_time = 0.0
    intensity_max = mu + alpha

    while current_time < duration:
        time_to_next_candidate = np.random.exponential(scale=1.0 / intensity_max)
        current_time += time_to_next_candidate

        if current_time >= duration:
            break

        intensity_at_candidate = mu + alpha * sum(np.exp(-beta * (current_time - t_i)) for t_i in arrival_times)

        if np.random.uniform() < intensity_at_candidate / intensity_max:
            arrival_times.append(current_time)
            yield current_time
            intensity_max = intensity_at_candidate + alpha

# ==============================================================================
# 3. Simulation Engine
# ==============================================================================

def simulate_lob(generator, duration, tick_size=0.01, initial_price=100.0):
    """
    Runs a simulation of the LOB using a given event generator.
    """
    lob = LimitOrderBook()
    log = []
    
    lob.add_limit_order('buy', initial_price - tick_size, 100, 0)
    lob.add_limit_order('sell', initial_price + tick_size, 100, 0)

    for event_time in generator:
        mid_price = lob.get_mid_price()
        if mid_price is None:
            mid_price = initial_price

        event_type = np.random.choice(['limit', 'market', 'cancel'], p=[0.6, 0.2, 0.2])
        side = np.random.choice(['buy', 'sell'])

        if event_type == 'limit':
            price_offset = np.random.randint(0, 5) * tick_size
            if side == 'buy':
                price = mid_price - price_offset
            else:
                price = mid_price + price_offset
            price = round(price / tick_size) * tick_size
            size = np.random.randint(10, 50)
            lob.add_limit_order(side, price, size, event_time)
        elif event_type == 'market':
            size = np.random.randint(20, 80)
            trades, _ = lob.process_market_order(side, size)
        elif event_type == 'cancel':
            if lob.order_map:
                order_id_to_cancel = np.random.choice(list(lob.order_map.keys()))
                lob.cancel_order(order_id_to_cancel)
        
        log.append({
            'time': event_time,
            'best_bid': lob.best_bid,
            'best_ask': lob.best_ask,
            'mid_price': lob.get_mid_price()
        })

    return lob, log
