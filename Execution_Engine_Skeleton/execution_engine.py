import time
import uuid
import logging
import unittest
from enum import Enum

# --- Enums for Order Types ---
class OrderStatus(Enum):
    PENDING = 1
    OPEN = 2
    CLOSED = 3
    CANCELLED = 4
    ERROR = 5

class OrderSide(Enum):
    BUY = 1
    SELL = 2

class OrderType(Enum):
    MARKET = 1
    LIMIT = 2

# --- Mock API Layer ---
class MockKrakenAPI:
    """A mock Kraken API to simulate responses."""
    def __init__(self):
        self.orders = {}
        self.current_id = 1000

    def place_order(self, pair, side, order_type, amount, price=None):
        if order_type == "LIMIT" and price is None:
            return {"error": ["EOrder:Invalid price"]}
        
        self.current_id += 1
        order_id = f"TXID_{self.current_id}"
        self.orders[order_id] = {
            "pair": pair, "side": side, "type": order_type,
            "amount": amount, "price": price, "status": "open"
        }
        logging.info(f"[Mock API] Placed order {order_id}")
        return {"error": [], "result": {"txid": [order_id]}}

    def cancel_order(self, order_id):
        if order_id in self.orders and self.orders[order_id]["status"] == "open":
            self.orders[order_id]["status"] = "canceled"
            logging.info(f"[Mock API] Canceled order {order_id}")
            return {"error": [], "result": {"count": 1}}
        return {"error": ["EOrder:Unknown order"]}

    def get_order_status(self, order_id):
        return self.orders.get(order_id, {"status": "unknown"})

# --- Core Execution Engine ---
class ExecutionEngine:
    """Manages order lifecycle and interacts with the exchange."""
    def __init__(self, dry_run=True):
        self.api = MockKrakenAPI()
        self.orders = {} # Internal state: {internal_id: order_details}
        self.dry_run = dry_run

    def place_order(self, pair, side: OrderSide, order_type: OrderType, amount, price=None):
        internal_id = str(uuid.uuid4())
        order_details = {
            "pair": pair, "side": side, "type": order_type,
            "amount": amount, "price": price,
            "status": OrderStatus.PENDING, "exchange_id": None
        }
        self.orders[internal_id] = order_details
        
        if self.dry_run:
            logging.info(f"[Engine DRY RUN] Would place order: {order_details}")
            # Simulate success for dry run
            self.orders[internal_id]["status"] = OrderStatus.OPEN
            self.orders[internal_id]["exchange_id"] = f"DRYRUN_{internal_id[:8]}"
            return internal_id

        response = self.api.place_order(pair, side.name, order_type.name, amount, price)
        
        if not response["error"]:
            exchange_id = response["result"]["txid"][0]
            self.orders[internal_id]["status"] = OrderStatus.OPEN
            self.orders[internal_id]["exchange_id"] = exchange_id
            logging.info(f"Placed order {internal_id} -> Exchange ID {exchange_id}")
        else:
            self.orders[internal_id]["status"] = OrderStatus.ERROR
            logging.error(f"Failed to place order {internal_id}: {response['error']}")
        
        return internal_id

    def cancel_order(self, internal_id):
        if internal_id not in self.orders:
            logging.error(f"Cannot cancel unknown internal ID {internal_id}")
            return False
        
        order = self.orders[internal_id]
        exchange_id = order.get("exchange_id")
        
        if order["status"] not in [OrderStatus.PENDING, OrderStatus.OPEN]:
             logging.warning(f"Order {internal_id} is not in a cancelable state ({order['status']}).")
             return False
        
        if not exchange_id:
            logging.warning(f"Order {internal_id} has no exchange ID, cannot cancel.")
            return False

        if self.dry_run:
            logging.info(f"[Engine DRY RUN] Would cancel order: {order}")
            order["status"] = OrderStatus.CANCELLED
            return True

        response = self.api.cancel_order(exchange_id)
        if not response["error"]:
            order["status"] = OrderStatus.CANCELLED
            logging.info(f"Cancelled order {internal_id} (Exchange ID {exchange_id})")
            return True
        else:
            logging.error(f"Failed to cancel order {internal_id}: {response['error']}")
            return False
