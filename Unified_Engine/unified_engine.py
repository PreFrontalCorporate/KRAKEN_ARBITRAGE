import sys
import os
import time
import numpy as np
import pandas as pd
import krakenex
from pykrakenapi import KrakenAPI
from pathlib import Path
from dotenv import load_dotenv
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import pytz # Import the new timezone library

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# --- Advanced Logging Configuration ---
# 1. Create a timezone-aware formatter
class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=pytz.timezone('America/Los_Angeles')):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s

# 2. Configure the logger with the new formatter and handler
log_formatter = TimezoneFormatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S %Z')
log_handler = TimedRotatingFileHandler(LOGS_DIR / "engine.log", when="midnight", interval=1, backupCount=30)
log_handler.setFormatter(log_formatter)
log_handler.suffix = "%Y-%m-%d" # This will append the date to the rotated log file

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- Module Imports ---
from Execution_Engine_Skeleton.execution_engine import OrderSide, OrderType
from Risk_Sizing_Kelly_Tail_Risk.kelly_optimization import KellySizer

# --- Live Kraken Execution Engine ---
class LiveExecutionEngine:
    def __init__(self, api_key, private_key, dry_run=True):
        self.dry_run = dry_run
        try:
            self.api = krakenex.API(api_key, private_key)
            self.kraken = KrakenAPI(self.api)
            logging.info(f"✅ Kraken API connection successful. Dry Run: {self.dry_run}")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Kraken API: {e}")
            self.kraken = None

    def get_account_balance(self):
        # ... (function is unchanged)
        if not self.kraken: return {}
        if self.dry_run:
            return {'ZUSD': 2.26, 'PROMPT': 21.993}
        try:
            balance = self.kraken.get_account_balance()
            return {asset: qty for asset, qty in balance['vol'].items() if qty > 1e-8}
        except Exception as e:
            logging.error(f"Could not get account balance: {e}")
            return {}

    def get_ticker_info(self, pair_csv):
        # ... (function is unchanged)
        if not self.kraken: return None
        try:
            return self.kraken.get_ticker_information(pair_csv)
        except Exception as e: return None

    def get_tradable_asset_pairs(self, top_n_by_volume=50):
        # ... (function is unchanged)
        if not self.kraken: return ['XBTUSD', 'ETHUSD', 'SOLUSD']
        try:
            all_pairs = self.kraken.get_tradable_asset_pairs()
            ignore_list = ['DAI', 'USDT', 'USDC', 'USDG', 'ZEUR', 'ZGBP', 'AUD']
            ignore_pattern = '|'.join(ignore_list)
            usd_pairs = all_pairs[all_pairs.index.str.upper().str.endswith('USD')]
            usd_pairs = usd_pairs[~usd_pairs.index.str.upper().str.contains(ignore_pattern)]
            pair_list = list(usd_pairs.index)
            ticker_info = self.kraken.get_ticker_information(','.join(pair_list))
            volume_data = []
            for pair, data in ticker_info.iterrows():
                volume_in_quote_currency = float(data['v'][1]) * float(data['p'][1])
                volume_data.append({'pair': pair, 'volume_usd': volume_in_quote_currency})
            volume_df = pd.DataFrame(volume_data).set_index('pair')
            top_pairs = volume_df.sort_values(by='volume_usd', ascending=False).head(top_n_by_volume)
            return list(top_pairs.index)
        except Exception as e:
            logging.error(f"Could not dynamically fetch asset pairs: {e}", exc_info=True)
            return ['XBTUSD', 'ETHUSD', 'SOLUSD']

    def get_historical_data(self, pair, interval, since):
        # ... (function is unchanged)
        if not self.kraken: return None
        try:
            ohlc, _ = self.kraken.get_ohlc_data(pair, interval, since)
            return ohlc
        except Exception as e: return None

    def place_order(self, pair, side: OrderSide, order_type: OrderType, amount, price=None):
        # ... (function is unchanged)
        logging.info(f"--- Placing Order ---")
        logging.info(f"  Pair: {pair}, Side: {side.name}, Type: {order_type.name}, Amount: {amount:.8f}")
        if self.dry_run:
            logging.warning(f"[DRY RUN] Order would be placed.")
            return f"DRYRUN_{int(time.time())}"
        try:
            response = self.kraken.add_standard_order(
                pair=pair, type=side.name.lower(), ordertype=order_type.name.lower(),
                volume=f"{amount:.8f}", validate=False
            )
            txid = response['txid'][0]
            logging.info(f"✅✅✅ LIVE ORDER PLACED: {txid} ✅✅✅")
            return txid
        except Exception as e:
            logging.error(f"❌❌❌ LIVE ORDER FAILED: {e} ❌❌❌")
            return None

# --- Main Engine ---
class UnifiedEngine:
    def __init__(self, mode='live'):
        self.mode = mode
        self.state = "RUNNING"
        self._load_env()

        self.execution_engine = LiveExecutionEngine(
            self.kraken_api_key, self.kraken_private_key, dry_run=(self.mode != 'live')
        )
        self.risk_sizer = KellySizer()
        self.trading_pairs = self.execution_engine.get_tradable_asset_pairs()
        self.portfolio = {}
        logging.info(f"⚙️ Engine initialized in {self.mode.upper()} mode. Monitoring {len(self.trading_pairs)} pairs.")

    def _load_env(self):
        # ... (function is unchanged)
        env_path = PROJECT_ROOT / '.agent' / 'agent.env'
        load_dotenv(dotenv_path=env_path)
        self.kraken_api_key = os.getenv("KRAKEN_API_KEY")
        self.kraken_private_key = os.getenv("KRAKEN_PRIVATE_KEY")

    def _calculate_rsi(self, series, period=14):
        # ... (function is unchanged)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        if loss.empty or loss.iloc[-1] == 0: return 100
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _update_portfolio_value(self):
        # ... (function is unchanged)
        self.portfolio = {'total_value_usd': 0, 'cash_usd': 0, 'positions': {}}
        balance = self.execution_engine.get_account_balance()
        if not balance:
            logging.error("Could not retrieve account balance.")
            return

        pairs_to_query = []
        for asset, qty in balance.items():
            asset_clean = asset.replace('X', '').replace('Z', '')
            if 'USD' in asset_clean:
                self.portfolio['cash_usd'] += qty
            elif qty > 0:
                pair = asset_clean + 'USD'
                if pair in self.trading_pairs:
                    pairs_to_query.append(pair)

        if pairs_to_query:
            tickers = self.execution_engine.get_ticker_info(','.join(pairs_to_query))
            if tickers is not None:
                for pair in pairs_to_query:
                    asset_name = pair.replace('USD', '')
                    original_asset = next((a for a in balance.keys() if asset_name in a.replace('X','').replace('Z','')), None)
                    if original_asset:
                        qty = balance[original_asset]
                        price = float(tickers.loc[pair]['c'][0])
                        value = qty * price
                        if pair not in self.portfolio.get('positions', {}):
                            self.portfolio['positions'][pair] = {'qty': qty, 'value': value, 'entry_price': price}
                        else:
                            self.portfolio['positions'][pair].update({'qty': qty, 'value': value, 'price': price})

        self.portfolio['total_value_usd'] = sum(pos['value'] for pos in self.portfolio.get('positions', {}).values()) + self.portfolio['cash_usd']

    def run_live_trading(self, rsi_period=14, oversold=30, overbought=65, min_trade_usd=5.0):
        # ... (function is unchanged)
        logging.info(f"\n--- ✅ Live Portfolio Manager Started ---")

        while True:
            try:
                self._update_portfolio_value()
                logging.info(f"[State] Total Portfolio Value: ${self.portfolio['total_value_usd']:.2f} | Cash: ${self.portfolio['cash_usd']:.2f}")

                all_opportunities = []
                logging.info("--- Starting market analysis cycle ---")
                for pair in self.trading_pairs:
                    since = int(time.time()) - (rsi_period + 100) * 900
                    data = self.execution_engine.get_historical_data(pair, 15, since)
                    if data is None or data.empty or len(data) < rsi_period:
                        time.sleep(3)
                        continue

                    data['rsi'] = self._calculate_rsi(data['close'], rsi_period)
                    current_rsi = data['rsi'].iloc[-1]
                    logging.info(f"  > Analyzed {pair}: RSI = {current_rsi:.2f}")
                    all_opportunities.append({'pair': pair, 'rsi': current_rsi})
                    time.sleep(3)

                assets_to_sell = []
                current_positions = list(self.portfolio.get('positions', {}).keys())

                for pair in current_positions:
                    current_asset_info = next((item for item in all_opportunities if item["pair"] == pair), None)
                    if current_asset_info and current_asset_info['rsi'] > overbought:
                         assets_to_sell.append({'pair': pair, 'qty': self.portfolio['positions'][pair]['qty']})

                for asset in assets_to_sell:
                    logging.warning(f"  [REBALANCE] SELL SIGNAL for {asset['pair']}. Exiting position.")
                    self.execution_engine.place_order(asset['pair'], OrderSide.SELL, OrderType.MARKET, asset['qty'])
                    time.sleep(5)

                if assets_to_sell: self._update_portfolio_value()
                cash_to_deploy = self.portfolio['cash_usd']

                buy_candidates = [opp for opp in all_opportunities if opp['rsi'] < oversold and opp['pair'] not in self.portfolio.get('positions', {})]
                buy_candidates.sort(key=lambda x: x['rsi'])

                for candidate in buy_candidates:
                    if cash_to_deploy < min_trade_usd: break
                    pair = candidate['pair']
                    price_ticker = self.execution_engine.get_ticker_info(pair)
                    if price_ticker is None: continue
                    price = float(price_ticker['c'][0][0])
                    trade_capital = cash_to_deploy * 0.98
                    if trade_capital >= min_trade_usd:
                        logging.warning(f"  [REBALANCE] BUY SIGNAL for {pair}. Allocating up to ${trade_capital:.2f}.")
                        trade_size = trade_capital / price
                        if self.execution_engine.place_order(pair, OrderSide.BUY, OrderType.MARKET, trade_size):
                            cash_to_deploy -= trade_capital
                    else:
                        break

                logging.info("\n--- Cycle complete. Sleeping for 10 minutes. ---")
                time.sleep(10 * 60)

            except KeyboardInterrupt:
                logging.info("\n--- Engine shutdown requested ---")
                break
            except Exception as e:
                logging.error(f"An error occurred in the main loop: {e}", exc_info=True)
                time.sleep(60)

if __name__ == "__main__":
    engine = UnifiedEngine(mode='live')
    engine.run_live_trading()
