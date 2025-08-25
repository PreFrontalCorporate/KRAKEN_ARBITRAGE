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

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='unified_engine.log', filemode='w')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# --- Live Kraken Data Connection ---
class KrakenConnection:
    def __init__(self, api_key, private_key):
        try:
            self.api = krakenex.API(api_key, private_key)
            self.kraken = KrakenAPI(self.api)
            logging.info("✅ Kraken API connection successful.")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Kraken API: {e}")
            self.kraken = None

    def get_tradable_asset_pairs(self, top_n_by_volume=50):
        """Fetches all tradable asset pairs and filters for the top N by 24h volume."""
        if not self.kraken: return []
        try:
            logging.info("Fetching all tradable pairs from Kraken...")
            all_pairs = self.kraken.get_tradable_asset_pairs()

            # Filter for pairs ending in USD, excluding leveraged tokens and stablecoins
            usd_pairs = all_pairs[all_pairs.index.str.upper().str.endswith('USD')]
            usd_pairs = usd_pairs[~usd_pairs.index.str.upper().str.contains('DAI|USDT|USDC')]
            usd_pairs = usd_pairs[~usd_pairs.index.str.contains(r'\d[LPS]$')] # Regex for 2L, 3S etc.

            pair_list = list(usd_pairs.index)
            logging.info(f"Found {len(pair_list)} USD pairs. Fetching ticker data for volume...")

            # Fetch ticker info in batches to get volume
            ticker_info = self.kraken.get_ticker_information(','.join(pair_list))

            volume_data = []
            for pair, data in ticker_info.iterrows():
                volume_in_quote_currency = float(data['v'][1]) * float(data['p'][1])
                volume_data.append({'pair': pair, 'volume_usd': volume_in_quote_currency})

            volume_df = pd.DataFrame(volume_data).set_index('pair')

            # Sort by volume and return the top N
            top_pairs = volume_df.sort_values(by='volume_usd', ascending=False).head(top_n_by_volume)
            logging.info(f"Dynamically selected Top {top_n_by_volume} pairs by 24h volume.")
            return list(top_pairs.index)

        except Exception as e:
            logging.error(f"❌ Could not dynamically fetch asset pairs: {e}", exc_info=True)
            return ['XBTUSD', 'ETHUSD', 'SOLUSD'] # Fallback list

    def get_historical_data(self, pair, interval, since):
        if not self.kraken: return None
        try:
            ohlc, _ = self.kraken.get_ohlc_data(pair, interval, since)
            return ohlc
        except Exception as e:
            logging.error(f"❌ Could not fetch historical data for {pair}: {e}")
            return None

# --- Main Engine ---
class UnifiedEngine:
    def __init__(self):
        logging.info("🚀 Initializing Dynamic All-Coins Engine...")
        self._load_env()
        self.connection = KrakenConnection(self.kraken_api_key, self.kraken_private_key)
        # THIS IS THE CHANGE: It now requests the Top 50
        self.trading_pairs = self.connection.get_tradable_asset_pairs(top_n_by_volume=50)
        if not self.trading_pairs:
            logging.error("❌ FATAL: Could not retrieve any tradable pairs. Exiting.")
            sys.exit(1)

        self.positions = {pair: 0 for pair in self.trading_pairs}
        logging.info(f"⚙️ Engine initialized. Monitoring {len(self.trading_pairs)} pairs: {self.trading_pairs}")

    def _load_env(self):
        env_path = PROJECT_ROOT / '.agent' / 'agent.env'
        load_dotenv(dotenv_path=env_path)
        self.kraken_api_key = os.getenv("KRAKEN_API_KEY")
        self.kraken_private_key = os.getenv("KRAKEN_PRIVATE_KEY")

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        if loss.iloc[-1] == 0: return 100
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def run_weekend_dry_run(self, rsi_period=14, overbought=70, oversold=30):
        logging.info(f"\n--- 🌵 Starting Weekend Dry Run for TOP 50 COINS ---")
        logging.warning("This will run indefinitely. To stop, use 'kill <PID>'.")

        while True:
            try:
                for pair in self.trading_pairs:
                    logging.info(f"\n--- Analyzing {pair} ---")

                    since = int(time.time()) - (rsi_period + 100) * 900
                    data = self.connection.get_historical_data(pair, 15, since)
                    if data is None or data.empty or len(data) < rsi_period:
                        logging.warning(f"  [Data] Insufficient data for {pair}. Skipping.")
                        time.sleep(3)
                        continue

                    data['rsi'] = self._calculate_rsi(data['close'], rsi_period)
                    current_rsi = data['rsi'].iloc[-1]
                    current_price = data['close'].iloc[-1]

                    logging.info(f"  [State] Current Price: ${current_price:,.4f}")
                    logging.info(f"  [Signal] Current RSI({rsi_period}): {current_rsi:.2f}")

                    if current_rsi < oversold and self.positions[pair] == 0:
                        logging.warning(f"  [DRY RUN SIGNAL] BUY (Oversold) DETECTED for {pair} at ${current_price:,.4f}")
                        self.positions[pair] = 1

                    elif current_rsi > overbought and self.positions[pair] == 1:
                        logging.warning(f"  [DRY RUN SIGNAL] SELL (Overbought) DETECTED for {pair} at ${current_price:,.4f}")
                        self.positions[pair] = 0
                    else:
                        logging.info(f"  [Signal] No signal. RSI is neutral.")

                    time.sleep(3) # Short sleep to respect API rate limits

                logging.info("\n--- Cycle complete. Sleeping for 5 minutes. ---")
                time.sleep(5 * 60)

            except KeyboardInterrupt:
                logging.info("\n--- Dry run stopped ---")
                break
            except Exception as e:
                logging.error(f"An error occurred in the main loop: {e}", exc_info=True)
                time.sleep(60)

if __name__ == "__main__":
    engine = UnifiedEngine()
    engine.run_weekend_dry_run()
