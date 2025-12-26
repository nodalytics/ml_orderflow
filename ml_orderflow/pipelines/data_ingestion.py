import os
import pandas as pd
import numpy as np
import logging
import time
from functools import wraps
from datetime import datetime
from typing import List, Optional, Callable
import ccxt

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from ml_orderflow.utils.config import settings
from ml_orderflow.utils.initializer import logger_instance
from ml_orderflow.core.circuit_breaker import CircuitBreaker, CircuitBreakerError
 
logger = logger_instance.get_logger()

# Initialize circuit breakers for external services
robustness_config = settings.params.get('robustness', {})
cb_config = robustness_config.get('circuit_breaker', {})

ccxt_circuit_breaker = CircuitBreaker(
    failure_threshold=cb_config.get('failure_threshold', 5),
    timeout_seconds=cb_config.get('timeout_seconds', 60),
    half_open_max_calls=cb_config.get('half_open_max_calls', 3),
    name="ccxt_exchange"
)

mt5_circuit_breaker = CircuitBreaker(
    failure_threshold=cb_config.get('failure_threshold', 5),
    timeout_seconds=cb_config.get('timeout_seconds', 60),
    half_open_max_calls=cb_config.get('half_open_max_calls', 3),
    name="mt5_terminal"
)


def retry(exceptions: tuple, tries: int = 3, delay: int = 2, backoff: int = 2):
    """
    Retry decorator for functions that may fail due to transient issues.
    """
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retrying {f.__name__} in {_delay}s due to: {e} ({_tries-1} tries left)")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return f(*args, **kwargs)
        return wrapper
    return decorator


def validate_data(df: pd.DataFrame, source: str) -> bool:
    """
    Validates the fetched data.
    """
    if df is None or df.empty:
        logger.error(f"Validation failed: No data fetched from {source}")
        return False
    
    # Check for minimum number of bars
    if len(df) < 2:
        logger.error(f"Validation failed: Dataframe from {source} is too small ({len(df)} bars)")
        return False
        
    # Check for NaN values in critical columns
    critical_cols = ['open', 'high', 'low', 'close']
    if all(col in df.columns for col in critical_cols):
        nan_counts = df[critical_cols].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values detected in critical columns from {source}:\n{nan_counts[nan_counts > 0]}")
            # Depending on strictness, could return False if NaN > threshold
            
    logger.info(f"Data from {source} validated successfully ({len(df)} bars).")
    return True


def connect_mt5() -> bool:
    """
    Initializes the MetaTrader 5 connection.
    Returns True if successful, False otherwise.
    """
    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 module is not installed.")
        return False
    
    if not mt5.initialize():
        logger.error(f"mt5.initialize() failed, error code = {mt5.last_error()}")
        return False
    
    return True


@retry((ccxt.NetworkError, ccxt.ExchangeError, ccxt.RateLimitExceeded), tries=3, delay=2)
def get_data_ccxt(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch historical data using CCXT with circuit breaker protection.
    """
    def _fetch():
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            # Ensure only standardized columns are returned
            return df[['open', 'high', 'low', 'close', 'volume']]
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RateLimitExceeded) as e:
            # Re-raise for retry decorator
            raise e
        except Exception as e:
            logger.error(f"Unexpected error fetching data from CCXT ({exchange_id}): {e}")
            raise RuntimeError(f"Could not retrieve data from {exchange_id}")
    
    try:
        return ccxt_circuit_breaker.call(_fetch)
    except CircuitBreakerError as e:
        logger.error(f"Circuit breaker open for CCXT: {e}")
        return pd.DataFrame()  # Return empty DataFrame for graceful degradation




@retry((ValueError, RuntimeError), tries=3, delay=1)
def get_data_mt5(symbol: str, n_bars: int, timeframe, start_pos=None) -> pd.DataFrame:
    """
    Fetch historical data from MetaTrader 5.
    
    - `symbol`: Trading instrument (e.g., "BTCUSD").
    - `n_bars`: Number of bars to retrieve.
    - `timeframe`: MT5 timeframe (e.g., H1).
    - `start_pos`: Offset from the most recent bar (default `None` for live trading).
    
    If `start_pos` is `None`, fetches the latest `n_bars` (useful for live trading).
    If `start_pos` is given, fetches `n_bars` from that historical position (useful for backtesting).
    """
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 module is not installed. Data cannot be fetched from MT5.")

    if not mt5.terminal_info() and not connect_mt5():
        raise RuntimeError("Failed to connect to MetaTrader 5 terminal.")

    if start_pos is None:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)  # Latest n_bars for live trading
    else:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, n_bars)  # Historical data for backtesting

    if rates is None:
        raise ValueError(f"Could not retrieve data for {symbol}. Error: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Standardize headers: Rename tick_volume to volume
    if 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    # Ensure only standardized columns are returned
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_exchange_symbols(group: Optional[str] = None) -> List[str]:
    """
    Retrieves a list of all available symbol names from the MetaTrader 5 terminal.

    An active connection to MT5 is required before calling this function.
    Use `connect_mt5()` to establish a connection.

    :param group: Optional string to filter symbols. It's a comma-separated list
                  of masks. '*' means all symbols. '!EUR' excludes symbols starting with EUR.
                  Example: `group="*,!EUR*,!USD*"`
    :return: A list of symbol names (strings).
    :raises RuntimeError: If the MT5 connection is not initialized or symbols cannot be retrieved.
    """
    if not MT5_AVAILABLE:
        logger.warning("MetaTrader5 module is not installed. Returning empty symbol list.")
        return []

    if not mt5.terminal_info() and not connect_mt5():
        raise RuntimeError("MetaTrader 5 is not initialized. Please connect first using connect_mt5().")

    symbols = mt5.symbols_get(group)
    if symbols is None:
        logger.error(f"symbols_get() failed, error code = {mt5.last_error()}")
        # Depending on desired strictness, you could return [] or raise an error.
        raise RuntimeError("Failed to retrieve symbols from MetaTrader 5.")

    return [s.name for s in symbols]


def ingest_data():
    ingestion_params = settings.params['data_ingestion']
    raw_data_path = ingestion_params['raw_data_path']
    source = ingestion_params.get('source', 'mt5')
    mt5_initialized = False
    
    logger.info(f"Starting multi-symbol data ingestion from {source} to {raw_data_path}")
    
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path, exist_ok=True)

    all_data = []
    failed_symbols = []  # Track failed symbols for graceful degradation
    try:
        if source == 'mt5':
            if MT5_AVAILABLE:
                if not mt5.terminal_info() and not connect_mt5():
                    raise RuntimeError("Failed to connect to MetaTrader 5 terminal.")
                mt5_initialized = True
                
                mt5_params = ingestion_params.get('mt5', {})
                symbols = mt5_params.get('symbols', ['BTCUSD'])
                n_bars = mt5_params.get('n_bars', 100)
                # Support list of timeframes or single timeframe for backward compatibility
                timeframes_conf = mt5_params.get('timeframes', mt5_params.get('timeframe', ['H1']))
                if isinstance(timeframes_conf, str):
                    timeframes_conf = [timeframes_conf]
                
                for symbol in symbols:
                    for tf_str in timeframes_conf:
                        logger.info(f"Fetching data for {symbol} ({tf_str}) from MT5...")
                        try:
                            # Safely eval the timeframe constant
                            # Assuming tf_str is like 'H1', 'M15', etc.
                            # Verify if it exists in mt5 module
                            if not hasattr(mt5, f"TIMEFRAME_{tf_str}"):
                                logger.error(f"Invalid MT5 timeframe: {tf_str}")
                                failed_symbols.append(f"{symbol}_{tf_str}")
                                continue
                                
                            timeframe = getattr(mt5, f"TIMEFRAME_{tf_str}")
                            df = get_data_mt5(symbol, n_bars, timeframe)
                            
                            if df.empty:
                                logger.warning(f"No data retrieved for {symbol} ({tf_str})")
                                failed_symbols.append(f"{symbol}_{tf_str}")
                                continue
                            
                            df['symbol'] = symbol
                            df['timeframe'] = tf_str
                            all_data.append(df)
                            logger.info(f"Successfully fetched {len(df)} bars for {symbol} ({tf_str})")
                        except Exception as e:
                            logger.error(f"Failed to fetch {symbol} ({tf_str}) from MT5: {e}")
                            failed_symbols.append(f"{symbol}_{tf_str}")
            else:
                logger.error("MetaTrader 5 module is NOT available. Ingestion failed.")
                return
        elif hasattr(ccxt, source):
            exchange_params = ingestion_params.get(source, {})
            symbols = exchange_params.get('symbols', ['BTC/USDT'])
            
            # Support list or single
            timeframes_conf = exchange_params.get('timeframes', exchange_params.get('timeframe', ['1h']))
            if isinstance(timeframes_conf, str):
                timeframes_conf = [timeframes_conf]
                
            limit = exchange_params.get('limit', 100)
            
            for symbol in symbols:
                for tf_str in timeframes_conf:
                    logger.info(f"Fetching data for {symbol} ({tf_str}) from {source.upper()}...")
                    try:
                        df = get_data_ccxt(source, symbol, tf_str, limit)
                        
                        if df.empty:
                            logger.warning(f"No data retrieved for {symbol} ({tf_str})")
                            failed_symbols.append(f"{symbol}_{tf_str}")
                            continue
                        
                        df['symbol'] = symbol
                        df['timeframe'] = tf_str
                        all_data.append(df)
                        logger.info(f"Successfully fetched {len(df)} bars for {symbol} ({tf_str})")
                    except Exception as e:
                        logger.error(f"Failed to fetch {symbol} ({tf_str}) from {source.UPPER()}: {e}")
                        failed_symbols.append(f"{symbol}_{tf_str}")
        else:
            logger.error(f"Unknown data source: {source}")
            return

        # Graceful degradation: Continue with available data even if some symbols failed
        if failed_symbols:
            logger.warning(
                f"Failed to fetch {len(failed_symbols)} symbol/timeframe combinations: {failed_symbols}. "
                f"Continuing with {len(all_data)} successful fetches."
            )

        # Consolidate, validate, and save
        if all_data:
            master_df = pd.concat(all_data)
            
            # Reorder columns to have 'symbol' and 'timeframe' as first columns
            cols = ['symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
            master_df = master_df[cols]

            if validate_data(master_df, source):
                output_file = os.path.join(raw_data_path, "dataset.csv")
                master_df.to_csv(output_file)
                logger.info(
                    f"Successfully ingested data for {len(all_data)} symbol/timeframe combinations. "
                    f"Saved to {output_file}"
                )
            else:
                logger.error("Consolidated data failed validation. Not saving.")
        else:
            logger.error("No data fetched for any symbol. All fetches failed.")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
    finally:
        if mt5_initialized:
            mt5.shutdown()
            logger.info("MetaTrader 5 connection closed.")

if __name__ == "__main__":
    ingest_data()
