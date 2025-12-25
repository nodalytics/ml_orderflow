import os
import pandas as pd
import numpy as np
from typing import Literal
from trend_analysis.utils.config import settings
from trend_analysis.utils.initializer import logger_instance

logger = logger_instance.get_logger()

def linear_regression_channel(
    df: pd.DataFrame,
    src: str = "close",
    period: int = 100,
    devlen: float = 2.0,
    trend_mode: Literal["avg", "vote", "extreme"] = "avg",
    prev_slopes_len: int = 3,
) -> pd.DataFrame:
    """
    Linear Regression Channels calculation.
    """
    if src == "hlc3":
        prices = (df["high"] + df["low"] + df["close"]) / 3.0
    elif src == "ohlc4":
        prices = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    else:
        prices = df[src]

    values, uppers, lowers = [], [], []
    slopes, intercepts, degrees = [], [], []
    stdevs, r2s, cfos, trends = [], [], [], []

    # slope_history for trend detection
    slope_history = []

    for i in range(len(prices)):
        if i < period:
            values.append(np.nan)
            uppers.append(np.nan)
            lowers.append(np.nan)
            slopes.append(np.nan)
            intercepts.append(np.nan)
            degrees.append(np.nan)
            stdevs.append(np.nan)
            r2s.append(np.nan)
            cfos.append(np.nan)
            trends.append("⇒")
            continue

        y = prices.iloc[i - period : i].values
        x_arr = np.arange(1, period + 1, dtype=np.float64)
        y_arr = y.astype(float)

        x_sum = 0.5 * period * (period + 1)
        x2_sum = x_sum * (2 * period + 1) / 3.0
        divisor = period * x2_sum - x_sum * x_sum
        y_sum = np.sum(y_arr)
        xy_sum = np.sum(x_arr * y_arr)

        slope = (period * xy_sum - x_sum * y_sum) / divisor
        intercept = (y_sum * x2_sum - x_sum * xy_sum) / divisor

        residuals = (slope * x_arr + intercept) - y_arr
        res_sum_sq = np.sum(residuals**2)

        value = residuals[-1] + y_arr[-1]
        degree = 180.0 / np.pi * np.arctan(slope)
        cfo = 100.0 * residuals[-1] / y_arr[-1] if y_arr[-1] != 0 else 0.0
        stdev = np.sqrt(res_sum_sq / period)

        mean_y = np.mean(y_arr)
        sst = np.sum((y_arr - mean_y) ** 2)
        r2 = 1.0 - res_sum_sq / sst if sst != 0 else -np.inf

        upper = value + stdev * devlen
        lower = value - stdev * devlen

        slope_history.append(slope)
        if len(slope_history) > prev_slopes_len:
            slope_history.pop(0)

        if len(slope_history) < prev_slopes_len:
            trend = "⇒"
        else:
            prev_slopes = slope_history[:-1]
            if trend_mode == "avg":
                avg_prev = np.mean(prev_slopes)
                if slope > 0:
                    trend = "⇑" if slope > avg_prev else "⇗"
                elif slope < 0:
                    trend = "⇓" if slope < avg_prev else "⇘"
                else:
                    trend = "⇒"
            elif trend_mode == "vote":
                greater = sum(slope > ps for ps in prev_slopes)
                smaller = sum(slope < ps for ps in prev_slopes)
                if slope > 0:
                    trend = "⇑" if greater > smaller else "⇗"
                elif slope < 0:
                    trend = "⇓" if smaller > greater else "⇘"
                else:
                    trend = "⇒"
            elif trend_mode == "extreme":
                min_slope, max_slope = min(prev_slopes), max(prev_slopes)
                if slope > 0:
                    trend = "⇑" if slope > max_slope else "⇗"
                elif slope < 0:
                    trend = "⇓" if slope < min_slope else "⇘"
                else:
                    trend = "⇒"
            else:
                trend = "⇒"

        values.append(value)
        uppers.append(upper)
        lowers.append(lower)
        slopes.append(slope)
        intercepts.append(intercept)
        degrees.append(degree)
        stdevs.append(stdev)
        r2s.append(r2)
        cfos.append(cfo)
        trends.append(trend)

    df["reg_value"] = values
    df["reg_upper"] = uppers
    df["reg_lower"] = lowers
    df["reg_slope"] = slopes
    df["reg_intercept"] = intercepts
    df["reg_degree"] = degrees
    df["reg_stdev"] = stdevs
    df["reg_r2"] = r2s
    df["reg_cfo"] = cfos
    df["reg_trend"] = trends

    return df

def calculate_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Computes a set of statistical and technical features for a given symbol's DataFrame.
    Includes robustness checks for small dataframes and numeric stability.
    """
    min_bars = params.get('min_bars', 100)
    if len(df) < min_bars:
        return pd.DataFrame(columns=df.columns)

    df = df.sort_index()

    # Returns (with safety for zero Close)
    epsilon = 1e-9
    close_safe = df['close'] + epsilon
    
    df['returns'] = close_safe.pct_change()
    df['log_returns'] = np.log(close_safe / close_safe.shift(1))
    
    # Replace Inf with NaN for stability
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Volatility
    for window in params.get('volatility_windows', [7, 24]):
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

    # Price Action Ratios
    df['hl_ratio'] = (df['high'] - df['low']) / close_safe
    df['co_ratio'] = (df['close'] - df['open']) / (df['open'] + epsilon)
    
    # Indicators
    for window in params.get('sma_windows', [20, 50]):
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
    
    rsi_window = params.get('rsi_window', 14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    # Handle division by zero in RS
    rs = gain / (loss + epsilon)
    df[f'rsi_{rsi_window}'] = 100 - (100 / (1 + rs))

    # Existing Linear Regression Channel
    try:
        lrc_period = params.get('lrc_period', 100)
        df = linear_regression_channel(df, src='hlc3', period=lrc_period)
    except Exception as e:
        logger.warning(f"LRC calculation failed for a segment: {e}")
    
    # Time
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # Target (next return)
    target_window = params.get('target_window', 1)
    df['target_next_return'] = df['returns'].shift(-target_window)
    
    return df

def preprocess_data():
    raw_data_path = settings.params['data_ingestion']['raw_data_path']
    processed_data_path = settings.params['preprocessing']['processed_data_path']
    
    input_file = os.path.join(raw_data_path, "dataset.csv")
    output_file = os.path.join(processed_data_path, "processed_dataset.csv")

    logger.info(f"Preprocessing data from {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)
    
    try:
        df = pd.read_csv(input_file)
        
        # Robustness: Check for required columns
        required_cols = ['time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Dataset is missing required columns: {missing}")
            return

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # -------------------------------------------------------------------------
    # Robustness: Data Validation
    # -------------------------------------------------------------------------
    logger.info(f"Preprocessing data from {input_file}") # Moved here to be after successful load
    validation_config = settings.params.get('validation', {})
    
    # 1. Schema Validation
    if validation_config.get('enforce_schema', True):
        required_cols = validation_config.get('required_columns', ['open', 'high', 'low', 'close', 'volume'])
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            error_msg = f"Data validation failed. Missing required columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info("Schema validation passed.")

    # 2. Anomaly Detection (Z-Score on Returns)
    anomaly_config = validation_config.get('anomaly_protection', {})
    if anomaly_config.get('check_zscore', False):
        threshold = anomaly_config.get('zscore_threshold', 4.0)
        action = anomaly_config.get('action', 'warn')
        
        # Calculate returns for anomaly check if not present (simple close-to-close)
        # We process anomalies per symbol to avoid cross-asset contamination
        logger.info(f"Running anomaly detection (Z-Score > {threshold})...")
        
        def check_anomalies(group):
            # Calculate simple pct change
            returns = group['close'].pct_change()
            # Calculate z-score
            z_scores = (returns - returns.mean()) / returns.std()
            abs_z_scores = z_scores.abs()
            
            anomalies = abs_z_scores > threshold
            if anomalies.any():
                anomaly_indices = anomalies[anomalies].index
                msg = f"Symbol {group.name}: Found {len(anomaly_indices)} anomalies (Z-Score > {threshold})"
                if action == 'drop':
                    logger.warning(f"{msg}. Dropping rows.")
                    return group[~anomalies]
                else:
                    logger.warning(f"{msg}. Proceeding with caution.")
                    return group
            return group

        # Apply anomaly check per symbol
        if 'symbol' in df.columns:
            # group_keys=False to keep original index structure if possible, 
            # though reset_index usually happens. ensuring alignment.
            df = df.groupby('symbol', group_keys=False).apply(check_anomalies)
        else:
            df = check_anomalies(df)

    logger.info("Data validation complete. Proceeding to feature extraction.")

    # -------------------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------------------
    
    try:
        logger.info("Extracting features for each symbol and timeframe...")
        # Use simple progress indication via logging if dataset is large, 
        # but here we rely on standard logs.
        
        # NOTE: Deprecation warning for groupby.apply in pandas 2.2+ is suppressed or handled
        # by explicit include_groups=False if supported, or accepting it.
        # We ensure 'symbol' and 'timeframe' are available in the applied function or index.
        
        processed_df = df.groupby(['symbol', 'timeframe'], group_keys=False).apply(
            lambda x: calculate_features(x, settings.params['preprocessing']), 
            include_groups=True
        )
        
        if processed_df.empty:
            logger.warning("No data remains after feature extraction. Possible cause: symbols have < 100 bars.")
            return

        if processed_df.empty:
            logger.warning("No data remains after feature extraction. Possible cause: symbols have < 100 bars.")
            return

        # Explicitly handle any remaining Inf values
        processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Multivariate Merging Logic
        target_tf = settings.params['preprocessing'].get('target_timeframe')
        context_tfs = settings.params['preprocessing'].get('context_timeframes', [])
        
        if target_tf:
            logger.info(f"Target timeframe detected: {target_tf}. Checking for multivariate contexts...")
            
            # Reset index to make 'time' a column for merging
            processed_df = processed_df.reset_index()
            
            # Separate target and context dataframes
            target_df = processed_df[processed_df['timeframe'] == target_tf].copy()
            
            # Apply Target Symbol Filtering
            target_sym = settings.params['preprocessing'].get('target_symbol')
            if target_sym:
                logger.info(f"Filtering target dataset for symbol: {target_sym}")
                target_df = target_df[target_df['symbol'] == target_sym]
            
            if target_df.empty:
                logger.error(f"No data found for target timeframe {target_tf} (and symbol {target_sym} if specified)")
                return

            final_dfs = []
            
            # Unified Multivariate Merging
            enable_multivariate = settings.params['preprocessing'].get('enable_multivariate', False)
            multivariate_config = settings.params['preprocessing'].get('multivariate_config', {})
            
            # Determine Target Settings
            if enable_multivariate and multivariate_config:
                # In multivariate mode, targets MUST be defined in the config
                target_tf = multivariate_config.get('target_timeframe')
                target_sym = multivariate_config.get('target_symbol')
                context_config = multivariate_config.get('context', {})
                
                if not target_tf or not target_sym:
                    logger.error("Multivariate mode enabled but 'target_timeframe' or 'target_symbol' missing in 'multivariate_config'")
                    return
                    
                logger.info(f"Multivariate Mode: Target={target_sym} {target_tf}, Contexts={list(context_config.keys())}")
            else:
                # Legacy/Batch Mode
                target_tf = settings.params['preprocessing'].get('target_timeframe')
                target_sym = settings.params['preprocessing'].get('target_symbol') # Optional in batch mode
                context_config = {}
                logger.info(f"Batch Mode: Target Timeframe={target_tf}, Target Symbol Filter={target_sym}")

            if not target_tf:
                logger.error("Target timeframe not specified.")
                return

            # Separate target and context dataframes
            target_df = processed_df[processed_df['timeframe'] == target_tf].copy()
            
            # Apply Target Symbol Filtering
            if target_sym:
                logger.info(f"Filtering target dataset for symbol: {target_sym}")
                target_df = target_df[target_df['symbol'] == target_sym]
            
            if target_df.empty:
                logger.error(f"No data found for target timeframe {target_tf} (and symbol {target_sym} if specified)")
                return

            final_dfs = []
            
            if enable_multivariate and context_config:
                
                # Pre-calculate time deltas map
                tf_map = {
                    "M1": pd.Timedelta(minutes=1),
                    "M5": pd.Timedelta(minutes=5),
                    "M15": pd.Timedelta(minutes=15),
                    "M30": pd.Timedelta(minutes=30),
                    "H1": pd.Timedelta(hours=1),
                    "H4": pd.Timedelta(hours=4),
                    "D1": pd.Timedelta(days=1),
                    "W1": pd.Timedelta(weeks=1),
                    "1h": pd.Timedelta(hours=1),
                    "4h": pd.Timedelta(hours=4),
                    "1d": pd.Timedelta(days=1),
                    "1w": pd.Timedelta(weeks=1)
                }

                # Group by symbol to perform merging per symbol
                for symbol, group_target in target_df.groupby('symbol'):
                    symbol_df = group_target.sort_values('time')
                    
                    # Iterate through the configuration: Context Symbol -> List[Context Timeframes]
                    for ctx_sym, ctx_tfs in context_config.items():
                        if not isinstance(ctx_tfs, list):
                            ctx_tfs = [ctx_tfs]
                            
                        for ctx_tf in ctx_tfs:
                            # Skip if we are trying to merge the exact same series onto itself (Identity)
                            if ctx_sym == symbol and ctx_tf == target_tf:
                                continue

                            # Fetch Context Data
                            ctx_source = processed_df[(processed_df['symbol'] == ctx_sym) & (processed_df['timeframe'] == ctx_tf)].copy()
                            
                            if ctx_source.empty:
                                logger.warning(f"No context data found for {ctx_sym} {ctx_tf}")
                                continue
                                
                            ctx_source = ctx_source.sort_values('time')

                            # Exclude target column from context to prevent leakage
                            target_col = settings.params['train'].get('target_col', 'target_next_return')
                            if target_col in ctx_source.columns:
                                ctx_source = ctx_source.drop(columns=[target_col])

                            # Determine Merge Key and Rename Suffix
                            suffix = ""
                            if ctx_sym == symbol:
                                suffix = f"_{ctx_tf}" # E.g. _H4
                            elif ctx_tf == target_tf:
                                suffix = f"_{ctx_sym}" # E.g. _BTCUSD
                            else:
                                suffix = f"_{ctx_sym}_{ctx_tf}" # E.g. _BTCUSD_H4
                            
                            # Rename columns
                            cols_to_use = [c for c in ctx_source.columns if c not in ['time', 'symbol', 'timeframe', 'close_time']]
                            rename_map = {c: f"{c}{suffix}" for c in cols_to_use}
                            
                            # Lookahead Prevention Logic
                            if ctx_tf != target_tf:
                                # Different TF: Use 'close_time' matching logic
                                delta = tf_map.get(ctx_tf)
                                if not delta:
                                    logger.warning(f"Unknown duration for {ctx_tf}, skipping.")
                                    continue
                                    
                                ctx_source['close_time'] = ctx_source['time'] + delta
                                ctx_renamed = ctx_source[['close_time'] + cols_to_use].rename(columns=rename_map)
                                
                                symbol_df = pd.merge_asof(
                                    symbol_df,
                                    ctx_renamed,
                                    left_on='time',
                                    right_on='close_time',
                                    direction='backward'
                                )
                                if 'close_time' in symbol_df.columns:
                                    symbol_df.drop(columns=['close_time'], inplace=True)
                            else:
                                # Same TF: Direct merge on 'time'
                                ctx_renamed = ctx_source[['time'] + cols_to_use].rename(columns=rename_map)
                                symbol_df = pd.merge_asof(
                                    symbol_df,
                                    ctx_renamed,
                                    on='time',
                                    direction='backward'
                                )

                    final_dfs.append(symbol_df)
            else:
                # If disabled, just append as is
                for symbol, group_target in target_df.groupby('symbol'):
                    final_dfs.append(group_target)
            
            if final_dfs:
                processed_df = pd.concat(final_dfs)
                # Set index back to time
                processed_df.set_index('time', inplace=True)
                logger.info(f"Multivariate merging complete. Shape: {processed_df.shape}")
            else:
                logger.error("Multivariate merging resulted in empty dataframe.")
                return

        # Drop initial NaN rows created by rolling windows/pct_change (and now merging)
        before_drop = len(processed_df)
        # critical_cols = ['returns', 'rsi_14', 'reg_slope'] 
        # For multivariate, we might have NaNs at the start of H1 that don't match any H4 yet
        # Just use dropna generic or keep it strict
        processed_df.dropna(inplace=True) # Being strict to ensure quality data
        
        logger.info(f"Dropped {before_drop - len(processed_df)} rows containing NaNs/Infs.")

        if not processed_df.empty:
            processed_df.to_csv(output_file)
            logger.info(f"Successfully saved {len(processed_df)} processed samples to {output_file}")
        else:
            logger.error("All rows dropped during validation. Nothing to save.")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    preprocess_data()
