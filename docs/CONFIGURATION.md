# Configuration Guide

Complete reference for `params.yaml` configuration options.

## Table of Contents
1. [Base Configuration](#base-configuration)
2. [Data Ingestion](#data-ingestion)
3. [Preprocessing](#preprocessing)
4. [Training](#training)
5. [Validation](#validation)
6. [LLM Configuration](#llm-configuration)
7. [Robustness](#robustness)
8. [MLflow](#mlflow)

---

## Base Configuration

```yaml
base:
  project: ml_orderflow
  data_dir: data
  model_dir: models
  log_dir: logs
```

---

## Data Ingestion

### Source Selection
```yaml
data_ingestion:
  source: "mt5"  # Options: mt5, mexc, binance, kraken, etc.
  raw_data_path: data/raw
```

### MetaTrader 5
```yaml
  mt5:
    symbols: ["BTCUSD", "ETHUSD", "XAUUSD", "GBPUSD"]
    timeframes: ["H4", "W1"]  # MT5 format: M1, M5, M15, H1, H4, D1, W1
    n_bars: 1000
```

### CCXT Exchanges
```yaml
  mexc:  # or binance, kraken, etc.
    symbols: ["BTC/USDT", "ETH/USDT"]
    timeframes: ["1h", "1d"]  # CCXT format: 1m, 5m, 15m, 1h, 4h, 1d
    limit: 1000
```

---

## Preprocessing

### Basic Settings
```yaml
preprocessing:
  processed_data_path: data/processed
  min_bars: 150  # Minimum bars required for feature calculation
```

### Target Configuration ⭐ NEW

**Flexible target generation** - predict any column at any horizon:

```yaml
  # Target Configuration
  target_column: "next_return"      # Name of target column to create
  target_source: "returns"          # Source column to shift
  target_window: 1                  # Periods ahead to predict
```

**Examples**:

1. **Predict next return** (default):
   ```yaml
   target_column: "next_return"
   target_source: "returns"
   target_window: 1
   ```

2. **Predict close price 3 steps ahead**:
   ```yaml
   target_column: "close_3_ahead"
   target_source: "close"
   target_window: 3
   ```

3. **Predict next volume**:
   ```yaml
   target_column: "next_volume"
   target_source: "volume"
   target_window: 1
   ```

4. **Predict log returns 5 steps ahead**:
   ```yaml
   target_column: "log_return_5"
   target_source: "log_returns"
   target_window: 5
   ```

### Technical Indicators

```yaml
  # Volatility
  volatility_windows: [7, 24]
  
  # Moving Averages
  sma_windows: [20, 50]
  
  # RSI
  rsi_window: 14
  
  # Linear Regression Channels
  lrc_period: 100
```

### Liquidity Sentiment Profile ⭐ NEW

```yaml
  # Liquidity Sentiment Profile
  lsp_window: 100       # Lookback window
  lsp_bins: 20          # Number of price bins
```

**Generates**:
- `lsp_poc_price` - Point of Control (highest volume price)
- `lsp_high_value_low/high` - High liquidity zone bounds
- `lsp_low_value_low/high` - Low liquidity zone bounds

### Candlestick Patterns ⭐ NEW

```yaml
  # Candlestick Patterns (requires pandas-ta-classic)
  enable_candlestick_patterns: false  # Set true to enable
```

**Warning**: Adds 60+ columns. Enable only if needed.

### Multi-Timeframe Context

```yaml
  enable_multivariate: false
  target_timeframe: "H1"
  target_symbol: "BTCUSD"  # Optional: filter to single symbol
  
  multivariate_config:
    context_timeframes: ["H4", "D1"]
    
    context_features:
      H4: ["close", "volume", "rsi_14", "sma_20"]
      D1: ["sma_50", "volatility_24"]
    
    context_symbols:
      BTCUSD: ["H4"]
      ETHUSD: ["H4"]
```

---

## Training

```yaml
train:
  model_name: "ml_orderflow"
  model_type: "xgboost"  # Options: xgboost, lightgbm, randomforest, nhits, lstm
  
  # Lag features
  lags: [1, 2, 3, 7, 14, 21]
  
  # Forecasting
  forecast_horizon: 5  # N-step ahead
  
  # Deep learning models only
  input_size: 128  # Context window for NHITS/LSTM
  
  # Static features
  use_static_features: false  # Use symbol/timeframe as features
```

**Note**: Target column is **auto-detected** from preprocessing metadata. No need to specify `target_col` here!

---

## Validation

```yaml
validation:
  schema:
    required_columns: ["open", "high", "low", "close", "volume"]
    numeric_columns: ["open", "high", "low", "close", "volume"]
  
  anomaly_protection:
    check_zscore: true
    zscore_threshold: 4.0
    action: "warn"  # Options: warn, drop
```

---

## LLM Configuration

### Provider Selection

```yaml
llm:
  provider: "gemini"  # Options: gemini, openai, dummy
```

### Gemini
```yaml
  model: "gemini-1.5-flash"  # or gemini-pro, gemini-1.5-pro
  temperature: 0.7
  max_output_tokens: 1024
```

Set `GEMINI_API_KEY` environment variable.

### OpenAI
```yaml
  provider: "openai"
  openai_model: "gpt-3.5-turbo"  # or gpt-4
  temperature: 0.7
  max_output_tokens: 1024
```

Set `OPENAI_API_KEY` environment variable.

### Dummy (Testing)
```yaml
  provider: "dummy"  # No API key needed
```

### Caching ⭐ NEW

```yaml
  # Caching
  cache_enabled: true
  cache_ttl_seconds: 3600  # 1 hour
```

**Benefits**:
- Reduces API costs
- Faster responses for repeated queries
- Automatic cache invalidation after TTL

### Advanced Prompting ⭐ NEW

```yaml
  # Advanced prompting
  use_technical_context: true      # Include RSI, volatility, etc.
  include_market_regime: true      # Detect bullish/bearish momentum
  detect_anomalies: true           # Alert on price spikes
  anomaly_z_threshold: 3.0         # Z-score threshold
```

---

## Robustness

### Circuit Breaker ⭐ NEW

```yaml
robustness:
  circuit_breaker:
    failure_threshold: 5        # Open after N failures
    timeout_seconds: 60         # Stay open for N seconds
    half_open_max_calls: 3      # Test with N calls before closing
```

**Protects**:
- CCXT exchange API calls
- MT5 terminal connections

### Retry Logic ⭐ NEW

```yaml
  retry:
    max_attempts: 3
    base_delay_seconds: 2
    max_delay_seconds: 30
    jitter: true  # Add randomness to avoid thundering herd
```

### Timeouts

```yaml
  timeouts:
    data_fetch_seconds: 30
    model_training_minutes: 60
    inference_seconds: 10
```

---

## MLflow

```yaml
mlflow:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: ml_orderflow
  model_registry_name: ml_orderflow
```

**View UI**:
```bash
mlflow ui
```

---

## Schedule

```yaml
schedule:
  interval_minutes: 60
  misfire_grace_time: 60
  enabled: true
```

---

## Quick Reference

### Minimal Configuration
```yaml
data_ingestion:
  source: "mt5"
  mt5:
    symbols: ["BTCUSD"]
    timeframes: ["H4"]
    n_bars: 500

preprocessing:
  target_source: "returns"
  target_window: 1

train:
  model_type: "xgboost"
  lags: [1, 7, 14]

llm:
  provider: "dummy"  # No API key needed
```

### Production Configuration
```yaml
data_ingestion:
  source: "mt5"
  mt5:
    symbols: ["BTCUSD", "ETHUSD", "XAUUSD"]
    timeframes: ["H4", "W1"]
    n_bars: 1000

preprocessing:
  target_source: "returns"
  target_window: 1
  lsp_window: 100
  enable_candlestick_patterns: false

train:
  model_type: "xgboost"
  lags: [1, 2, 3, 7, 14, 21]
  forecast_horizon: 5

llm:
  provider: "gemini"
  model: "gemini-1.5-flash"
  cache_enabled: true
  use_technical_context: true
  include_market_regime: true
  detect_anomalies: true

robustness:
  circuit_breaker:
    failure_threshold: 5
    timeout_seconds: 60
  retry:
    max_attempts: 3
```

---

**Last Updated**: 2025-12-26
