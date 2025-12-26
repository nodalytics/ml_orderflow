# ML Orderflow - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Pipeline Stages](#pipeline-stages)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Overview

ML Orderflow is a production-ready machine learning pipeline for time series forecasting in financial markets. It combines traditional technical analysis with modern ML techniques and LLM-powered insights.

**Tech Stack:**
- **Data Sources**: MetaTrader 5, CCXT exchanges
- **ML Frameworks**: MLForecast, NeuralForecast, XGBoost, LightGBM
- **MLOps**: DVC (pipeline orchestration), MLflow (experiment tracking)
- **API**: FastAPI with streaming support
- **LLM**: Google Gemini, OpenAI GPT (configurable)

---

## Features

### Core Capabilities
- ✅ Multi-symbol, multi-timeframe data ingestion
- ✅ Advanced technical indicators (100+ features)
- ✅ N-step ahead forecasting
- ✅ Automated model training and versioning
- ✅ Real-time inference API

### Robustness Features
- ✅ Circuit breaker pattern for external APIs
- ✅ Retry logic with exponential backoff
- ✅ Graceful degradation (continues with partial data)
- ✅ Data quality validation and anomaly detection
- ✅ Health check endpoints

### LLM Integration
- ✅ Natural language market summaries
- ✅ Multi-provider support (Gemini, OpenAI, Dummy)
- ✅ Response caching (1-hour TTL)
- ✅ Streaming responses
- ✅ Technical indicator interpretation
- ✅ Market regime detection
- ✅ Anomaly alerts in summaries

### Advanced Indicators
- ✅ Linear regression channels
- ✅ Liquidity sentiment profile (POC, high/low value areas)
- ✅ Candlestick patterns (60+ patterns via pandas-ta-classic)
- ✅ Custom volatility metrics
- ✅ Multi-timeframe context merging

---

## Installation

### Prerequisites
- Python 3.11+
- MetaTrader 5 (optional, for MT5 data)
- API keys: `GEMINI_API_KEY` or `OPENAI_API_KEY`

### Setup

```bash
# Clone repository
git clone <repo-url>
cd ml_orderflow

# Install dependencies
uv sync

# Set environment variables
cp .env.example .env
# Edit .env and add your API keys

# Initialize DVC
dvc init

# Run the pipeline
uv run dvc repro
```

---

## Configuration

All configuration is in [`params.yaml`](../params.yaml). Key sections:

### Data Ingestion
```yaml
data_ingestion:
  source: "mt5"  # or "mexc", "binance", etc.
  raw_data_path: data/raw
  
  mt5:
    symbols: ["BTCUSD", "ETHUSD", "XAUUSD"]
    timeframes: ["H4", "W1"]
    n_bars: 1000
```

### Preprocessing
```yaml
preprocessing:
  # Target configuration (FLEXIBLE!)
  target_column: "next_return"      # Name of target column
  target_source: "returns"          # Source column to shift
  target_window: 1                  # Periods ahead
  
  # Examples:
  # - Predict next close: target_source="close"
  # - Predict volume: target_source="volume"
  # - 3 steps ahead: target_window=3
  
  # Liquidity sentiment profile
  lsp_window: 100
  lsp_bins: 20
  
  # Candlestick patterns
  enable_candlestick_patterns: false  # Set true to enable
```

### Training
```yaml
train:
  model_name: "ml_orderflow"
  model_type: "xgboost"  # xgboost, lightgbm, randomforest, nhits, lstm
  lags: [1, 2, 3, 7, 14, 21]
  forecast_horizon: 5
```

### LLM Configuration
```yaml
llm:
  provider: "gemini"  # or "openai", "dummy"
  model: "gemini-1.5-flash"
  
  # Caching
  cache_enabled: true
  cache_ttl_seconds: 3600
  
  # Advanced features
  use_technical_context: true
  include_market_regime: true
  detect_anomalies: true
```

### Robustness
```yaml
robustness:
  circuit_breaker:
    failure_threshold: 5
    timeout_seconds: 60
  
  retry:
    max_attempts: 3
    base_delay_seconds: 2
```

---

## Pipeline Stages

### 1. Data Ingestion
**Command**: `uv run python -m ml_orderflow.pipelines.data_ingestion`

**Features**:
- Fetches data from MT5 or CCXT exchanges
- Circuit breaker protection
- Retry logic with exponential backoff
- Graceful degradation (continues if some symbols fail)
- Validates data quality

**Output**: `data/raw/dataset.csv`

### 2. Preprocessing
**Command**: `uv run python -m ml_orderflow.pipelines.preprocess`

**Features**:
- Calculates 100+ technical indicators
- Liquidity sentiment profile
- Candlestick patterns (optional)
- Multi-timeframe context merging
- **Flexible target generation** (any column, any window)
- Saves metadata for training stage

**Output**: 
- `data/processed/processed_dataset.csv`
- `data/processed/preprocessing_metadata.json`

**Metadata Example**:
```json
{
  "target_column": "next_return",
  "target_source": "returns",
  "target_window": 1,
  "feature_columns": ["open", "high", "low", ...],
  "timestamp": "2025-12-26T13:00:00"
}
```

### 3. Training
**Command**: `uv run python -m ml_orderflow.pipelines.train`

**Features**:
- **Auto-detects target column** from metadata
- Supports MLForecast and NeuralForecast
- Multiple model types (XGBoost, LightGBM, NHITS, LSTM)
- Logs to MLflow
- Registers model in MLflow Model Registry

**Output**: `models/ml_orderflow.pkl`

### 4. Inference
**Command**: `uv run python -m ml_orderflow.pipelines.inference`

**Features**:
- N-step ahead forecasting
- Uses latest registered model
- Saves predictions with timestamps

**Output**: `data/results/inferences.csv`

---

## Advanced Features

### Flexible Target Column

You can predict **any column** at **any time horizon**:

**Example 1: Predict next return** (default)
```yaml
target_column: "next_return"
target_source: "returns"
target_window: 1
```

**Example 2: Predict close price 3 steps ahead**
```yaml
target_column: "close_3_ahead"
target_source: "close"
target_window: 3
```

**Example 3: Predict volume**
```yaml
target_column: "next_volume"
target_source: "volume"
target_window: 1
```

The training stage automatically reads the target column from metadata - **no manual configuration needed**!

### Liquidity Sentiment Profile

Identifies high-liquidity and low-liquidity zones:

**Features**:
- Point of Control (POC) - highest volume price
- High Value Area (70% of volume)
- Low Value Area (10% of volume)

**Columns Added**:
- `lsp_poc_price`
- `lsp_high_value_low/high`
- `lsp_low_value_low/high`
- `lsp_bin_width`

**LLM Context**: "Price in high-liquidity zone (POC: 1850.50)"

### Candlestick Patterns

60+ patterns via pandas-ta-classic:

**Enable**:
```yaml
enable_candlestick_patterns: true
```

**Columns Added**: `CDL_DOJI`, `CDL_ENGULFING`, `CDL_HAMMER`, etc.

**LLM Context**: "Candlestick patterns: Bullish Engulfing, Hammer"

### Multi-Timeframe Context

Merge features from different timeframes:

```yaml
enable_multivariate: true
multivariate_config:
  target_timeframe: "H1"
  context_timeframes: ["H4", "D1"]
  
  context_features:
    H4: ["close", "volume", "rsi_14"]
    D1: ["sma_50", "volatility_24"]
```

---

## API Reference

### Start API Server
```bash
uv run python -m ml_orderflow.api.gateway
```

Server runs on `http://localhost:5000`

### Endpoints

#### 1. GET /health
Health check with LLM provider status.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "llm": {
    "provider": "GeminiProvider",
    "provider_healthy": true,
    "cache_enabled": true
  }
}
```

#### 2. GET /analysis
Latest market analysis from inferences.

#### 3. POST /forecast
Dynamic forecasting with custom data.

#### 4. POST /predict/summary
Generate market summary with LLM (cached).

**Request**:
```json
{
  "symbol": "BTCUSD",
  "use_cache": true
}
```

**Response**:
```json
{
  "symbol": "BTCUSD",
  "summary": "The overall outlook for BTCUSD is Bullish...",
  "provider": "GeminiProvider"
}
```

#### 5. POST /predict/summary/stream
Streaming market summary (real-time tokens).

#### 6. GET /anomalies/detect
Standalone anomaly detection (no LLM).

**Query**: `?symbol=XAUUSD`

**Response**:
```json
{
  "anomalies": [
    {
      "type": "price_spike",
      "severity": 2.5,
      "z_score": 3.2,
      "message": "Extreme upward price prediction detected"
    }
  ],
  "summary": {
    "total_anomalies": 3,
    "critical_count": 1
  }
}
```

#### 7. POST /anomalies/detect/stream
Streaming anomaly detection (NDJSON).

#### 8. POST /anomalies/explain
LLM-enhanced anomaly explanation.

**Response includes**: structured anomalies + natural language explanation

#### 9. POST /llm/cache/clear
Clear LLM response cache.

#### 10. GET /llm/cache/stats
Cache performance statistics.

---

## Troubleshooting

### Common Issues

**1. DVC Pipeline Fails**
```bash
# Check DVC status
dvc status

# Force rerun a stage
dvc repro -f preprocess
```

**2. Model Not Loading**
```bash
# Check MLflow registry
mlflow ui

# Verify model exists
ls models/
```

**3. LLM API Errors**
- Check API key: `echo $GEMINI_API_KEY`
- Switch to dummy provider for testing:
  ```yaml
  llm:
    provider: "dummy"
  ```

**4. Circuit Breaker Open**
- Wait for timeout (default 60s)
- Or adjust threshold in `params.yaml`:
  ```yaml
  robustness:
    circuit_breaker:
      failure_threshold: 10  # More tolerant
  ```

**5. Target Column Not Found**
- Ensure preprocessing ran successfully
- Check `data/processed/preprocessing_metadata.json`
- Verify `target_source` column exists in data

### Logs

Check logs in `logs/` directory:
```bash
tail -f logs/ml_orderflow.log
```

---

## Best Practices

1. **Start Simple**: Use default configuration first
2. **Validate Data**: Check `data/raw/dataset.csv` after ingestion
3. **Monitor Metrics**: Use MLflow UI to track experiments
4. **Cache LLM**: Enable caching to reduce API costs
5. **Test Locally**: Use dummy LLM provider for development
6. **Version Control**: Commit `params.yaml` changes
7. **DVC Tracking**: Use `dvc push` to backup data/models

---

## Next Steps

- Check [`params.yaml`](../params.yaml) for all configuration options
- Explore notebooks in `notebooks/` for analysis examples

---

**Version**: 2.0.0  
**Last Updated**: 2025-12-26
