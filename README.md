# ML Orderflow

A professional machine learning pipeline for extracting trend insights and forecasting price movements from OHLCV market data.

## Features
- **Multi-Symbol Data Ingestion**: Robust data fetching from MetaTrader 5 (MT5) and CCXT-supported exchanges (MEXC, etc.).
- **Statistical Feature Extraction**: Automatic calculation of 20+ features including Log Returns, RSI, SMA, Volatility, and Linear Regression Channels.
- **Time Series Forecasting**: Unified training pipeline using `mlforecast` and XGBoost, with multi-symbol support.
- **Model Registry**: Full integration with MLflow for experiment tracking and model versioning.
- **Automation**: Built-in scheduling and DVC-powered pipeline management.

## Setup
1. **Dependencies**: Managed by `uv`.
   ```bash
   uv sync
   ```
2. **Configuration**: Edit `params.yaml` to configure data sources, features, and training parameters.
3. **Run Pipeline**:
   ```bash
   uv run dvc repro
   ```

## Configuration (`params.yaml`)

The pipeline behavior is controlled by `params.yaml`. Key sections include:

### Data Sources
- **`mexc` / `mt5`**:
    - **`symbols`**: List of assets to trade/analyze (e.g., `["BTCUSD", "ETHUSD"]`).
    - **`timeframes`**: List of timeframes to fetch (e.g., `["H1", "H4"]`).

### Preprocessing
- **`target_timeframe`**: The base timeframe for the final dataset (e.g., `"H1"`).
- **`target_symbol`**: (Optional) Filter final dataset to a specific symbol (e.g., `"ETHUSD"`). Useful for specialized models.

#### Multivariate Context
Enrich the dataset with features from other contexts (preventing lookahead bias):
- **`enable_multivariate`**: `true`/`false`. Enable merging of context features.
- **`multivariate_config`**: Defines the single target and its contexts.
    - **`target_timeframe`**: Base timeframe (e.g., "H1").
    - **`target_symbol`**: Base symbol (e.g., "ETHUSD").
    - **`context`**: Dictionary defining contexts to merge (`Symbol: [Timeframes]`).
    - Example:
      ```yaml
      multivariate_config:
        target_timeframe: "H1"
        target_symbol: "ETHUSD"
        context:
          BTCUSD: ["H1", "H4"]  # Merge BTC H1 and H4 context
          ETHUSD: ["H4"]        # Merge ETH H4 context (self-context)
      ```
    - The pipeline automatically handles suffixing (e.g., `_BTCUSD`, `_H4`, `_BTCUSD_H4`) and lookahead prevention.

### Training
- **`model_name`**: Name of the model registered in MLflow.
- **`use_static_features`**: `true`/`false`.
    - **Global Model**: Set to `true` when training on multiple symbols (Batch Mode) to let the model distinguish between assets.
    - **Multivariate Mode**: Automatically disabled (ignored) when `enable_multivariate` is `true`, as the model trains on a single target series.

## Automation & Scheduling

### 1. Python-based Scheduler (Cross-platform)
Use the built-in scheduler to run the pipeline at regular intervals defined in `params.yaml`.
```bash
uv run python -m trend_analysis.scheduler
```
_Configure `interval_minutes` in the `schedule` section of `params.yaml`._

### 2. Windows Task Scheduler (Recommended for Windows)
To ensure the pipeline runs even after system restarts or without a manual terminal session:

1. Open **Task Scheduler** on Windows.
2. Click **Create Basic Task**.
3. **Trigger**: Choose "Daily" or "When I log on". 
4. **Action**: "Start a Program".
5. **Program/script**: `uv`.
6. **Add arguments**: `run dvc repro` (or `run python -m trend_analysis.scheduler`).
7. **Start in**: `trend-analysis`.

## Tech Stack
- **Data**: MT5, CCXT, Pandas
- **Features**: NumPy, Scipy
- **ML**: MLForecast, NeuralForecast, XGBoost, Scikit-Learn
- **MLOps**: DVC, MLflow
- **API**: FastAPI, Uvicorn