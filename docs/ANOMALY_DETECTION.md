# Anomaly Detection Feature - Quick Test

## What Was Added

The LLM service now automatically detects and reports:

1. **Price Spike Alerts** - Extreme predictions (z-score > 3.0)
2. **Volume Anomalies** - Unusual high/low volume (z-score > 3.0)
3. **Volatility Alerts** - Elevated market volatility (z-score > 2.5)
4. **Trend Reversals** - Sharp directional changes

## Test Commands

### Restart API (Required)
```powershell
# Stop current API (Ctrl+C), then:
uv run python -m ml_orderflow.api.gateway
```

### Test Streaming with Anomaly Detection
```bash
curl -X POST http://localhost:5000/predict/summary/stream \
  -H "Content-Type: application/json" \
  -d '{"symbol": "XAUUSD"}' \
  --no-buffer
```

### Test Standard Endpoint
```bash
curl -X POST http://localhost:5000/predict/summary \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD"}'
```

## Example Output

When anomalies are detected, you'll see alerts like:

```
**⚠️ ANOMALY ALERTS:**
- ⚠️ **Price Spike Alert**: Detected extreme upward prediction (z-score > 3.0, value: 0.0456)
- ⚠️ **High Volatility Alert**: Market volatility is significantly elevated (z-score: 2.87)
- ⚠️ **Trend Reversal**: Sharp reversal detected (bullish to bearish)
```

## Configuration

In `params.yaml`:
```yaml
llm:
  detect_anomalies: true
  anomaly_z_threshold: 3.0  # Adjust sensitivity
```

## How It Works

- **Z-Score Analysis**: Compares current values to historical mean/std
- **Multi-Factor Detection**: Checks predictions, volume, volatility
- **Real-Time Alerts**: Included in both streaming and standard responses
- **Contextual**: LLM explains the impact of detected anomalies
