# API Reference

Complete reference for the ML Orderflow FastAPI endpoints.

**Base URL**: `http://localhost:5000`

---

## Table of Contents
1. [Health & Status](#health--status)
2. [Analysis & Forecasting](#analysis--forecasting)
3. [LLM Summaries](#llm-summaries)
4. [Anomaly Detection](#anomaly-detection)
5. [Cache Management](#cache-management)

---

## Health & Status

### GET /health

Comprehensive health check including LLM provider status.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "Time Series Forecasting",
  "llm": {
    "provider": "GeminiProvider",
    "provider_healthy": true,
    "cache_enabled": true,
    "cache_stats": {
      "size": 15,
      "hits": 42,
      "misses": 18,
      "hit_rate_percent": 70.0
    }
  }
}
```

---

## Analysis & Forecasting

### GET /analysis

Returns analysis of latest pre-computed inferences.

**Response**: Array of analysis objects
```json
[
  {
    "symbol": "BTCUSD",
    "timestamp": "2025-12-26T12:00:00",
    "predicted_return": 0.0023,
    "trend": "bullish",
    "momentum_score": 0.75,
    "volatility_impact": "moderate"
  }
]
```

### POST /forecast

Dynamic forecasting with custom input data.

**Request**:
```json
[
  {
    "unique_id": "BTCUSD_H1",
    "ds": "2025-12-26T10:00:00",
    "open": 42000,
    "high": 42500,
    "low": 41800,
    "close": 42300,
    "volume": 1500000
  }
]
```

**Response**: Forecast results
```json
[
  {
    "unique_id": "BTCUSD_H1",
    "ds": "2025-12-26T11:00:00",
    "y_pred": 0.0015
  }
]
```

---

## LLM Summaries

### POST /predict/summary

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
  "summary": "**Market Summary: BTCUSD**\n\nThe overall outlook for BTCUSD is **Bullish**. The model predictions indicate consistent positive returns across multiple timeframes, suggesting strong upward momentum.\n\n**Market Regime:** Strong Bullish Momentum\n\n**Technical Indicators:**\n- RSI is neutral at 55.3\n- Current volatility: 0.0234\n- Trend: strong upward (slope: 0.6543)\n- Price in high-liquidity zone (POC: 42150.00)\n\n**Key Observations:** The H4 timeframe shows the strongest bullish signal with a predicted return of 0.0045. The weekly timeframe confirms this with sustained positive momentum.\n\n**Risk Assessment:** Confidence is high given the alignment across timeframes. However, elevated volatility suggests potential for short-term pullbacks.\n\n**Actionable Insight:** Consider long positions with tight stop-losses below the POC price. Monitor for volume confirmation.",
  "provider": "GeminiProvider"
}
```

**Features**:
- ✅ Response caching (1-hour TTL)
- ✅ Technical indicator analysis
- ✅ Market regime detection
- ✅ Anomaly alerts
- ✅ Liquidity zone context

### POST /predict/summary/stream

Streaming market summary for real-time display.

**Request**:
```json
{
  "symbol": "ETHUSD"
}
```

**Response**: Text stream (progressive tokens)

**Use Case**: Real-time UI updates, better UX for long responses

**Example (curl)**:
```bash
curl -X POST http://localhost:5000/predict/summary/stream \
  -H "Content-Type: application/json" \
  -d '{"symbol": "ETHUSD"}' \
  --no-buffer
```

---

## Anomaly Detection

### GET /anomalies/detect

Standalone anomaly detection (no LLM, fast).

**Query Parameters**:
- `symbol` (optional): Filter by symbol

**Example**: `GET /anomalies/detect?symbol=XAUUSD`

**Response**:
```json
{
  "symbol": "XAUUSD",
  "anomalies": [
    {
      "type": "price_spike",
      "severity": 2.5,
      "direction": "upward",
      "value": 0.0456,
      "z_score": 3.2,
      "timestamp": "2025-12-26T12:00:00",
      "message": "Extreme upward price prediction detected"
    },
    {
      "type": "volume_anomaly",
      "severity": 1.8,
      "direction": "spike",
      "value": 2500000,
      "z_score": 3.5,
      "deviation_percent": 250.0,
      "timestamp": "2025-12-26T12:00:00",
      "message": "Unusual high volume detected"
    }
  ],
  "summary": {
    "total_anomalies": 3,
    "max_severity": 2.5,
    "critical_count": 1,
    "types": {
      "price_spike": 1,
      "volume_anomaly": 1,
      "volatility_spike": 1
    },
    "timestamp": "2025-12-26T12:05:00"
  }
}
```

**Anomaly Types**:
- `price_spike`: Extreme predictions (z-score > 3.0)
- `volume_anomaly`: Unusual volume (z-score > 3.0)
- `volatility_spike`: Elevated volatility (z-score > 2.5)
- `trend_reversal`: Sharp directional changes

**Best For**: Automated systems, high-frequency monitoring

### POST /anomalies/detect/stream

Real-time anomaly streaming in NDJSON format.

**Request**:
```json
{
  "symbol": "BTCUSD"
}
```

**Response** (NDJSON):
```json
{"type": "summary", "data": {"total_anomalies": 3, "critical_count": 1}}
{"type": "anomaly", "data": {"type": "price_spike", "severity": 2.5, ...}}
{"type": "anomaly", "data": {"type": "volume_anomaly", "severity": 1.8, ...}}
```

**Best For**: WebSocket clients, live dashboards

### POST /anomalies/explain

LLM-enhanced anomaly explanation.

**Request**:
```json
{
  "symbol": "GBPUSD"
}
```

**Response**:
```json
{
  "symbol": "GBPUSD",
  "anomalies": [...],
  "summary": {...},
  "llm_explanation": "The most critical anomaly is an extreme upward price spike (z-score 3.2), suggesting potential breakout momentum. This could be driven by strong buying pressure or news events. The elevated volatility (z-score 2.8) indicates increased market uncertainty. Traders should monitor for confirmation signals before entering positions, as false breakouts are common in high-volatility environments. Consider setting tight stop-losses below the recent low and scaling into positions gradually."
}
```

**Best For**: Human traders, reports, notifications

---

## Cache Management

### POST /llm/cache/clear

Clear the LLM response cache.

**Response**:
```json
{
  "status": "success",
  "message": "LLM cache cleared"
}
```

### GET /llm/cache/stats

Get cache performance statistics.

**Response**:
```json
{
  "size": 15,
  "max_size": 100,
  "hits": 42,
  "misses": 18,
  "hit_rate_percent": 70.0,
  "ttl_seconds": 3600
}
```

---

## Error Responses

All endpoints return standard HTTP error codes:

**404 Not Found**:
```json
{
  "detail": "Inference results not found. Run pipeline first."
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Summary generation failed: API rate limit exceeded"
}
```

**503 Service Unavailable**:
```json
{
  "detail": "Model not loaded"
}
```

---

## Examples

### PowerShell

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get

# LLM summary
$body = @{
    symbol = "BTCUSD"
    use_cache = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict/summary" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

# Anomaly detection
Invoke-RestMethod -Uri "http://localhost:5000/anomalies/detect?symbol=XAUUSD" -Method Get

# LLM-enhanced anomalies
$body = @{ symbol = "GBPUSD" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/anomalies/explain" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

# Cache stats
Invoke-RestMethod -Uri "http://localhost:5000/llm/cache/stats" -Method Get
```

### Bash/curl

```bash
# Health check
curl http://localhost:5000/health

# LLM summary
curl -X POST http://localhost:5000/predict/summary \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD", "use_cache": true}'

# Streaming summary
curl -X POST http://localhost:5000/predict/summary/stream \
  -H "Content-Type: application/json" \
  -d '{"symbol": "ETHUSD"}' \
  --no-buffer

# Anomaly detection
curl "http://localhost:5000/anomalies/detect?symbol=XAUUSD"

# Streaming anomalies
curl -X POST http://localhost:5000/anomalies/detect/stream \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD"}' \
  --no-buffer

# Clear cache
curl -X POST http://localhost:5000/llm/cache/clear
```

---

**Last Updated**: 2025-12-26
