# ML Orderflow API Reference

Complete API documentation for the enhanced ML Orderflow system.

## Base URL
```
http://localhost:5000
```

---

## Analysis Endpoints

### GET /analysis
Returns analysis of the latest pre-computed inferences.

**Response**: Array of analysis objects with trend, momentum, and volatility metrics.

---

## Forecasting Endpoints

### POST /forecast
Dynamic forecasting with provided data.

**Request Body**:
```json
[
  {
    "unique_id": "BTCUSD_H1",
    "ds": "2025-12-26T10:00:00",
    "feature1": 0.123,
    "feature2": 0.456
  }
]
```

**Response**: Forecast results as array of predictions.

---

## LLM Summary Endpoints

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
  "summary": "The overall outlook for BTCUSD is Bullish...",
  "provider": "GeminiProvider"
}
```

**Features**:
- ✅ Response caching (1-hour TTL)
- ✅ Technical indicator analysis
- ✅ Market regime detection
- ✅ Anomaly alerts

### POST /predict/summary/stream
Streaming market summary for real-time display.

**Request**:
```json
{
  "symbol": "ETHUSD"
}
```

**Response**: Text stream (progressive token generation)

**Use Case**: Real-time UI updates, better UX for long responses

---

## Anomaly Detection Endpoints

### GET /anomalies/detect
Standalone anomaly detection (no LLM, fast).

**Query Parameters**:
- `symbol` (optional): Filter by symbol (e.g., "BTCUSD")

**Response**:
```json
{
  "symbol": "BTCUSD",
  "anomalies": [
    {
      "type": "price_spike",
      "severity": 2.5,
      "direction": "upward",
      "value": 0.0456,
      "z_score": 3.2,
      "timestamp": "2025-12-26T10:00:00",
      "message": "Extreme upward price prediction detected"
    }
  ],
  "summary": {
    "total_anomalies": 3,
    "max_severity": 2.5,
    "critical_count": 1,
    "types": {
      "price_spike": 1,
      "volume_anomaly": 1
    }
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
Real-time anomaly streaming.

**Request**:
```json
{
  "symbol": "XAUUSD"
}
```

**Response** (NDJSON format):
```json
{"type": "summary", "data": {"total_anomalies": 3}}
{"type": "anomaly", "data": {"type": "price_spike", "severity": 2.5}}
{"type": "anomaly", "data": {"type": "volume_anomaly", "severity": 1.8}}
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
  "llm_explanation": "The most critical anomaly is an extreme upward price spike (z-score 3.2), suggesting potential breakout momentum. This could be driven by strong buying pressure or news events. Traders should monitor for confirmation signals before entering positions..."
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

## Health Check

### GET /health
Comprehensive system health check.

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
      "hit_rate_percent": 70.0
    }
  }
}
```

---

## Quick Examples

### PowerShell

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get

# LLM summary
$body = @{ symbol = "BTCUSD"; use_cache = $true } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/predict/summary" `
  -Method Post -ContentType "application/json" -Body $body

# Anomaly detection
Invoke-RestMethod -Uri "http://localhost:5000/anomalies/detect?symbol=XAUUSD" -Method Get

# LLM-enhanced anomalies
$body = @{ symbol = "GBPUSD" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/anomalies/explain" `
  -Method Post -ContentType "application/json" -Body $body
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
curl http://localhost:5000/anomalies/detect?symbol=XAUUSD

# Streaming anomalies
curl -X POST http://localhost:5000/anomalies/detect/stream \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSD"}' \
  --no-buffer
```

---

## Configuration

All endpoints respect the configuration in `params.yaml`:

```yaml
llm:
  provider: "gemini"  # or "openai", "dummy"
  model: "gemini-2.5-flash"
  cache_enabled: true
  cache_ttl_seconds: 3600
  use_technical_context: true
  include_market_regime: true
  detect_anomalies: true
  anomaly_z_threshold: 3.0
```

---

## Error Responses

All endpoints return standard HTTP error codes:

- **404**: Resource not found (e.g., no inference results)
- **500**: Internal server error
- **503**: Service unavailable (e.g., model not loaded)

**Example Error**:
```json
{
  "detail": "Inference results not found. Run pipeline first."
}
```
