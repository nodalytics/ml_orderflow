# Quick Test Commands for ML Orderflow Enhancements

## Prerequisites
The API is running on `http://localhost:5000`

## 1. Health Check
```bash
curl http://localhost:5000/health
```

Expected: Shows model status and LLM provider health

## 2. Test LLM Summary (Standard)
```bash
curl -X POST http://localhost:5000/predict/summary \
  -H "Content-Type: application/json" \
  -d "{\"symbol\": \"BTCUSD\", \"use_cache\": true}"
```

Expected: Returns market summary with technical indicators and regime analysis

## 3. Test LLM Summary (Streaming)
```bash
curl -X POST http://localhost:5000/predict/summary/stream \
  -H "Content-Type: application/json" \
  -d "{\"symbol\": \"ETHUSD\"}" \
  --no-buffer
```

Expected: Tokens appear progressively in real-time

## 4. Check Cache Stats
```bash
curl http://localhost:5000/llm/cache/stats
```

Expected: Shows cache size, hits, misses, and hit rate

## 5. Test Cache Performance
```bash
# First request (cache miss - slower)
time curl -X POST http://localhost:5000/predict/summary \
  -H "Content-Type: application/json" \
  -d "{\"symbol\": \"BTCUSD\"}"

# Second request (cache hit - much faster)
time curl -X POST http://localhost:5000/predict/summary \
  -H "Content-Type: application/json" \
  -d "{\"symbol\": \"BTCUSD\"}"
```

Expected: Second request should be < 100ms

## 6. Clear Cache
```bash
curl -X POST http://localhost:5000/llm/cache/clear
```

Expected: `{"status": "success", "message": "LLM cache cleared"}`

## 7. Get Latest Analysis
```bash
curl http://localhost:5000/analysis
```

Expected: Returns analysis of latest inferences

## PowerShell Equivalents

### Health Check
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get
```

### LLM Summary
```powershell
$body = @{
    symbol = "BTCUSD"
    use_cache = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict/summary" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Cache Stats
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/llm/cache/stats" -Method Get
```

### Clear Cache
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/llm/cache/clear" -Method Post
```

## Notes

- **Gemini API Key**: Set `GEMINI_API_KEY` environment variable if using Gemini provider
- **OpenAI**: To use OpenAI instead, update `params.yaml` and set `OPENAI_API_KEY`
- **Cache TTL**: Default is 3600 seconds (1 hour), configurable in `params.yaml`
- **Streaming**: Works best with `--no-buffer` flag in curl

## Switching to OpenAI

1. Install OpenAI package:
   ```bash
   uv add openai
   ```

2. Update `params.yaml`:
   ```yaml
   llm:
     provider: "openai"
     openai_model: "gpt-3.5-turbo"  # or "gpt-4"
   ```

3. Set environment variable:
   ```bash
   $env:OPENAI_API_KEY = "your-key-here"
   ```

4. Restart the API
