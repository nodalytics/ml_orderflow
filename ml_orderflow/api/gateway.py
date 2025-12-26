import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn
from contextlib import asynccontextmanager
from ml_orderflow.utils.config import settings
from ml_orderflow.utils.initializer import logger_instance
from ml_orderflow.services.market_analyzer import MarketAnalyzer
from ml_orderflow.services.llm_service import LLMService
from ml_orderflow.services.anomaly_detector import AnomalyDetector

logger = logger_instance.get_logger()

# Global variables
model = None
analyzer = MarketAnalyzer()
llm_service = LLMService()
anomaly_detector = AnomalyDetector(z_threshold=3.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Use the specific time series model registry name
    model_name = settings.params['mlflow']['model_registry_name']
    model_uri = f"models:/{model_name}/latest"
    logger.info(f"Loading Time Series model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback to local model if possible
        model_path = os.path.join(settings.params['base']['model_dir'], f"{settings.params['train']['model_name']}.pkl")
        if os.path.exists(model_path):
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded local model from {model_path} as fallback.")
        else:
            model = None
    yield

app = FastAPI(title="Trend Analysis AI Gateway", version="2.0.0", lifespan=lifespan)

class AnalysisResponse(BaseModel):
    symbol: str
    timestamp: Any
    predicted_return: float
    trend: str
    momentum_score: float
    volatility_impact: str

@app.get("/analysis", response_model=List[AnalysisResponse])
async def get_latest_analysis():
    """
    Returns the analysis for the latest pre-computed inferences.
    This reads from the automated inference pipeline output.
    """
    inference_file = os.path.join(settings.params['base']['data_dir'], "results", "inferences.csv")
    if not os.path.exists(inference_file):
        raise HTTPException(status_code=404, detail="Inference results not found. Run pipeline first.")
    
    try:
        inferences_df = pd.read_csv(inference_file)
        # The column name in inferences.csv is the model_name defined in params.yaml
        model_name = settings.params['train']['model_name']
        
        report = analyzer.analyze_results(inferences_df, model_name)
        return report
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
async def dynamic_forecast(data: List[Dict[str, Any]]):
    """
    Directly forecast returns for provided data using the loaded MLForecast model.
    Expected format: List of dicts with 'unique_id', 'ds', and exogenous features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_df = pd.DataFrame(data)
        if 'ds' in input_df.columns:
            input_df['ds'] = pd.to_datetime(input_df['ds'])
        
        # If it's the pyfunc wrapper from MLflow, it takes dict or DF
        # If it's the raw pickle, we use its predict method
        if hasattr(model, 'predict'):
            # In case of MLflow PyFunc model
            forecasts = model.predict(input_df)
        else:
            # High-level MLForecast predict
            forecasts = model.predict(h=1, X_df=input_df)
            
        return forecasts.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Dynamic forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SummaryRequest(BaseModel):
    symbol: Optional[str] = None
    use_cache: Optional[bool] = True
    
@app.post("/predict/summary")
async def generate_prediction_summary(request: SummaryRequest):
    """
    Generates a natural language summary of the latest market predictions using an LLM.
    """
    inference_file = os.path.join(settings.params['base']['data_dir'], "results", "inferences.csv")
    if not os.path.exists(inference_file):
        raise HTTPException(status_code=404, detail="Inference results not found. Run pipeline first.")
    
    try:
        inferences_df = pd.read_csv(inference_file)
        # Convert to list of dicts for the service
        inferences_data = inferences_df.to_dict(orient='records')
        
        summary = llm_service.generate_market_summary(inferences_data, request.symbol)
        
        return {
            "symbol": request.symbol or "All",
            "summary": summary,
            "provider": type(llm_service.provider).__name__
        }
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/summary/stream")
async def generate_prediction_summary_stream(request: SummaryRequest):
    """
    Generates a streaming natural language summary of the latest market predictions using an LLM.
    Returns chunks of text as they're generated for real-time display.
    """
    inference_file = os.path.join(settings.params['base']['data_dir'], "results", "inferences.csv")
    if not os.path.exists(inference_file):
        raise HTTPException(status_code=404, detail="Inference results not found. Run pipeline first.")
    
    try:
        inferences_df = pd.read_csv(inference_file)
        inferences_data = inferences_df.to_dict(orient='records')
        
        def generate():
            for chunk in llm_service.generate_market_summary_stream(inferences_data, request.symbol):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Streaming summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/cache/clear")
async def clear_llm_cache():
    """
    Clears the LLM response cache.
    """
    try:
        llm_service.clear_cache()
        return {"status": "success", "message": "LLM cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/cache/stats")
async def get_cache_stats():
    """
    Returns LLM cache statistics.
    """
    return llm_service.get_cache_stats()

@app.get("/anomalies/detect")
async def detect_anomalies(symbol: Optional[str] = None):
    """
    Standalone anomaly detection endpoint (no LLM).
    Returns structured anomaly data for automated systems.
    """
    inference_file = os.path.join(settings.params['base']['data_dir'], "results", "inferences.csv")
    if not os.path.exists(inference_file):
        raise HTTPException(status_code=404, detail="Inference results not found. Run pipeline first.")
    
    try:
        inferences_df = pd.read_csv(inference_file)
        
        # Filter by symbol if requested
        if symbol:
            inferences_df = inferences_df[
                inferences_df['unique_id'].str.startswith(symbol) if 'unique_id' in inferences_df.columns
                else inferences_df.get('symbol', pd.Series()) == symbol
            ]
            
            if inferences_df.empty:
                return {
                    "symbol": symbol,
                    "anomalies": [],
                    "summary": {"total_anomalies": 0}
                }
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_all(inferences_df)
        summary = anomaly_detector.get_summary_stats(anomalies)
        
        return {
            "symbol": symbol or "all",
            "anomalies": anomalies,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anomalies/detect/stream")
async def detect_anomalies_stream(request: SummaryRequest):
    """
    Streaming anomaly detection endpoint.
    Sends anomalies as they're detected in Server-Sent Events format.
    """
    inference_file = os.path.join(settings.params['base']['data_dir'], "results", "inferences.csv")
    if not os.path.exists(inference_file):
        raise HTTPException(status_code=404, detail="Inference results not found. Run pipeline first.")
    
    try:
        inferences_df = pd.read_csv(inference_file)
        
        # Filter by symbol if requested
        if request.symbol:
            inferences_df = inferences_df[
                inferences_df['unique_id'].str.startswith(request.symbol) if 'unique_id' in inferences_df.columns
                else inferences_df.get('symbol', pd.Series()) == request.symbol
            ]
        
        def generate():
            import json
            
            # Detect anomalies
            anomalies = anomaly_detector.detect_all(inferences_df)
            
            if not anomalies:
                yield json.dumps({"message": "No anomalies detected", "count": 0}) + "\n"
                return
            
            # Send summary first
            summary = anomaly_detector.get_summary_stats(anomalies)
            yield json.dumps({"type": "summary", "data": summary}) + "\n"
            
            # Stream each anomaly
            for anomaly in anomalies:
                yield json.dumps({"type": "anomaly", "data": anomaly}) + "\n"
        
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except Exception as e:
        logger.error(f"Streaming anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anomalies/explain")
async def explain_anomalies_with_llm(request: SummaryRequest):
    """
    LLM-enhanced anomaly explanation endpoint.
    Combines structured anomaly detection with natural language explanation.
    """
    inference_file = os.path.join(settings.params['base']['data_dir'], "results", "inferences.csv")
    if not os.path.exists(inference_file):
        raise HTTPException(status_code=404, detail="Inference results not found. Run pipeline first.")
    
    try:
        inferences_df = pd.read_csv(inference_file)
        
        # Filter by symbol if requested
        if request.symbol:
            inferences_df = inferences_df[
                inferences_df['unique_id'].str.startswith(request.symbol) if 'unique_id' in inferences_df.columns
                else inferences_df.get('symbol', pd.Series()) == request.symbol
            ]
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_all(inferences_df)
        summary = anomaly_detector.get_summary_stats(anomalies)
        
        # Get LLM explanation if anomalies found
        llm_explanation = None
        if anomalies:
            # Create focused prompt for anomaly explanation
            import json
            anomaly_data = json.dumps(anomalies[:5], indent=2)  # Limit to top 5
            
            prompt = f"""You are a financial risk analyst. Analyze these detected market anomalies:

{anomaly_data}

Provide a concise explanation (max 150 words) covering:
1. The most critical anomaly and its implications
2. Potential causes
3. Recommended actions for traders

Be direct and actionable."""
            
            try:
                llm_explanation = llm_service.provider.generate_content(prompt)
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}")
                llm_explanation = "LLM explanation unavailable"
        
        return {
            "symbol": request.symbol or "all",
            "anomalies": anomalies,
            "summary": summary,
            "llm_explanation": llm_explanation
        }
    except Exception as e:
        logger.error(f"Anomaly explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    """
    Comprehensive health check endpoint.
    """
    llm_health = llm_service.health_check()
    
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "mode": "Time Series Forecasting",
        "llm": llm_health
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
