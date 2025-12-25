import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn
from contextlib import asynccontextmanager
from trend_analysis.utils.config import settings
from trend_analysis.utils.initializer import logger_instance
from trend_analysis.core.analysis import MarketAnalyzer

logger = logger_instance.get_logger()

# Global variables
model = None
analyzer = MarketAnalyzer()

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

from trend_analysis.services.llm_service import LLMService

# Global variables
model = None
analyzer = MarketAnalyzer()
llm_service = LLMService()

# ... (Previous code remains) ...

class SummaryRequest(BaseModel):
    symbol: Optional[str] = None
    
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

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "mode": "Time Series Forecasting",
        "llm_provider": type(llm_service.provider).__name__
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
