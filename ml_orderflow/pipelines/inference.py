import os
import pandas as pd
import pickle
import numpy as np
from trend_analysis.utils.config import settings
from trend_analysis.utils.initializer import logger_instance
from trend_analysis.pipelines.preprocess import calculate_features
from mlforecast import MLForecast  # for type checking if needed

# Try importing NeuralForecast to check instance types cleanly
try:
    from neuralforecast import NeuralForecast
except ImportError:
    NeuralForecast = None

logger = logger_instance.get_logger()

def run_inference():
    prep_params = settings.params['preprocessing']
    train_params = settings.params['train']
    results_dir = os.path.join(settings.params['base']['data_dir'], "results")
    os.makedirs(results_dir, exist_ok=True)
    
    processed_data_file = os.path.join(prep_params['processed_data_path'], "processed_dataset.csv")
    model_name = train_params['model_name']
    
    # We load the raw model pickle
    model_file = os.path.join(settings.params['base']['model_dir'], f"{model_name}.pkl")
    
    logger.info(f"Starting automated inference using model: {model_name}")
    
    if not os.path.exists(model_file):
        logger.error(f"Model file not found at {model_file}. Please run training first.")
        return
        
    if not os.path.exists(processed_data_file):
        logger.error(f"Processed data not found at {processed_data_file}.")
        return

    # Load Model Object
    with open(model_file, 'rb') as f:
        fcst = pickle.load(f)
        
    df = pd.read_csv(processed_data_file)
    df['time'] = pd.to_datetime(df['time'])
    target_col = train_params['target_col']
    horizon = train_params.get('forecast_horizon', 24)
    
    # Prepare base DataFrame
    # Create composite unique_id
    if 'timeframe' in df.columns:
        df['unique_id'] = df['symbol'] + "_" + df['timeframe']
    else:
        df['unique_id'] = df['symbol']

    mlf_df = df.rename(columns={
        'time': 'ds',
        target_col: 'y'
    })
    
    # Drop symbol and timeframe as they are now encoded in unique_id or redundant
    cols_to_drop = ['symbol', 'timeframe']
    mlf_df = mlf_df.drop(columns=[c for c in cols_to_drop if c in mlf_df.columns], errors='ignore')
    
    # Convert non-numeric columns (like reg_trend or merged context trends) to category/numeric
    for col in mlf_df.columns:
        if mlf_df[col].dtype == 'object' and col not in ['unique_id', 'ds']:
            mlf_df[col] = mlf_df[col].astype('category').cat.codes
        
    # Drop rows where y is NaN (important for NeuralForecast)
    mlf_df = mlf_df.dropna(subset=['y'])
        
    logger.info(f"Generating {horizon}-step forecasts...")

    try:
        # Check if it is NeuralForecast
        is_neural = False
        if NeuralForecast is not None and isinstance(fcst, NeuralForecast):
            is_neural = True
            
        if is_neural:
            # NeuralForecast predict is simpler. It takes a future dataframe (if exog used) or just works if h is set?
            # Actually NeuralForecast.predict() usually returns forecasts for the horizon specified at init.
            # If we used exogenous variables in training (we did pass the full df), we might need future exog values.
            # But standard NHITS/LSTM often just use the target series unless configured otherwise.
            # For robustness, we will try standard predict().
            # If it uses historic-only context, it needs the input 'df' which is effectively the history.
            # Wait, fcst.predict() in NeuralForecast usually requires 'futr_df' only if we have future exog.
            
            # To be safe for NeuralForecast, we generally don't need the intricate iterative loop 
            # unless we have dynamic exog we MUST forecast manually. 
            # Let's assume standard behavior: generate forecasts based on latest window in training (or passed data?).
            # fcst.predict(df=...) might be needed if we want to predict on NEW data not seen in fit.
            # Since we just loaded processed data (which matches training data roughly), we can pass it.
            
            logger.info("Using NeuralForecast prediction path.")
            forecasts = fcst.predict(df=mlf_df) # This returns full horizon forecasts
            
            # Filter to ensure we only save future predictions if it returns history? 
            # NeuralForecast predict usually returns 'h' steps into future.
            
        else:
            # MLForecast Path (Iterative or Naive for Exogenous)
            logger.info("Using MLForecast prediction path.")
            
            # Naive Exogenous Approach: Forward fill last known values for future horizon
            future_df = fcst.make_future_dataframe(h=horizon) 
            last_values = mlf_df.groupby('unique_id').tail(1).drop(columns=['ds', 'y'])
            future_with_features = future_df.merge(last_values, on='unique_id', how='left')
            
            forecasts = fcst.predict(h=horizon, X_df=future_with_features)
        
        # Save results
        output_file = os.path.join(results_dir, "inferences.csv")
        forecasts.to_csv(output_file, index=False)
        logger.info(f"Inference complete. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_inference()
