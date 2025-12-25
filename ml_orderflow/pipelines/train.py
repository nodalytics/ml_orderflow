import os
import pandas as pd
import mlflow
import pickle
from mlforecast import MLForecast
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from trend_analysis.utils.config import settings
from trend_analysis.utils.initializer import logger_instance
from trend_analysis.ml.wrappers import ForecastWrapper
from mlflow.models import infer_signature

logger = logger_instance.get_logger()

def train_model():
    prep_params = settings.params['preprocessing']
    train_params = settings.params['train']
    
    processed_data_file = os.path.join(prep_params['processed_data_path'], "processed_dataset.csv")
    model_name = train_params['model_name']
    
    logger.info(f"Starting time series training for model: {model_name}")
    
    if not os.path.exists(processed_data_file):
        logger.error(f"Processed data not found at {processed_data_file}. Please run preprocessing first.")
        return

    # Load and format data for mlforecast
    df = pd.read_csv(processed_data_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Create composite unique_id
    if 'timeframe' in df.columns:
        df['unique_id'] = df['symbol'] + "_" + df['timeframe']
    else:
        df['unique_id'] = df['symbol']

    # mlforecast requires specific column names: unique_id, ds, y
    mlf_df = df.rename(columns={
        'time': 'ds',
        train_params['target_col']: 'y'
    })
    
    # Drop certain columns but keep static ones if requested
    use_static = train_params.get('use_static_features', False)
    
    # Check for multivariate mode (Single Series Mode)
    enable_multivariate = settings.params['preprocessing'].get('enable_multivariate', False)
    if enable_multivariate:
        logger.info("Multivariate mode enabled (Single Series). Disabling static features 'symbol'/'timeframe' as they are constant.")
        use_static = False

    static_features = []
    
    if use_static:
        logger.info("Using static features: ['symbol', 'timeframe']")
        # Encode symbol and timeframe
        if 'symbol' in mlf_df.columns:
            mlf_df['symbol'] = mlf_df['symbol'].astype('category').cat.codes
        if 'timeframe' in mlf_df.columns:
            mlf_df['timeframe'] = mlf_df['timeframe'].astype('category').cat.codes
            
        static_features = ['symbol', 'timeframe']
    else:
        # Drop symbol and timeframe as they are now encoded in unique_id or redundant
        cols_to_drop = ['symbol', 'timeframe']
        mlf_df = mlf_df.drop(columns=[c for c in cols_to_drop if c in mlf_df.columns], errors='ignore')

    # Convert non-numeric columns (like reg_trend or merged context trends) to category/numeric
    for col in mlf_df.columns:
        if mlf_df[col].dtype == 'object' and col not in ['unique_id', 'ds']:
            mlf_df[col] = mlf_df[col].astype('category').cat.codes

    # Drop rows where y is NaN (expected for shifted targets)
    before_drop = len(mlf_df)
    mlf_df = mlf_df.dropna(subset=['y'])
    logger.info(f"Dropped {before_drop - len(mlf_df)} rows with NaN targets.")
    
    # Log features for verification
    feature_cols = [c for c in mlf_df.columns if c not in ['unique_id', 'ds', 'y'] and c not in static_features]
    logger.info(f"Training with {len(feature_cols)} dynamic features: {feature_cols}")

    # Drop non-feature columns if they are not automated by mlforecast
    # Symbol and Time are handled by unique_id and ds
    # Any other column in the df will be treated as an exogenous feature
    
    # Select Model and Framework
    model_type = train_params.get('model_type', 'xgboost')
    input_size = train_params.get('input_size', 24)
    forecast_horizon = train_params.get('forecast_horizon', 24) # Ensure this is in params or default
    
    fcst = None # Framework object (MLForecast or NeuralForecast)
    
    if model_type in ['nhits', 'lstm']:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, LSTM
        
        logger.info(f"Initializing NeuralForecast with {model_type}...")
        
        # Prepare DL models
        # NeuralForecast automatically handles static features if they are in the dataframe
        # and we pass them or let it detect them.
        # But simpler to just rely on them being in df.
        
        if model_type == 'nhits':
            models = [NHITS(h=forecast_horizon, input_size=input_size, max_steps=100)]
        elif model_type == 'lstm':
            models = [LSTM(h=forecast_horizon, input_size=input_size, max_steps=100)]
            
        fcst = NeuralForecast(
            models=models,
            freq=train_params.get('freq', 'h')
        )
        
        # NeuralForecast fit expects: unique_id, ds, y
        logger.info(f"Fitting {model_type} model...")
        fcst.fit(df=mlf_df)
        
    else:
        # Standard MLForecast for Tree-based models
        if model_type == 'xgboost':
            model = XGBRegressor()
        elif model_type == 'lightgbm':
            model = LGBMRegressor()
        else:
            model = RandomForestRegressor()
    
        # Initialize MLForecast
        fcst = MLForecast(
            models={model_name: model},
            freq=train_params.get('freq', 'h'),
            lags=train_params.get('lags', [1, 7]),
        )
    
        # Fit Model
        lags = train_params.get('lags', [1, 7])
        logger.info(f"Fitting {model_type} model with lags {lags}...")
        
        # MLForecast needs static_features usually specified
        # If we leave them in mlf_df, it treats them as dynamic exogenous if not careful,
        # UNLESS they are constant per unique_id, then it might auto-detect?
        # Better to be explicit:
        fcst.fit(mlf_df, static_features=static_features if static_features else [])

    # Save local artifact
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", f"{model_name}.pkl")
    with open(model_save_path, 'wb') as f:
        pickle.dump(fcst, f)
    
    # MLflow tracking
    mlflow.set_tracking_uri(settings.params['mlflow']['tracking_uri'])
    mlflow.set_experiment(settings.params['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        mlflow.log_params(train_params)
        
        # Log model with custom ForecastWrapper
        artifacts = {"model_path": model_save_path}
        
        # Signature inference
        try:
            # We already have unique_id in mlf_df
            sample_ids = mlf_df['unique_id'].unique()
            sample_id = sample_ids[0]
            sample_input = mlf_df[mlf_df['unique_id'] == sample_id].tail(2).copy()
            
            # Predict
            sample_prediction = fcst.predict(h=1)

            # Input for signature: drop 'y' if exists
            inference_input = sample_input.drop(columns=['y']) if 'y' in sample_input.columns else sample_input
            
            # MLflow signature inference
            signature = infer_signature(inference_input, sample_prediction)
        except Exception as e:
            logger.warning(f"MLflow signature inference encountered an error: {e}. Proceeding without signature.")
            signature = None

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ForecastWrapper(),
            artifacts=artifacts,
            signature=signature,
            registered_model_name=settings.params['mlflow']['model_registry_name']
        )
        
        logger.info(f"Training complete. Model registered as {settings.params['mlflow']['model_registry_name']}")

if __name__ == "__main__":
    train_model()
