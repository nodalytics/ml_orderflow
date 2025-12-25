import mlflow
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

class ForecastWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for MLForecast models.
    Handles data formatting and ensures time series consistency during inference.
    """
    def load_context(self, context):
        import pickle
        with open(context.artifacts["model_path"], 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: Union[pd.DataFrame, Dict[str, Any]]) -> Any:
        # If input is a dictionary, convert to DataFrame
        if isinstance(model_input, dict):
            df = pd.DataFrame([model_input])
        else:
            df = model_input.copy()

        # Ensure required columns for mlforecast (unique_id, ds)
        if 'time' in df.columns and 'ds' not in df.columns:
            df.rename(columns={'time': 'ds'}, inplace=True)
        if 'symbol' in df.columns and 'unique_id' not in df.columns:
            df.rename(columns={'symbol': 'unique_id'}, inplace=True)

        # mlforecast expects 'ds' to be datetime
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])

        # Basic prediction using the loaded MLForecast object
        try:
            # Check for NeuralForecast
            # Functionally, we check if it has a 'predict' method that accepts 'df' but not 'h' needed?
            # Or inspect class name string to avoid import dependency if possible, or just try/except.
            # But safer to check attributes. MLForecast has 'ts', NeuralForecast has 'models'.
            
            # Simplified detection: MLForecast objects usually require 'h' in predict (or have defaults).
            # NeuralForecast objects predict based on internal 'h' set at init.
            
            # Let's try to detect if it's NeuralForecast by class name or attribute
            model_class = self.model.__class__.__name__
            
            if 'NeuralForecast' in model_class:
                # NeuralForecast requires 'df' argument usually
                # And it outputs pandas DF directly
                forecasts = self.model.predict(df=df)
            else:
                # MLForecast
                # We predict the next period (h=1) for the given input
                # If X_df is provided, mlforecast uses it for exogenous features
                forecasts = self.model.predict(h=1, X_df=df)
                
            return forecasts.to_dict(orient='records')
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
