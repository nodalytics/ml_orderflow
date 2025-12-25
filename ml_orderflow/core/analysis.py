import pandas as pd
import numpy as np
from typing import List, Dict, Any
from trend_analysis.utils.initializer import logger_instance

logger = logger_instance.get_logger()

class MarketAnalyzer:
    """
    Analyzes model inferences to produce human-readable trend classifications
    and momentum/volume impact assessments.
    """
    def __init__(self, thresholds: Dict[str, float] = None):
        # Default thresholds for classifying returns (percent changes)
        self.thresholds = thresholds or {
            "strong_bullish": 0.005,  # 0.5% move
            "bullish": 0.001,         # 0.1% move
            "bearish": -0.001,
            "strong_bearish": -0.005
        }

    def classify_trend(self, predicted_return: float) -> str:
        """
        Classifies a single predicted return into a trend category.
        """
        if predicted_return >= self.thresholds["strong_bullish"]:
            return "Strong Bullish"
        elif predicted_return >= self.thresholds["bullish"]:
            return "Bullish"
        elif predicted_return > self.thresholds["bearish"]:
            return "Neutral"
        elif predicted_return > self.thresholds["strong_bearish"]:
            return "Bearish"
        else:
            return "Strong Bearish"

    def analyze_results(self, inferences_df: pd.DataFrame, model_name: str) -> List[Dict[str, Any]]:
        """
        Processes a DataFrame of inferences and returns a list of detailed analysis objects.
        Expected columns: unique_id, ds, [model_name]
        """
        analysis_report = []
        
        for _, row in inferences_df.iterrows():
            symbol = row['unique_id']
            prediction = row[model_name]
            
            trend = self.classify_trend(prediction)
            
            # Simple momentum impact (strength of the prediction relative to a "Neutral" state)
            momentum_impact = abs(prediction) * 1000  # Scaling for readability
            
            # Note: Volume impact would ideally compare predicted move vs avg volume
            # For now, we scale it by a constant for placeholder logic as requested
            volume_impact = "Moderate" if 0.001 <= abs(prediction) <= 0.005 else ("High" if abs(prediction) > 0.005 else "Low")

            analysis_report.append({
                "symbol": symbol,
                "timestamp": row['ds'],
                "predicted_return": round(prediction, 6),
                "trend": trend,
                "momentum_score": round(momentum_impact, 2),
                "volatility_impact": volume_impact
            })
            
        return analysis_report

    def generate_summary(self, analysis_report: List[Dict[str, Any]]) -> str:
        """
        Generates a text summary for LLM consumption or console logging.
        """
        summary = "Market Analysis Summary:\n"
        for item in analysis_report:
            summary += (f"- {item['symbol']}: {item['trend']} (MOM: {item['momentum_score']}, VOL: {item['volatility_impact']}) "
                        f"Predicted: {item['predicted_return']*100:.4f}%\n")
        return summary
