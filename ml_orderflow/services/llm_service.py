import json
import pandas as pd
from typing import List, Dict, Any
from trend_analysis.services.llm_provider import get_llm_provider
from trend_analysis.utils.config import settings
from trend_analysis.utils.initializer import logger_instance

logger = logger_instance.get_logger()

class LLMService:
    def __init__(self):
        llm_config = settings.params.get('llm', {})
        self.provider = get_llm_provider(llm_config)
    
    def generate_market_summary(self, inferences: List[Dict[str, Any]], symbol: str = None) -> str:
        """
        Generates a natural language summary of the market predictions.
        """
        if not inferences:
            return "No data available to analyze."
            
        # Filter for symbol if requested
        if symbol:
            inferences = [x for x in inferences if x.get('symbol') == symbol]
            if not inferences:
                return f"No predictions found for symbol {symbol}."
        
        # Limit data to prevent token overflow (e.g., last 5 predictions)
        recent_data = inferences[-5:]
        
        # Construct Prompt
        prompt = self._construct_prompt(recent_data, symbol)
        
        # Call LLM
        logger.info(f"Generating summary with {type(self.provider).__name__}...")
        summary = self.provider.generate_content(prompt)
        return summary

    def _construct_prompt(self, data: List[Dict[str, Any]], symbol: str = None) -> str:
        data_str = json.dumps(data, indent=2)
        target = symbol if symbol else "the market"
        
        return f"""
        You are an expert financial analyst. Analyze the following model predictions for {target}.
        
        Data (JSON):
        {data_str}
        
        The 'y_pred' field represents predicted future returns. 
        Positive values indicate a potential uptrend, negative values indicate a downtrend.
        
        Please provide a concise, professional summary of the market outlook based STRICTLY on this data.
        Include:
        1. The overall trend direction (Bullish/Bearish/Neutral).
        2. Key observations from the data points.
        3. A brief risk assessment.
        
        Summary:
        """
