import json
import pandas as pd
from typing import List, Dict, Any, Iterator, Optional
from ml_orderflow.services.llm_provider import get_llm_provider
from ml_orderflow.core.cache import LRUCache
from ml_orderflow.utils.config import settings
from ml_orderflow.utils.initializer import logger_instance

logger = logger_instance.get_logger()

class LLMService:
    def __init__(self):
        llm_config = settings.params.get('llm', {})
        self.provider = get_llm_provider(llm_config)
        
        # Initialize cache if enabled
        self.cache_enabled = llm_config.get('cache_enabled', True)
        if self.cache_enabled:
            cache_ttl = llm_config.get('cache_ttl_seconds', 3600)
            self.cache = LRUCache(ttl_seconds=cache_ttl, max_size=100)
            logger.info(f"LLM response caching enabled (TTL: {cache_ttl}s)")
        else:
            self.cache = None
            logger.info("LLM response caching disabled")
        
        # Advanced prompting settings
        self.use_technical_context = llm_config.get('use_technical_context', True)
        self.include_market_regime = llm_config.get('include_market_regime', True)
    
    def generate_market_summary(
        self, 
        inferences: List[Dict[str, Any]], 
        symbol: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generates a natural language summary of the market predictions.
        
        Args:
            inferences: List of inference data dictionaries
            symbol: Optional symbol filter
            use_cache: Whether to use cached responses
        
        Returns:
            Natural language market summary
        """
        if not inferences:
            return "No data available to analyze."
        
        # Filter for symbol if requested
        if symbol:
            inferences = [x for x in inferences if x.get('symbol') == symbol or x.get('unique_id', '').startswith(symbol)]
            if not inferences:
                return f"No predictions found for symbol {symbol}."
        
        # Check cache first
        if use_cache and self.cache:
            cache_key = {"inferences": inferences[-5:], "symbol": symbol}
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.info("Returning cached LLM response")
                return cached_response
        
        # Limit data to prevent token overflow (last 5 predictions)
        recent_data = inferences[-5:]
        
        # Construct Prompt
        prompt = self._construct_advanced_prompt(recent_data, symbol)
        
        # Call LLM
        logger.info(f"Generating summary with {type(self.provider).__name__}...")
        try:
            summary = self.provider.generate_content(prompt)
            
            # Cache the response
            if use_cache and self.cache:
                cache_key = {"inferences": recent_data, "symbol": symbol}
                self.cache.set(cache_key, summary)
            
            return summary
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating summary: {str(e)}"
    
    def generate_market_summary_stream(
        self, 
        inferences: List[Dict[str, Any]], 
        symbol: Optional[str] = None
    ) -> Iterator[str]:
        """
        Generates a streaming natural language summary of market predictions.
        
        Args:
            inferences: List of inference data dictionaries
            symbol: Optional symbol filter
        
        Yields:
            Chunks of the market summary as they're generated
        """
        if not inferences:
            yield "No data available to analyze."
            return
        
        # Filter for symbol if requested
        if symbol:
            inferences = [x for x in inferences if x.get('symbol') == symbol or x.get('unique_id', '').startswith(symbol)]
            if not inferences:
                yield f"No predictions found for symbol {symbol}."
                return
        
        # Limit data
        recent_data = inferences[-5:]
        
        # Construct Prompt
        prompt = self._construct_advanced_prompt(recent_data, symbol)
        
        # Stream from LLM
        logger.info(f"Streaming summary with {type(self.provider).__name__}...")
        try:
            for chunk in self.provider.generate_content_stream(prompt):
                yield chunk
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield f"Error generating summary: {str(e)}"
    
    def _construct_advanced_prompt(self, data: List[Dict[str, Any]], symbol: Optional[str] = None) -> str:
        """
        Construct an advanced prompt with technical context and market regime analysis.
        """
        target = symbol if symbol else "the market"
        
        # Analyze data for technical context
        df = pd.DataFrame(data)
        
        # Extract technical indicators if available
        technical_summary = ""
        if self.use_technical_context and not df.empty:
            technical_summary = self._extract_technical_context(df)
        
        # Detect market regime
        market_regime = ""
        if self.include_market_regime and not df.empty:
            market_regime = self._detect_market_regime(df)
        
        # Prepare data string (limit to essential columns to save tokens)
        essential_cols = ['unique_id', 'ds', 'y_pred'] if 'y_pred' in df.columns else list(df.columns)[:5]
        data_subset = df[essential_cols].to_dict(orient='records') if not df.empty else data
        data_str = json.dumps(data_subset, indent=2, default=str)
        
        prompt = f"""You are an expert financial analyst with deep knowledge of technical analysis and market dynamics.

Analyze the following model predictions for {target}.

{market_regime}

{technical_summary}

Prediction Data (JSON):
{data_str}

The prediction values represent forecasted returns. Positive values indicate potential uptrend, negative values indicate downtrend.

Please provide a concise, professional market summary (max 200 words) that includes:

1. **Overall Trend Direction**: Clearly state if the outlook is Bullish, Bearish, or Neutral
2. **Key Observations**: Highlight the most significant patterns or changes in the predictions
3. **Technical Context**: Reference any notable technical indicators if provided
4. **Risk Assessment**: Briefly assess the confidence level and potential risks
5. **Actionable Insight**: Provide one clear takeaway for traders/investors

Format your response in clear paragraphs. Be direct and avoid unnecessary hedging.

Summary:
"""
        return prompt
    
    def _extract_technical_context(self, df: pd.DataFrame) -> str:
        """Extract technical indicator context from dataframe"""
        context_parts = []
        
        # Check for RSI
        rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
        if rsi_cols:
            latest_rsi = df[rsi_cols[0]].iloc[-1] if not df[rsi_cols[0]].isna().all() else None
            if latest_rsi:
                if latest_rsi > 70:
                    context_parts.append(f"RSI is overbought at {latest_rsi:.1f}")
                elif latest_rsi < 30:
                    context_parts.append(f"RSI is oversold at {latest_rsi:.1f}")
                else:
                    context_parts.append(f"RSI is neutral at {latest_rsi:.1f}")
        
        # Check for volatility
        vol_cols = [col for col in df.columns if 'volatility' in col.lower()]
        if vol_cols:
            latest_vol = df[vol_cols[0]].iloc[-1] if not df[vol_cols[0]].isna().all() else None
            if latest_vol:
                context_parts.append(f"Current volatility: {latest_vol:.4f}")
        
        # Check for regression slope (trend strength)
        slope_cols = [col for col in df.columns if 'slope' in col.lower()]
        if slope_cols:
            latest_slope = df[slope_cols[0]].iloc[-1] if not df[slope_cols[0]].isna().all() else None
            if latest_slope:
                trend_strength = "strong" if abs(latest_slope) > 0.5 else "moderate" if abs(latest_slope) > 0.1 else "weak"
                trend_dir = "upward" if latest_slope > 0 else "downward"
                context_parts.append(f"Trend: {trend_strength} {trend_dir} (slope: {latest_slope:.4f})")
        
        if context_parts:
            return "**Technical Indicators:**\n" + "\n".join(f"- {part}" for part in context_parts)
        return ""
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime from predictions"""
        if 'y_pred' not in df.columns or df['y_pred'].isna().all():
            return ""
        
        predictions = df['y_pred'].dropna()
        if len(predictions) < 2:
            return ""
        
        # Calculate trend consistency
        positive_count = (predictions > 0).sum()
        negative_count = (predictions < 0).sum()
        total = len(predictions)
        
        # Determine regime
        if positive_count / total > 0.7:
            regime = "**Market Regime:** Strong Bullish Momentum"
        elif negative_count / total > 0.7:
            regime = "**Market Regime:** Strong Bearish Momentum"
        elif positive_count / total > 0.55:
            regime = "**Market Regime:** Mild Bullish Bias"
        elif negative_count / total > 0.55:
            regime = "**Market Regime:** Mild Bearish Bias"
        else:
            regime = "**Market Regime:** Consolidation/Ranging"
        
        # Add volatility context
        std_dev = predictions.std()
        if std_dev > predictions.abs().mean():
            regime += " with High Uncertainty"
        
        return regime
    
    def clear_cache(self):
        """Clear the LLM response cache"""
        if self.cache:
            self.cache.clear()
            logger.info("LLM cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of LLM service"""
        provider_healthy = self.provider.health_check()
        
        return {
            "provider": type(self.provider).__name__,
            "provider_healthy": provider_healthy,
            "cache_enabled": self.cache_enabled,
            "cache_stats": self.get_cache_stats() if self.cache else None
        }

