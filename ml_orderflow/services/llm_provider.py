import os
import abc
from typing import Dict, Any, Optional
import google.generativeai as genai
from trend_analysis.utils.initializer import logger_instance

logger = logger_instance.get_logger()

class LLMProvider(abc.ABC):
    """
    Abstract Base Class for LLM Providers.
    Ensures a consistent interface for swapping providers (Gemini, OpenAI, etc).
    """
    
    @abc.abstractmethod
    def generate_content(self, prompt: str) -> str:
        """
        Generates text content based on the provided prompt.
        """
        pass

class GeminiProvider(LLMProvider):
    """
    Implementation for Google's Gemini API.
    """
    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. GeminiProvider will fail if called.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def generate_content(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: GEMINI_API_KEY not configured."
            
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error generating summary: {str(e)}"

class DummyProvider(LLMProvider):
    """
    Dummy provider for testing without API keys.
    """
    def generate_content(self, prompt: str) -> str:
        return "This is a simulated market summary. The trend appears bullish based on the provided data."

def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    provider_type = config.get("provider", "dummy")
    
    if provider_type == "gemini":
        return GeminiProvider(
            model_name=config.get("model", "gemini-pro"),
            api_key=os.getenv("GEMINI_API_KEY") # Prioritize env var
        )
    elif provider_type == "dummy":
        return DummyProvider()
    else:
        logger.warning(f"Unknown provider {provider_type}, falling back to Dummy.")
        return DummyProvider()
