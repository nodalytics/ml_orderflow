import os
import abc
from typing import Dict, Any, Optional, Iterator
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ml_orderflow.utils.initializer import logger_instance

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
    
    @abc.abstractmethod
    def generate_content_stream(self, prompt: str) -> Iterator[str]:
        """
        Generates text content as a stream of tokens.
        """
        pass
    
    @abc.abstractmethod
    def health_check(self) -> bool:
        """
        Check if the provider is healthy and can accept requests.
        """
        pass

class GeminiProvider(LLMProvider):
    """
    Implementation for Google's Gemini API.
    """
    def __init__(
        self, 
        model_name: str = "gemini-pro", 
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. GeminiProvider will fail if called.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens
                }
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate_content(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: GEMINI_API_KEY not configured."
            
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    def generate_content_stream(self, prompt: str) -> Iterator[str]:
        """Generate content as a stream of tokens"""
        if not self.api_key:
            yield "Error: GEMINI_API_KEY not configured."
            return
            
        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield f"Error generating summary: {str(e)}"
    
    def health_check(self) -> bool:
        """Check if Gemini API is accessible"""
        if not self.api_key:
            return False
        try:
            # Simple test generation
            response = self.model.generate_content("test")
            return bool(response.text)
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False

class OpenAIProvider(LLMProvider):
    """
    Implementation for OpenAI's GPT models.
    """
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. OpenAIProvider will fail if called.")
        else:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai package not installed. Run: uv add openai")
                self.client = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate_content(self, prompt: str) -> str:
        if not self.api_key or not self.client:
            return "Error: OPENAI_API_KEY not configured or openai package not installed."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Log token usage
            usage = response.usage
            logger.info(
                f"OpenAI tokens used: {usage.total_tokens} "
                f"(prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})"
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def generate_content_stream(self, prompt: str) -> Iterator[str]:
        """Generate content as a stream of tokens"""
        if not self.api_key or not self.client:
            yield "Error: OPENAI_API_KEY not configured or openai package not installed."
            return
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            yield f"Error generating summary: {str(e)}"
    
    def health_check(self) -> bool:
        """Check if OpenAI API is accessible"""
        if not self.api_key or not self.client:
            return False
        try:
            # Simple test generation
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

class DummyProvider(LLMProvider):
    """
    Dummy provider for testing without API keys.
    """
    def generate_content(self, prompt: str) -> str:
        return "This is a simulated market summary. The trend appears bullish based on the provided data."
    
    def generate_content_stream(self, prompt: str) -> Iterator[str]:
        """Simulate streaming response"""
        message = "This is a simulated market summary. The trend appears bullish based on the provided data."
        words = message.split()
        for word in words:
            yield word + " "
    
    def health_check(self) -> bool:
        """Dummy provider is always healthy"""
        return True

def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    provider_type = config.get("provider", "dummy")
    
    if provider_type == "gemini":
        return GeminiProvider(
            model_name=config.get("model", "gemini-pro"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=config.get("temperature", 0.7),
            max_output_tokens=config.get("max_output_tokens", 1024)
        )
    elif provider_type == "openai":
        return OpenAIProvider(
            model_name=config.get("openai_model", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_output_tokens", 1024)
        )
    elif provider_type == "dummy":
        return DummyProvider()
    else:
        logger.warning(f"Unknown provider {provider_type}, falling back to Dummy.")
        return DummyProvider()

