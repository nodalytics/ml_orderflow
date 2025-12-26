"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by monitoring failures and temporarily blocking requests
to failing services, allowing them time to recover.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests are blocked
- HALF_OPEN: Testing if service has recovered
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from ml_orderflow.utils.initializer import logger_instance

logger = logger_instance.get_logger()


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation with configurable thresholds.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Time to wait before attempting recovery (HALF_OPEN)
        half_open_max_calls: Max successful calls in HALF_OPEN before closing
        name: Identifier for this circuit breaker
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.name = name
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={timeout_seconds}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self._last_failure_time is None:
            return False
        return (time.time() - self._last_failure_time) >= self.timeout_seconds
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable. Retry after {self.timeout_seconds}s."
                    )
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self._failure_count = 0
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker '{self.name}' HALF_OPEN success "
                    f"({self._success_count}/{self.half_open_max_calls})"
                )
                
                if self._success_count >= self.half_open_max_calls:
                    logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker '{self.name}' failure "
                f"({self._failure_count}/{self.failure_threshold})"
            )
            
            if self._state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN, reopening")
                self._state = CircuitState.OPEN
                self._success_count = 0
            elif self._failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}' threshold reached, "
                    f"transitioning to OPEN"
                )
                self._state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}' manually reset")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
    
    def get_stats(self) -> dict:
        """Get current circuit breaker statistics"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "failure_threshold": self.failure_threshold,
                "timeout_seconds": self.timeout_seconds
            }


def circuit_breaker(
    failure_threshold: int = 5,
    timeout_seconds: int = 60,
    half_open_max_calls: int = 3,
    name: str = "default"
):
    """
    Decorator to wrap functions with circuit breaker protection.
    
    Usage:
        @circuit_breaker(failure_threshold=3, timeout_seconds=30, name="api_call")
        def fetch_data():
            # ... potentially failing operation
            pass
    """
    cb = CircuitBreaker(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
        half_open_max_calls=half_open_max_calls,
        name=name
    )
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        # Attach circuit breaker instance for external access
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator
