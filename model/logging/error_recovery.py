"""
Error Recovery Implementation

Error recovery and retry mechanisms for the BreadthFlow system
including retry patterns, circuit breakers, and fallback strategies.
"""

from typing import Callable, Dict, Any, Optional
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorRecovery:
    """Error recovery and retry mechanisms"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def retry_on_error(self, retryable_errors: list = None):
        """Decorator for retrying operations on specific errors"""
        
        if retryable_errors is None:
            retryable_errors = ['ConnectionError', 'TimeoutError', 'TemporaryError']
        
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        # Check if error is retryable
                        if type(e).__name__ not in retryable_errors:
                            raise e
                        
                        # If this is the last attempt, raise the exception
                        if attempt == self.max_retries:
                            raise e
                        
                        # Calculate backoff delay
                        delay = (self.backoff_factor ** attempt)
                        logger.warning(f"Retry attempt {attempt + 1}/{self.max_retries + 1} after {delay}s for {func.__name__}")
                        time.sleep(delay)
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def circuit_breaker(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Circuit breaker pattern for preventing cascade failures"""
        
        def decorator(func: Callable):
            failure_count = 0
            last_failure_time = None
            circuit_open = False
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal failure_count, last_failure_time, circuit_open
                
                # Check if circuit is open
                if circuit_open:
                    if time.time() - last_failure_time > recovery_timeout:
                        circuit_open = False
                        failure_count = 0
                        logger.info(f"Circuit breaker closed for {func.__name__}")
                    else:
                        raise Exception("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    # Reset failure count on success
                    failure_count = 0
                    return result
                except Exception as e:
                    failure_count += 1
                    last_failure_time = time.time()
                    
                    # Open circuit if threshold reached
                    if failure_count >= failure_threshold:
                        circuit_open = True
                        logger.error(f"Circuit breaker opened for {func.__name__} after {failure_count} failures")
                    
                    raise e
            
            return wrapper
        return decorator
    
    def fallback_strategy(self, fallback_func: Callable):
        """Fallback strategy for when primary operation fails"""
        
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Log the primary failure
                    logger.warning(f"Primary operation failed, using fallback: {e}")
                    
                    # Use fallback
                    return fallback_func(*args, **kwargs)
            
            return wrapper
        return decorator
