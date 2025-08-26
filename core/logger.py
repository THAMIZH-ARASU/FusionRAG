import structlog
import logging
import sys
from typing import Any, Dict
from contextlib import contextmanager
import time

def setup_logging(log_level: str = "INFO"):
    """Setup structured logging with metrics tracking."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

class RAGLogger:
    def __init__(self, component: str):
        self.logger = structlog.get_logger(component)
        self.component = component
    
    @contextmanager
    def log_operation(self, operation: str, **kwargs):
        """Context manager for logging operations with timing and metrics."""
        start_time = time.time()
        self.logger.info(f"{operation}_started", component=self.component, **kwargs)
        
        try:
            yield self.logger
            duration = time.time() - start_time
            self.logger.info(
                f"{operation}_completed",
                component=self.component,
                duration=duration,
                **kwargs
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"{operation}_failed",
                component=self.component,
                duration=duration,
                error=str(e),
                **kwargs
            )
            raise