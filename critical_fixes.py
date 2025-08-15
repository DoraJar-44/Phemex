#!/usr/bin/env python3
"""
CRITICAL FIXES FOR TRADING BOT
Apply these fixes immediately to resolve critical issues
"""

import asyncio
import threading
import logging
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager
import httpx
from functools import wraps
import time

# ============================================================================
# FIX 1: Thread-Safe Global State Management
# ============================================================================

class ThreadSafeGlobals:
    """Thread-safe management of global state"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._shutdown_requested = False
        self._tui_instance = None
        self._tui_thread = None
    
    @property
    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_requested
    
    @shutdown_requested.setter
    def shutdown_requested(self, value: bool):
        with self._lock:
            self._shutdown_requested = value
    
    def get_tui_instance(self):
        with self._lock:
            return self._tui_instance
    
    def set_tui_instance(self, instance):
        with self._lock:
            self._tui_instance = instance
    
    def get_tui_thread(self):
        with self._lock:
            return self._tui_thread
    
    def set_tui_thread(self, thread):
        with self._lock:
            self._tui_thread = thread

# Global instance
safe_globals = ThreadSafeGlobals()

# ============================================================================
# FIX 2: Async TUI Manager (Fixes Event Loop Conflict)
# ============================================================================

class AsyncTUIManager:
    """Manages TUI in async-compatible way"""
    
    def __init__(self):
        self.tui = None
        self.executor = None
        self.running = False
    
    async def start(self, tui_class):
        """Start TUI without blocking event loop"""
        if self.running:
            return
        
        self.executor = asyncio.get_event_loop().run_in_executor
        self.tui = tui_class()
        
        # Run TUI in thread pool executor instead of creating new thread
        try:
            await self.executor(None, self._run_tui)
            self.running = True
        except Exception as e:
            logging.error(f"Failed to start TUI: {e}")
            self.tui = None
    
    def _run_tui(self):
        """Run TUI in executor thread"""
        if self.tui:
            self.tui.run()
    
    async def stop(self):
        """Stop TUI gracefully"""
        if self.tui and self.running:
            self.tui.running = False
            self.running = False
            # Give TUI time to cleanup
            await asyncio.sleep(0.5)

# ============================================================================
# FIX 3: Connection Pool Manager
# ============================================================================

class ConnectionPoolManager:
    """Manages HTTP connection pools for better performance"""
    
    def __init__(self, max_connections: int = 100, max_keepalive: int = 20):
        self.transport = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
                keepalive_expiry=30.0
            ),
            retries=3
        )
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(transport=self.transport, timeout=30.0)
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

# ============================================================================
# FIX 4: Resource Cleanup Manager
# ============================================================================

class ResourceManager:
    """Ensures proper cleanup of resources"""
    
    def __init__(self):
        self.resources = []
        self.cleanup_tasks = []
    
    def register_resource(self, resource, cleanup_func):
        """Register a resource for cleanup"""
        self.resources.append((resource, cleanup_func))
    
    async def cleanup_all(self):
        """Clean up all registered resources"""
        for resource, cleanup_func in self.resources:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func(resource)
                else:
                    cleanup_func(resource)
            except Exception as e:
                logging.error(f"Error cleaning up resource: {e}")
        
        self.resources.clear()

# ============================================================================
# FIX 5: Parallel Symbol Scanner
# ============================================================================

async def scan_symbols_parallel(client, symbols: List[str], scan_func, batch_size: int = 10):
    """Scan symbols in parallel batches for better performance"""
    results = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        
        # Create tasks for batch
        tasks = [scan_func(client, symbol) for symbol in batch]
        
        # Execute batch in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        for symbol, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logging.error(f"Error scanning {symbol}: {result}")
            elif result is not None:
                results.append(result)
    
    return results

# ============================================================================
# FIX 6: Circuit Breaker for API Calls
# ============================================================================

class CircuitBreaker:
    """Prevents cascading failures in API calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func):
        """Decorator for circuit breaker protection"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not await self._can_attempt():
                raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                await self._on_success()
                return result
            except Exception as e:
                await self._on_failure()
                raise e
        
        return wrapper
    
    async def _can_attempt(self) -> bool:
        """Check if we can attempt the call"""
        with self._lock:
            if self.state == "closed":
                return True
            
            if self.state == "open":
                if self.last_failure_time and \
                   (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            
            return True  # half-open state
    
    async def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self.failure_count = 0
            if self.state == "half-open":
                self.state = "closed"
    
    async def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"

# ============================================================================
# FIX 7: Memory-Efficient Checkpoint Manager
# ============================================================================

class OptimizedCheckpointManager:
    """Memory-efficient checkpoint management with automatic cleanup"""
    
    def __init__(self, max_checkpoints: int = 10, checkpoint_file: str = "bot_checkpoint.json"):
        self.max_checkpoints = max_checkpoints
        self.checkpoint_file = checkpoint_file
        self.checkpoints = {}
        self._lock = threading.Lock()
        self.load_checkpoints()
    
    def save_checkpoint(self, name: str, data: Dict[str, Any]):
        """Save checkpoint with automatic cleanup of old ones"""
        with self._lock:
            self.checkpoints[name] = {
                "timestamp": time.time(),
                "data": data
            }
            
            # Remove old checkpoints if exceeded max
            if len(self.checkpoints) > self.max_checkpoints:
                # Remove oldest checkpoint
                oldest = min(self.checkpoints.items(), 
                           key=lambda x: x[1]["timestamp"])
                del self.checkpoints[oldest[0]]
            
            self._persist_checkpoints()
    
    def _persist_checkpoints(self):
        """Atomically write checkpoints to disk"""
        import json
        import tempfile
        import os
        
        # Write to temp file first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                        dir=os.path.dirname(self.checkpoint_file)) as tmp:
            json.dump(self.checkpoints, tmp, indent=2)
            temp_name = tmp.name
        
        # Atomic rename
        os.replace(temp_name, self.checkpoint_file)
    
    def load_checkpoints(self):
        """Load checkpoints from disk"""
        import json
        import os
        
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoints = json.load(f)
                
                # Cleanup old checkpoints on load
                if len(self.checkpoints) > self.max_checkpoints:
                    sorted_checkpoints = sorted(
                        self.checkpoints.items(),
                        key=lambda x: x[1]["timestamp"],
                        reverse=True
                    )
                    self.checkpoints = dict(sorted_checkpoints[:self.max_checkpoints])
                    self._persist_checkpoints()
        except Exception as e:
            logging.warning(f"Failed to load checkpoints: {e}")
            self.checkpoints = {}

# ============================================================================
# FIX 8: Rate Limiter for API Calls
# ============================================================================

class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, calls_per_second: float = 10):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = time.time()

# ============================================================================
# FIX 9: Logging Rotation Setup
# ============================================================================

def setup_rotating_logger(name: str = "trading_bot", 
                         max_bytes: int = 10*1024*1024,  # 10MB
                         backup_count: int = 5):
    """Setup logger with automatic rotation"""
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create rotating file handler
    handler = RotatingFileHandler(
        'bot_debug.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Also add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def apply_critical_fixes():
    """Example of how to apply the critical fixes"""
    
    # 1. Setup rotating logger
    logger = setup_rotating_logger()
    
    # 2. Initialize thread-safe globals
    globals_manager = safe_globals
    
    # 3. Setup resource manager
    resource_manager = ResourceManager()
    
    # 4. Initialize optimized checkpoint manager
    checkpoint_manager = OptimizedCheckpointManager(max_checkpoints=20)
    
    # 5. Setup rate limiter
    rate_limiter = RateLimiter(calls_per_second=20)
    
    # 6. Initialize circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    
    # 7. Setup connection pool
    async with ConnectionPoolManager() as http_client:
        logger.info("All critical fixes applied successfully")
        
        # Example: Use parallel scanning
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # etc...
        
        # This would be your actual scan function
        async def mock_scan(client, symbol):
            await rate_limiter.acquire()  # Rate limit
            # Your scanning logic here
            return {"symbol": symbol, "score": 85}
        
        # Scan in parallel with rate limiting
        results = await scan_symbols_parallel(
            http_client, 
            symbols, 
            mock_scan,
            batch_size=10
        )
        
        logger.info(f"Scanned {len(results)} symbols in parallel")
    
    # Cleanup all resources
    await resource_manager.cleanup_all()
    
    logger.info("Critical fixes demonstration complete")

if __name__ == "__main__":
    # Run the example
    asyncio.run(apply_critical_fixes())