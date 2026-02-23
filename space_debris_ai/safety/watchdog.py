"""
Watchdog timers for module health monitoring.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class WatchdogState(Enum):
    """Watchdog states."""
    INACTIVE = "inactive"
    RUNNING = "running"
    TRIGGERED = "triggered"
    RECOVERED = "recovered"


@dataclass
class WatchdogStatus:
    """Status of a single watchdog."""
    name: str
    state: WatchdogState
    timeout: float
    last_feed: float
    time_remaining: float
    trigger_count: int
    
    @property
    def is_expired(self) -> bool:
        return self.time_remaining <= 0


class Watchdog:
    """
    Single watchdog timer for a module.
    
    Must be periodically "fed" to prevent timeout.
    On timeout, triggers callback and optionally recovery action.
    """
    
    def __init__(
        self,
        name: str,
        timeout: float,
        on_timeout: Optional[Callable[[], None]] = None,
        on_recovery: Optional[Callable[[], None]] = None,
        auto_recover: bool = True,
    ):
        """
        Initialize watchdog.
        
        Args:
            name: Watchdog/module name
            timeout: Timeout in seconds
            on_timeout: Callback when timeout occurs
            on_recovery: Callback when recovered
            auto_recover: Automatically recover after feed
        """
        self.name = name
        self.timeout = timeout
        self.on_timeout = on_timeout
        self.on_recovery = on_recovery
        self.auto_recover = auto_recover
        
        self.state = WatchdogState.INACTIVE
        self.last_feed = 0.0
        self.trigger_count = 0
        
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start the watchdog."""
        with self._lock:
            self.last_feed = time.time()
            self.state = WatchdogState.RUNNING
            logger.debug(f"Watchdog '{self.name}' started (timeout: {self.timeout}s)")
    
    def stop(self) -> None:
        """Stop the watchdog."""
        with self._lock:
            self.state = WatchdogState.INACTIVE
            logger.debug(f"Watchdog '{self.name}' stopped")
    
    def feed(self) -> None:
        """
        Feed the watchdog (reset timer).
        Call this periodically to prevent timeout.
        """
        with self._lock:
            self.last_feed = time.time()
            
            if self.state == WatchdogState.TRIGGERED and self.auto_recover:
                self.state = WatchdogState.RECOVERED
                logger.info(f"Watchdog '{self.name}' recovered")
                
                if self.on_recovery:
                    try:
                        self.on_recovery()
                    except Exception as e:
                        logger.error(f"Recovery callback failed: {e}")
                
                self.state = WatchdogState.RUNNING
    
    def check(self) -> bool:
        """
        Check if watchdog has timed out.
        
        Returns:
            True if still healthy, False if timed out
        """
        with self._lock:
            if self.state != WatchdogState.RUNNING:
                return self.state != WatchdogState.TRIGGERED
            
            elapsed = time.time() - self.last_feed
            
            if elapsed > self.timeout:
                self._trigger_timeout()
                return False
            
            return True
    
    def _trigger_timeout(self) -> None:
        """Handle timeout event."""
        self.state = WatchdogState.TRIGGERED
        self.trigger_count += 1
        
        logger.warning(
            f"Watchdog '{self.name}' TIMEOUT "
            f"(count: {self.trigger_count})"
        )
        
        if self.on_timeout:
            try:
                self.on_timeout()
            except Exception as e:
                logger.error(f"Timeout callback failed: {e}")
    
    def get_status(self) -> WatchdogStatus:
        """Get current watchdog status."""
        with self._lock:
            elapsed = time.time() - self.last_feed if self.last_feed > 0 else 0
            remaining = max(0, self.timeout - elapsed)
            
            return WatchdogStatus(
                name=self.name,
                state=self.state,
                timeout=self.timeout,
                last_feed=self.last_feed,
                time_remaining=remaining,
                trigger_count=self.trigger_count,
            )


class WatchdogManager:
    """
    Manager for multiple watchdog timers.
    Runs a background thread to check all watchdogs.
    """
    
    def __init__(
        self,
        check_interval: float = 0.1,
        on_any_timeout: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize watchdog manager.
        
        Args:
            check_interval: How often to check watchdogs (seconds)
            on_any_timeout: Callback when any watchdog times out
        """
        self.check_interval = check_interval
        self.on_any_timeout = on_any_timeout
        
        self._watchdogs: Dict[str, Watchdog] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        logger.info("WatchdogManager initialized")
    
    def add_watchdog(
        self,
        name: str,
        timeout: float,
        on_timeout: Optional[Callable[[], None]] = None,
        on_recovery: Optional[Callable[[], None]] = None,
        auto_start: bool = True,
    ) -> Watchdog:
        """
        Add a new watchdog.
        
        Args:
            name: Watchdog name
            timeout: Timeout in seconds
            on_timeout: Timeout callback
            on_recovery: Recovery callback
            auto_start: Start immediately
            
        Returns:
            Created Watchdog instance
        """
        with self._lock:
            watchdog = Watchdog(
                name=name,
                timeout=timeout,
                on_timeout=on_timeout,
                on_recovery=on_recovery,
            )
            
            self._watchdogs[name] = watchdog
            
            if auto_start:
                watchdog.start()
            
            return watchdog
    
    def remove_watchdog(self, name: str) -> None:
        """Remove a watchdog."""
        with self._lock:
            if name in self._watchdogs:
                self._watchdogs[name].stop()
                del self._watchdogs[name]
    
    def feed(self, name: str) -> None:
        """Feed a specific watchdog."""
        with self._lock:
            if name in self._watchdogs:
                self._watchdogs[name].feed()
    
    def feed_all(self) -> None:
        """Feed all watchdogs."""
        with self._lock:
            for watchdog in self._watchdogs.values():
                watchdog.feed()
    
    def start(self) -> None:
        """Start the watchdog manager background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        logger.info("WatchdogManager started")
    
    def stop(self) -> None:
        """Stop the watchdog manager."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        # Stop all watchdogs
        with self._lock:
            for watchdog in self._watchdogs.values():
                watchdog.stop()
        
        logger.info("WatchdogManager stopped")
    
    def _check_loop(self) -> None:
        """Background thread checking loop."""
        while self._running:
            self._check_all()
            time.sleep(self.check_interval)
    
    def _check_all(self) -> None:
        """Check all watchdogs."""
        with self._lock:
            for name, watchdog in self._watchdogs.items():
                if not watchdog.check():
                    if self.on_any_timeout:
                        try:
                            self.on_any_timeout(name)
                        except Exception as e:
                            logger.error(f"on_any_timeout callback failed: {e}")
    
    def get_status(self) -> Dict[str, WatchdogStatus]:
        """Get status of all watchdogs."""
        with self._lock:
            return {
                name: watchdog.get_status()
                for name, watchdog in self._watchdogs.items()
            }
    
    def get_unhealthy(self) -> List[str]:
        """Get list of unhealthy (triggered) watchdogs."""
        with self._lock:
            return [
                name for name, watchdog in self._watchdogs.items()
                if watchdog.state == WatchdogState.TRIGGERED
            ]
    
    @property
    def all_healthy(self) -> bool:
        """Check if all watchdogs are healthy."""
        with self._lock:
            return all(
                watchdog.state != WatchdogState.TRIGGERED
                for watchdog in self._watchdogs.values()
            )


class HeartbeatMonitor:
    """
    Heartbeat monitor for distributed system components.
    More flexible than watchdog for varying update rates.
    """
    
    def __init__(
        self,
        expected_rate: float = 10.0,
        tolerance: float = 0.5,
    ):
        """
        Initialize heartbeat monitor.
        
        Args:
            expected_rate: Expected heartbeat rate (Hz)
            tolerance: Tolerance factor (0-1)
        """
        self.expected_rate = expected_rate
        self.tolerance = tolerance
        
        self._heartbeats: Dict[str, List[float]] = {}
        self._window_size = int(expected_rate * 5)  # 5 seconds of history
        self._lock = threading.Lock()
    
    def heartbeat(self, source: str) -> None:
        """
        Record a heartbeat from a source.
        
        Args:
            source: Source identifier
        """
        with self._lock:
            if source not in self._heartbeats:
                self._heartbeats[source] = []
            
            self._heartbeats[source].append(time.time())
            
            # Keep only recent heartbeats
            if len(self._heartbeats[source]) > self._window_size:
                self._heartbeats[source] = self._heartbeats[source][-self._window_size:]
    
    def get_rate(self, source: str) -> float:
        """
        Get actual heartbeat rate for a source.
        
        Args:
            source: Source identifier
            
        Returns:
            Measured rate in Hz
        """
        with self._lock:
            if source not in self._heartbeats or len(self._heartbeats[source]) < 2:
                return 0.0
            
            beats = self._heartbeats[source]
            duration = beats[-1] - beats[0]
            
            if duration <= 0:
                return 0.0
            
            return (len(beats) - 1) / duration
    
    def is_healthy(self, source: str) -> bool:
        """
        Check if a source is sending heartbeats at expected rate.
        
        Args:
            source: Source identifier
            
        Returns:
            True if healthy
        """
        rate = self.get_rate(source)
        min_rate = self.expected_rate * (1 - self.tolerance)
        max_rate = self.expected_rate * (1 + self.tolerance)
        
        return min_rate <= rate <= max_rate
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all monitored sources."""
        with self._lock:
            return {
                source: {
                    "rate": self.get_rate(source),
                    "healthy": self.is_healthy(source),
                    "last_heartbeat": beats[-1] if beats else 0,
                }
                for source, beats in self._heartbeats.items()
            }
