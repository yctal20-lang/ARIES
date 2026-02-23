"""
Message Bus for inter-module communication in Space Debris Collector AI System.
Provides asynchronous, priority-based message passing between modules.
"""

import asyncio
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set
from queue import PriorityQueue
import uuid

from loguru import logger


class MessageType(IntEnum):
    """Message types with priority levels (lower = higher priority)."""
    EMERGENCY = 0       # Immediate action required
    COLLISION = 1       # Collision warning
    SAFETY = 2          # Safety-related messages
    COMMAND = 3         # Control commands
    TELEMETRY = 4       # Sensor data
    STATUS = 5          # Module status updates
    LOG = 6             # Logging messages
    DEBUG = 7           # Debug information


@dataclass(order=True)
class Message:
    """
    Message container for inter-module communication.
    
    Attributes:
        priority: Message priority (lower = higher priority)
        msg_type: Type of message
        source: Source module name
        target: Target module name (None for broadcast)
        payload: Message data
        timestamp: Creation timestamp
        msg_id: Unique message identifier
    """
    priority: int = field(compare=True)
    msg_type: MessageType = field(compare=False)
    source: str = field(compare=False)
    target: Optional[str] = field(compare=False, default=None)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    timestamp: float = field(compare=False, default_factory=time.time)
    msg_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4())[:8])
    
    @classmethod
    def create(
        cls,
        msg_type: MessageType,
        source: str,
        payload: Dict[str, Any],
        target: Optional[str] = None,
    ) -> "Message":
        """Create a message with automatic priority based on type."""
        return cls(
            priority=int(msg_type),
            msg_type=msg_type,
            source=source,
            target=target,
            payload=payload,
        )


# Type alias for message handlers
MessageHandler = Callable[[Message], None]
AsyncMessageHandler = Callable[[Message], asyncio.Future]


class MessageBus:
    """
    Central message bus for module communication.
    
    Features:
    - Priority-based message queue
    - Publish/subscribe pattern
    - Synchronous and asynchronous handlers
    - Message filtering by type and source
    - Thread-safe operations
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the message bus.
        
        Args:
            max_queue_size: Maximum messages in queue
        """
        self._queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        self._subscribers: Dict[str, Set[MessageHandler]] = defaultdict(set)
        self._type_subscribers: Dict[MessageType, Set[MessageHandler]] = defaultdict(set)
        self._async_subscribers: Dict[str, Set[AsyncMessageHandler]] = defaultdict(set)
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Metrics
        self._messages_sent = 0
        self._messages_delivered = 0
        self._messages_dropped = 0
        
        logger.info("MessageBus initialized")
    
    def subscribe(
        self,
        module_name: str,
        handler: MessageHandler,
        msg_types: Optional[List[MessageType]] = None,
    ) -> None:
        """
        Subscribe a module to receive messages.
        
        Args:
            module_name: Name of subscribing module
            handler: Callback function for messages
            msg_types: Optional list of message types to filter
        """
        with self._lock:
            self._subscribers[module_name].add(handler)
            
            if msg_types:
                for msg_type in msg_types:
                    self._type_subscribers[msg_type].add(handler)
            
            logger.debug(f"Module '{module_name}' subscribed to MessageBus")
    
    def subscribe_async(
        self,
        module_name: str,
        handler: AsyncMessageHandler,
    ) -> None:
        """Subscribe with an async handler."""
        with self._lock:
            self._async_subscribers[module_name].add(handler)
    
    def unsubscribe(self, module_name: str, handler: MessageHandler) -> None:
        """
        Unsubscribe a handler.
        
        Args:
            module_name: Module name
            handler: Handler to remove
        """
        with self._lock:
            self._subscribers[module_name].discard(handler)
            for type_handlers in self._type_subscribers.values():
                type_handlers.discard(handler)
    
    def publish(self, message: Message) -> bool:
        """
        Publish a message to the bus.
        
        Args:
            message: Message to publish
            
        Returns:
            True if message was queued successfully
        """
        try:
            self._queue.put_nowait(message)
            self._messages_sent += 1
            
            logger.debug(
                f"Message published: {message.msg_type.name} "
                f"from '{message.source}' to '{message.target or 'broadcast'}'"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            self._messages_dropped += 1
            return False
    
    def publish_sync(
        self,
        msg_type: MessageType,
        source: str,
        payload: Dict[str, Any],
        target: Optional[str] = None,
    ) -> bool:
        """
        Convenience method to create and publish a message.
        
        Args:
            msg_type: Message type
            source: Source module
            payload: Message data
            target: Target module (None for broadcast)
            
        Returns:
            True if successful
        """
        message = Message.create(msg_type, source, payload, target)
        return self.publish(message)
    
    def _deliver_message(self, message: Message) -> int:
        """
        Deliver a message to subscribers.
        
        Args:
            message: Message to deliver
            
        Returns:
            Number of handlers that received the message
        """
        delivered = 0
        
        with self._lock:
            # Get handlers for specific target
            if message.target:
                handlers = self._subscribers.get(message.target, set())
            else:
                # Broadcast: get all handlers
                handlers = set()
                for h_set in self._subscribers.values():
                    handlers.update(h_set)
            
            # Also include type-specific subscribers
            type_handlers = self._type_subscribers.get(message.msg_type, set())
            handlers.update(type_handlers)
        
        # Deliver to each handler
        for handler in handlers:
            try:
                handler(message)
                delivered += 1
            except Exception as e:
                logger.error(f"Handler error processing message {message.msg_id}: {e}")
        
        return delivered
    
    async def _deliver_async(self, message: Message) -> int:
        """Deliver message to async handlers."""
        delivered = 0
        
        with self._lock:
            handlers = set()
            if message.target:
                handlers = self._async_subscribers.get(message.target, set())
            else:
                for h_set in self._async_subscribers.values():
                    handlers.update(h_set)
        
        for handler in handlers:
            try:
                await handler(message)
                delivered += 1
            except Exception as e:
                logger.error(f"Async handler error: {e}")
        
        return delivered
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing messages."""
        logger.info("MessageBus worker started")
        
        while self._running:
            try:
                # Get message with timeout
                try:
                    message = self._queue.get(timeout=0.1)
                except Exception:
                    continue
                
                # Deliver message
                delivered = self._deliver_message(message)
                self._messages_delivered += delivered
                
                # Handle async subscribers if event loop available
                if self._async_loop and self._async_subscribers:
                    asyncio.run_coroutine_threadsafe(
                        self._deliver_async(message),
                        self._async_loop,
                    )
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
        
        logger.info("MessageBus worker stopped")
    
    def start(self, async_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        Start the message bus worker.
        
        Args:
            async_loop: Optional event loop for async handlers
        """
        if self._running:
            logger.warning("MessageBus already running")
            return
        
        self._running = True
        self._async_loop = async_loop
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("MessageBus started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the message bus.
        
        Args:
            timeout: Maximum time to wait for worker to stop
        """
        if not self._running:
            return
        
        self._running = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None
        
        logger.info("MessageBus stopped")
    
    def process_pending(self, max_messages: int = 100) -> int:
        """
        Process pending messages synchronously.
        Useful for testing or single-threaded execution.
        
        Args:
            max_messages: Maximum messages to process
            
        Returns:
            Number of messages processed
        """
        processed = 0
        
        while not self._queue.empty() and processed < max_messages:
            try:
                message = self._queue.get_nowait()
                self._deliver_message(message)
                processed += 1
            except Exception:
                break
        
        return processed
    
    def clear(self) -> None:
        """Clear all pending messages."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        
        logger.debug("MessageBus queue cleared")
    
    @property
    def pending_count(self) -> int:
        """Number of pending messages."""
        return self._queue.qsize()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Message bus statistics."""
        return {
            "running": self._running,
            "pending_messages": self.pending_count,
            "messages_sent": self._messages_sent,
            "messages_delivered": self._messages_delivered,
            "messages_dropped": self._messages_dropped,
            "subscribers": {name: len(handlers) for name, handlers in self._subscribers.items()},
            "type_subscribers": {t.name: len(h) for t, h in self._type_subscribers.items()},
        }
    
    def __repr__(self) -> str:
        return f"MessageBus(running={self._running}, pending={self.pending_count})"


# Global message bus instance (singleton pattern)
_global_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create the global message bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = MessageBus()
    return _global_bus


def reset_message_bus() -> None:
    """Reset the global message bus (for testing)."""
    global _global_bus
    if _global_bus:
        _global_bus.stop()
    _global_bus = None
