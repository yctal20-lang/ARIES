"""
Base module class for all AI modules in the Space Debris Collector system.
Provides unified interface for training, inference, saving, and loading.
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from loguru import logger


class ModuleState(Enum):
    """Module operational states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ModuleMetrics:
    """Runtime metrics for module monitoring."""
    inference_count: int = 0
    total_inference_time: float = 0.0
    last_inference_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_update: float = field(default_factory=time.time)
    
    @property
    def avg_inference_time(self) -> float:
        """Average inference time in milliseconds."""
        if self.inference_count == 0:
            return 0.0
        return (self.total_inference_time / self.inference_count) * 1000
    
    def record_inference(self, duration: float) -> None:
        """Record an inference execution."""
        self.inference_count += 1
        self.total_inference_time += duration
        self.last_inference_time = duration * 1000  # Convert to ms
        self.last_update = time.time()
    
    def record_error(self, error: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error
        self.last_update = time.time()


class BaseModule(ABC):
    """
    Abstract base class for all neural network modules.
    
    Provides:
    - Unified interface for forward pass, training, saving/loading
    - State management and health monitoring
    - Metrics collection
    - Thread-safe operations
    - Fail-safe integration
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        """
        Initialize the base module.
        
        Args:
            name: Module name for identification
            config: Module-specific configuration
            device: Compute device (cuda/cpu), auto-detected if None
        """
        self.name = name
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._state = ModuleState.UNINITIALIZED
        self._metrics = ModuleMetrics()
        self._lock = threading.RLock()
        self._model: Optional[nn.Module] = None
        self._fallback_enabled = config.get("fallback_enabled", True)
        
        logger.info(f"Initializing module '{name}' on device '{self.device}'")
    
    @property
    def state(self) -> ModuleState:
        """Current module state."""
        return self._state
    
    @property
    def metrics(self) -> ModuleMetrics:
        """Module runtime metrics."""
        return self._metrics
    
    @property
    def model(self) -> Optional[nn.Module]:
        """The underlying PyTorch model."""
        return self._model
    
    @property
    def is_ready(self) -> bool:
        """Check if module is ready for inference."""
        return self._state in (ModuleState.READY, ModuleState.RUNNING)
    
    def _set_state(self, state: ModuleState) -> None:
        """Thread-safe state transition."""
        with self._lock:
            old_state = self._state
            self._state = state
            logger.debug(f"Module '{self.name}': {old_state.value} -> {state.value}")
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build the neural network model.
        Must be implemented by subclasses.
        
        Returns:
            The constructed PyTorch model
        """
        pass
    
    @abstractmethod
    def _preprocess(self, inputs: Dict[str, Any]) -> Any:
        """
        Preprocess inputs before model forward pass.
        
        Args:
            inputs: Raw input data dictionary
            
        Returns:
            Preprocessed tensor(s) ready for the model
        """
        pass
    
    @abstractmethod
    def _postprocess(self, outputs: Any) -> Dict[str, Any]:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs dictionary
        """
        pass
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback classical algorithm when neural network fails.
        Override in subclasses to provide module-specific fallback.
        
        Args:
            inputs: Raw input data
            
        Returns:
            Fallback outputs
        """
        logger.warning(f"Module '{self.name}': Using default fallback (no-op)")
        return {"fallback": True, "error": "No fallback implemented"}
    
    def initialize(self) -> bool:
        """
        Initialize the module and build the model.
        
        Returns:
            True if initialization successful
        """
        try:
            self._set_state(ModuleState.INITIALIZING)
            
            # Build model
            self._model = self._build_model()
            self._model.to(self.device)
            self._model.eval()  # Default to eval mode
            
            # Count parameters
            total_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            logger.info(
                f"Module '{self.name}' built: {total_params:,} params "
                f"({trainable_params:,} trainable)"
            )
            
            self._set_state(ModuleState.READY)
            return True
            
        except Exception as e:
            logger.error(f"Module '{self.name}' initialization failed: {e}")
            self._metrics.record_error(str(e))
            self._set_state(ModuleState.ERROR)
            return False
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute forward pass with preprocessing and postprocessing.
        
        Args:
            inputs: Input data dictionary
            
        Returns:
            Output dictionary with results
        """
        if not self.is_ready:
            if self._fallback_enabled:
                logger.warning(f"Module '{self.name}' not ready, using fallback")
                return self._fallback(inputs)
            raise RuntimeError(f"Module '{self.name}' is not ready (state: {self._state})")
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                self._set_state(ModuleState.RUNNING)
                
                # Preprocess
                processed_inputs = self._preprocess(inputs)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self._model(processed_inputs)
                
                # Postprocess
                results = self._postprocess(outputs)
                
                # Record metrics
                duration = time.perf_counter() - start_time
                self._metrics.record_inference(duration)
                
                self._set_state(ModuleState.READY)
                return results
                
        except Exception as e:
            logger.error(f"Module '{self.name}' forward pass failed: {e}")
            self._metrics.record_error(str(e))
            self._set_state(ModuleState.ERROR)
            
            if self._fallback_enabled:
                logger.info(f"Module '{self.name}': Attempting fallback")
                return self._fallback(inputs)
            raise
    
    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Training batch
            optimizer: PyTorch optimizer
            
        Returns:
            Dictionary of loss values
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        self._model.train()
        optimizer.zero_grad()
        
        # Forward pass
        processed_inputs = self._preprocess(batch)
        outputs = self._model(processed_inputs)
        
        # Compute loss (to be implemented by subclasses)
        loss_dict = self._compute_loss(outputs, batch)
        total_loss = sum(loss_dict.values())
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        self._model.eval()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def _compute_loss(
        self,
        outputs: Any,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        Override in subclasses for custom loss functions.
        
        Args:
            outputs: Model outputs
            batch: Training batch with targets
            
        Returns:
            Dictionary of named losses
        """
        raise NotImplementedError("Subclass must implement _compute_loss")
    
    def save(self, path: Path) -> None:
        """
        Save module state to disk.
        
        Args:
            path: Save path
        """
        if self._model is None:
            raise RuntimeError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "name": self.name,
            "config": self.config,
            "model_state_dict": self._model.state_dict(),
            "metrics": {
                "inference_count": self._metrics.inference_count,
                "error_count": self._metrics.error_count,
            },
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Module '{self.name}' saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load module state from disk.
        
        Args:
            path: Checkpoint path
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if self._model is None:
            self._model = self._build_model()
            self._model.to(self.device)
        
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        
        logger.info(f"Module '{self.name}' loaded from {path}")
        self._set_state(ModuleState.READY)
    
    def reset(self) -> None:
        """Reset module to initial state."""
        with self._lock:
            self._metrics = ModuleMetrics()
            self._set_state(ModuleState.READY if self._model else ModuleState.UNINITIALIZED)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.
        
        Returns:
            Health status dictionary
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "is_ready": self.is_ready,
            "device": self.device,
            "metrics": {
                "inference_count": self._metrics.inference_count,
                "avg_inference_time_ms": self._metrics.avg_inference_time,
                "last_inference_time_ms": self._metrics.last_inference_time,
                "error_count": self._metrics.error_count,
                "last_error": self._metrics.last_error,
            },
            "model_loaded": self._model is not None,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', state={self._state.value})"
