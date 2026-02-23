"""
Multi-object tracking using DETR-like architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import uuid

from ....core.base_module import BaseModule


@dataclass
class TrackedObject:
    """Tracked object information."""
    track_id: str
    position: np.ndarray
    velocity: np.ndarray
    predicted_position: np.ndarray
    confidence: float
    age: int  # Frames since first detection
    hits: int  # Successful detections
    misses: int  # Consecutive misses
    class_id: int = 0
    size: float = 1.0


class TransformerEncoder(nn.Module):
    """Transformer encoder for detection features."""
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(src, src_key_padding_mask=mask)


class TransformerDecoder(nn.Module):
    """Transformer decoder for object queries."""
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(tgt, memory)


class MultiObjectTracker(nn.Module):
    """
    DETR-like multi-object tracker.
    Uses learnable object queries to detect and track objects.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_queries: int = 100,
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
        )
        
        # Learnable object queries
        self.object_queries = nn.Parameter(torch.randn(num_queries, d_model))
        
        # Decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
        )
        
        # Output heads
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no object
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 3),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 3),
        )
        self.size_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(
        self,
        detections: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            detections: Detection features [batch, num_detections, input_dim]
            mask: Padding mask [batch, num_detections]
            
        Returns:
            Dictionary of predictions
        """
        batch_size = detections.shape[0]
        
        # Project input
        src = self.input_proj(detections)
        
        # Encode
        memory = self.encoder(src, mask)
        
        # Decode with object queries
        queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        outputs = self.decoder(queries, memory)
        
        # Predict
        class_logits = self.class_head(outputs)
        positions = self.position_head(outputs)
        velocities = self.velocity_head(outputs)
        sizes = self.size_head(outputs).squeeze(-1)
        
        return {
            "class_logits": class_logits,
            "class_probs": F.softmax(class_logits, dim=-1),
            "positions": positions,
            "velocities": velocities,
            "sizes": sizes,
        }


class KalmanTracker:
    """Simple Kalman filter for track smoothing and prediction."""
    
    def __init__(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray = None,
    ):
        self.pos = initial_pos.copy()
        self.vel = initial_vel if initial_vel is not None else np.zeros(3)
        
        # State covariance
        self.P = np.eye(6) * 10
        
        # Process noise
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise
        self.R = np.eye(3) * 1.0
    
    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Predict next position."""
        # Simple constant velocity model
        self.pos = self.pos + self.vel * dt
        
        # Update covariance
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        self.P = F @ self.P @ F.T + self.Q
        
        return self.pos.copy()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with measurement."""
        # Measurement matrix
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        # State vector
        x = np.concatenate([self.pos, self.vel])
        
        # Innovation
        y = measurement - self.pos
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        x = x + K @ y
        self.pos = x[:3]
        self.vel = x[3:]
        
        # Update covariance
        self.P = (np.eye(6) - K @ H) @ self.P
        
        return self.pos.copy()


class TrackManager:
    """Manages active tracks."""
    
    def __init__(
        self,
        max_misses: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[str, Dict] = {}
        self.next_id = 0
    
    def update(
        self,
        detections: List[Dict],
        dt: float = 1.0,
    ) -> List[TrackedObject]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with position, velocity, etc.
            dt: Time step
            
        Returns:
            List of confirmed tracks
        """
        # Predict existing tracks
        for track_id, track in self.tracks.items():
            track["predicted_pos"] = track["kalman"].predict(dt)
            track["misses"] += 1
        
        # Associate detections with tracks
        if detections and self.tracks:
            associations = self._associate(detections)
            
            for det_idx, track_id in associations.items():
                det = detections[det_idx]
                track = self.tracks[track_id]
                
                # Update track
                track["kalman"].update(det["position"])
                track["position"] = track["kalman"].pos.copy()
                track["velocity"] = track["kalman"].vel.copy()
                track["confidence"] = det.get("confidence", 0.5)
                track["size"] = det.get("size", track["size"])
                track["class_id"] = det.get("class_id", track["class_id"])
                track["hits"] += 1
                track["misses"] = 0
                track["age"] += 1
            
            # Create new tracks for unassociated detections
            associated_dets = set(associations.keys())
            for i, det in enumerate(detections):
                if i not in associated_dets:
                    self._create_track(det)
        elif detections:
            # No existing tracks, create all
            for det in detections:
                self._create_track(det)
        
        # Remove dead tracks
        dead_ids = [
            tid for tid, t in self.tracks.items()
            if t["misses"] > self.max_misses
        ]
        for tid in dead_ids:
            del self.tracks[tid]
        
        # Return confirmed tracks
        confirmed = []
        for track_id, track in self.tracks.items():
            if track["hits"] >= self.min_hits:
                confirmed.append(TrackedObject(
                    track_id=track_id,
                    position=track["position"],
                    velocity=track["velocity"],
                    predicted_position=track["predicted_pos"],
                    confidence=track["confidence"],
                    age=track["age"],
                    hits=track["hits"],
                    misses=track["misses"],
                    class_id=track["class_id"],
                    size=track["size"],
                ))
        
        return confirmed
    
    def _associate(
        self,
        detections: List[Dict],
    ) -> Dict[int, str]:
        """Associate detections with tracks using distance."""
        associations = {}
        
        det_positions = np.array([d["position"] for d in detections])
        track_ids = list(self.tracks.keys())
        track_positions = np.array([
            self.tracks[tid]["predicted_pos"] for tid in track_ids
        ])
        
        # Distance matrix
        distances = np.linalg.norm(
            det_positions[:, None, :] - track_positions[None, :, :],
            axis=-1
        )
        
        # Greedy assignment (could use Hungarian algorithm for optimal)
        used_tracks = set()
        for det_idx in np.argsort(distances.min(axis=1)):
            track_idx = np.argmin(distances[det_idx])
            
            if track_ids[track_idx] in used_tracks:
                continue
            
            if distances[det_idx, track_idx] < 10.0:  # Distance threshold
                associations[det_idx] = track_ids[track_idx]
                used_tracks.add(track_ids[track_idx])
        
        return associations
    
    def _create_track(self, detection: Dict) -> str:
        """Create new track."""
        track_id = str(uuid.uuid4())[:8]
        
        self.tracks[track_id] = {
            "kalman": KalmanTracker(
                detection["position"],
                detection.get("velocity", None),
            ),
            "position": detection["position"].copy(),
            "velocity": detection.get("velocity", np.zeros(3)),
            "predicted_pos": detection["position"].copy(),
            "confidence": detection.get("confidence", 0.5),
            "size": detection.get("size", 1.0),
            "class_id": detection.get("class_id", 0),
            "hits": 1,
            "misses": 0,
            "age": 1,
        }
        
        return track_id
    
    def get_tracks(self) -> List[TrackedObject]:
        """Get all confirmed tracks."""
        return [
            TrackedObject(
                track_id=tid,
                position=t["position"],
                velocity=t["velocity"],
                predicted_position=t["predicted_pos"],
                confidence=t["confidence"],
                age=t["age"],
                hits=t["hits"],
                misses=t["misses"],
                class_id=t["class_id"],
                size=t["size"],
            )
            for tid, t in self.tracks.items()
            if t["hits"] >= self.min_hits
        ]


class TrackingModule(BaseModule):
    """Complete tracking module."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="object_tracking",
            config=config,
            device=device,
        )
        
        self.detector = None
        self.track_manager = TrackManager(
            max_misses=config.get("max_misses", 5),
            min_hits=config.get("min_hits", 3),
        )
    
    def _build_model(self) -> nn.Module:
        self.detector = MultiObjectTracker(
            input_dim=self.config.get("input_dim", 6),
            d_model=self.config.get("d_model", 256),
            num_queries=self.config.get("num_queries", 100),
            num_classes=self.config.get("num_classes", 10),
        ).to(self.device)
        
        return self.detector
    
    def _preprocess(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Convert detections to tensor."""
        detections = inputs.get("detections", [])
        
        if not detections:
            return torch.zeros(1, 1, 6, device=self.device)
        
        data = []
        for det in detections:
            pos = np.asarray(det.get("position", [0, 0, 0]))
            vel = np.asarray(det.get("velocity", [0, 0, 0]))
            data.append(np.concatenate([pos, vel]))
        
        return torch.FloatTensor(np.array(data)).unsqueeze(0).to(self.device)
    
    def _postprocess(self, outputs: List[TrackedObject]) -> Dict[str, Any]:
        return {
            "tracks": [
                {
                    "id": t.track_id,
                    "position": t.position.tolist(),
                    "velocity": t.velocity.tolist(),
                    "predicted_position": t.predicted_position.tolist(),
                    "confidence": t.confidence,
                    "age": t.age,
                    "class_id": t.class_id,
                    "size": t.size,
                }
                for t in outputs
            ],
            "num_tracks": len(outputs),
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        detections_tensor = self._preprocess(inputs)
        dt = inputs.get("dt", 1.0)
        
        with torch.no_grad():
            outputs = self.detector(detections_tensor)
        
        # Convert NN outputs to detections
        positions = outputs["positions"].cpu().numpy().squeeze()
        velocities = outputs["velocities"].cpu().numpy().squeeze()
        class_probs = outputs["class_probs"].cpu().numpy().squeeze()
        sizes = outputs["sizes"].cpu().numpy().squeeze()
        
        # Filter by confidence
        detections = []
        for i in range(len(positions)):
            conf = class_probs[i, :-1].max()  # Exclude "no object" class
            if conf > 0.3:
                detections.append({
                    "position": positions[i],
                    "velocity": velocities[i],
                    "confidence": conf,
                    "class_id": class_probs[i, :-1].argmax(),
                    "size": sizes[i],
                })
        
        # Update tracks
        tracks = self.track_manager.update(detections, dt)
        
        return self._postprocess(tracks)
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Direct tracking without neural network."""
        detections = inputs.get("detections", [])
        dt = inputs.get("dt", 1.0)
        
        det_list = [
            {
                "position": np.asarray(d.get("position", [0, 0, 0])),
                "velocity": np.asarray(d.get("velocity", [0, 0, 0])),
                "confidence": d.get("confidence", 0.5),
                "class_id": d.get("class_id", 0),
                "size": d.get("size", 1.0),
            }
            for d in detections
        ]
        
        tracks = self.track_manager.update(det_list, dt)
        result = self._postprocess(tracks)
        result["fallback"] = True
        return result
