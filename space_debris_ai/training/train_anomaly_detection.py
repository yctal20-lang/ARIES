"""
Training script for Anomaly Detection module (LSTM Autoencoder).
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from space_debris_ai.models.level2_safety.anomaly_detection import LSTMAutoencoder, AnomalyDetector
from space_debris_ai.simulation import OrbitalEnv, EnvConfig


class TelemetryDataset(Dataset):
    """Dataset for telemetry sequences."""
    
    def __init__(self, sequences, seq_len=50):
        """
        Args:
            sequences: List of telemetry sequences [num_samples, seq_len, features]
            seq_len: Sequence length
        """
        self.sequences = sequences
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Pad or truncate to seq_len
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), seq.shape[1]))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:self.seq_len]
        
        return torch.FloatTensor(seq)


def generate_training_data(num_episodes=100, seq_len=50, seed=42):
    """Generate telemetry data from simulation."""
    np.random.seed(seed)
    
    env_config = EnvConfig()
    env = OrbitalEnv(env_config)
    
    sequences = []
    
    print("Generating training data...")
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=seed + episode)
        episode_data = []
        
        for step in range(1000):
            # Sample random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # Extract telemetry features (position, velocity, attitude, etc.)
            # obs contains: position(3), velocity(3), attitude(4), angular_vel(3), fuel, battery, etc.
            features = obs[:12]  # First 12 features
            episode_data.append(features)
            
            if done or truncated:
                break
        
        if len(episode_data) >= seq_len:
            sequences.append(np.array(episode_data))
    
    env.close()
    return sequences


def train_anomaly_detection(
    input_dim: int = 12,
    seq_len: int = 50,
    hidden_dim: int = 64,
    num_layers: int = 2,
    latent_dim: int = 32,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    num_epochs: int = 100,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints/anomaly_detection",
    seed: int = 42,
):
    """
    Train anomaly detection autoencoder.
    
    Args:
        input_dim: Input feature dimension
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        latent_dim: Latent dimension
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
        device: Compute device
        checkpoint_dir: Checkpoint directory
        seed: Random seed
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    sequences = generate_training_data(num_episodes=100, seq_len=seq_len, seed=seed)
    
    # Create dataset and dataloader
    dataset = TelemetryDataset(sequences, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = LSTMAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"Training anomaly detection model for {num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            
            # Forward pass
            reconstructed, latent = model(batch)
            
            # Compute loss
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved best model (loss: {avg_loss:.6f})")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anomaly Detection Model")
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/anomaly_detection")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_anomaly_detection(
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

