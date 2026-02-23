"""
Training script for State Prediction module (TCN with Physics-Informed Loss).
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

from space_debris_ai.models.level3_mission_critical.state_prediction import (
    TCNPredictor,
    PhysicsInformedLoss,
    StatePredictionModule,
)
from space_debris_ai.simulation import OrbitalEnv, EnvConfig


class StateSequenceDataset(Dataset):
    """Dataset for state sequences."""
    
    def __init__(self, sequences, prediction_horizon=10):
        """
        Args:
            sequences: List of state sequences [num_samples, seq_len, state_dim]
            prediction_horizon: Prediction horizon
        """
        self.sequences = sequences
        self.prediction_horizon = prediction_horizon
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input: all but last prediction_horizon steps
        # Target: last prediction_horizon steps
        input_seq = seq[:-self.prediction_horizon]
        target_seq = seq[-self.prediction_horizon:]
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)


def generate_training_data(num_episodes=100, seq_len=100, seed=42):
    """Generate state sequence data from simulation."""
    np.random.seed(seed)
    
    env_config = EnvConfig()
    env = OrbitalEnv(env_config)
    
    sequences = []
    
    print("Generating training data...")
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset(seed=seed + episode)
        episode_data = []
        
        for step in range(seq_len):
            # Sample random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # Extract state (position, velocity, attitude, angular velocity)
            state = obs[:13]  # 13D state
            episode_data.append(state)
            
            if done or truncated:
                break
        
        if len(episode_data) >= seq_len:
            sequences.append(np.array(episode_data))
    
    env.close()
    return sequences


def train_state_prediction(
    input_dim: int = 13,
    output_dim: int = 10,
    prediction_horizon: int = 10,
    num_channels: list = [64, 128, 256],
    kernel_size: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    num_epochs: int = 100,
    physics_weight: float = 0.1,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints/state_prediction",
    seed: int = 42,
):
    """
    Train state prediction model.
    
    Args:
        input_dim: Input state dimension
        output_dim: Output state dimension
        prediction_horizon: Prediction horizon
        num_channels: TCN channel sizes
        kernel_size: TCN kernel size
        dropout: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
        physics_weight: Weight for physics-informed loss
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
    sequences = generate_training_data(num_episodes=100, seq_len=100, seed=seed)
    
    # Create dataset and dataloader
    dataset = StateSequenceDataset(sequences, prediction_horizon=prediction_horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = TCNPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        prediction_horizon=prediction_horizon,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    physics_loss = PhysicsInformedLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Training state prediction model for {num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for input_seq, target_seq in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Forward pass
            predictions, uncertainties = model(input_seq)
            
            # Compute losses
            mse = mse_loss(predictions, target_seq)
            physics = physics_loss(predictions, target_seq)
            
            loss = mse + physics_weight * physics
            
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
    parser = argparse.ArgumentParser(description="Train State Prediction Model")
    parser.add_argument("--input-dim", type=int, default=13)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--prediction-horizon", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/state_prediction")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_state_prediction(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        prediction_horizon=args.prediction_horizon,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        physics_weight=args.physics_weight,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

