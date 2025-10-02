"""
Transformer-based Time Series Prediction Model
Implements state-of-the-art attention mechanisms for stock price prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import MinMaxScaler
import math
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights

class TransformerBlock(nn.Module):
    """
    Single transformer block with multi-head attention and feed-forward network
    """
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class StockTransformer(nn.Module):
    """
    Transformer model for stock price prediction
    """
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 1024,
                 max_seq_length: int = 100,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super(StockTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        attention_weights = []
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)
        
        # Use the last time step for prediction
        output = self.output_projection(x[:, -1, :])
        
        return output, attention_weights

class TransformerPredictor:
    """
    Wrapper class for transformer-based stock prediction
    """
    def __init__(self, 
                 input_dim: int = 38,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 sequence_length: int = 60,
                 learning_rate: float = 0.001,
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        
        # Initialize model
        self.model = StockTransformer(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=sequence_length
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = MinMaxScaler()
        
        logger.info(f"TransformerPredictor initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for transformer training
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            tgt = target[i + self.sequence_length]
            
            sequences.append(seq)
            targets.append(tgt)
        
        return (torch.FloatTensor(sequences).to(self.device),
                torch.FloatTensor(targets).to(self.device))
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the transformer model
        """
        logger.info(f"Training transformer model for {epochs} epochs")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        
        # Split into train/validation
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        train_losses = []
        val_losses = []
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                self.optimizer.zero_grad()
                
                outputs, _ = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _ = self.model(X_val)
                    val_loss = self.criterion(val_outputs.squeeze(), y_val).item()
                    val_losses.append(val_loss)
                self.model.train()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        logger.info("Training completed")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Make predictions using the transformer model
        """
        self.model.eval()
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Get the last sequence
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros if insufficient data
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        last_sequence = X_scaled[-self.sequence_length:]
        
        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                output, attention_weights = self.model(current_sequence)
                pred = output.cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # For multi-step prediction, we would need to update the sequence
                # This is a simplified version for single-step prediction
                break
        
        return np.array(predictions)
    
    def get_attention_weights(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Get attention weights for interpretability
        """
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        
        if len(X_scaled) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        last_sequence = X_scaled[-self.sequence_length:]
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, attention_weights = self.model(sequence_tensor)
        
        # Convert to numpy arrays
        attention_arrays = []
        for attn in attention_weights:
            attention_arrays.append(attn.cpu().numpy())
        
        return attention_arrays
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_dim': self.input_dim,
                'sequence_length': self.sequence_length,
                'device': self.device
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        logger.info(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Test the transformer model
    logger.info("Testing Transformer Model...")
    
    # Create dummy data
    sequence_length = 60
    input_dim = 38
    num_samples = 1000
    
    X_dummy = np.random.randn(num_samples, input_dim)
    y_dummy = np.random.randn(num_samples)
    
    # Initialize and train
    predictor = TransformerPredictor(
        input_dim=input_dim,
        sequence_length=sequence_length,
        d_model=128,
        num_heads=4,
        num_layers=2
    )
    
    # Train on dummy data
    history = predictor.train(X_dummy, y_dummy, epochs=10, batch_size=16)
    
    # Make predictions
    test_X = X_dummy[-sequence_length:]
    predictions = predictor.predict(test_X, steps=1)
    
    logger.info(f"Test prediction: {predictions}")
    logger.info("Transformer model test completed successfully!")
