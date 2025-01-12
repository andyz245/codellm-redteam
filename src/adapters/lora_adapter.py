import math
import torch
import torch.nn as nn
from typing import Optional, Dict


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation module for efficient fine-tuning."""
    
    def __init__(
        self,
        input_dim: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.1,
        layer_name: str = "default"
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            input_dim: Dimension of input features
            rank: Rank of low-rank approximation
            alpha: Scaling factor
            dropout: Dropout probability
            layer_name: Name of the layer (for tracking)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.rank = rank
        self.alpha = alpha
        self.layer_name = layer_name
        
        # Initialize low-rank matrices
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, input_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Scaling factor
        self.scaling = alpha / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA adapter.
        
        Args:
            x: Input tensor
            
        Returns:
            Adapted tensor
        """
        # Apply low-rank transformation
        result = self.lora_B(
            self.dropout(
                self.lora_A(x)
            )
        )
        return result * self.scaling
        
    def update_weights(
        self,
        gradients: torch.Tensor,
        learning_rate: Optional[float] = None
    ):
        """
        Update adapter weights using provided gradients.
        
        Args:
            gradients: Computed gradients
            learning_rate: Optional learning rate override
        """
        with torch.no_grad():
            if learning_rate is not None:
                # Average gradients across batch and sequence dimensions
                avg_gradients = gradients.mean(dim=[0, 1])  # [hidden_dim]
                
                # Reshape gradients to match weight matrices
                # lora_A.weight: [rank, input_dim]
                # lora_B.weight: [input_dim, rank]
                
                # First reshape to match input dimension
                reshaped_grads = avg_gradients[:self.input_dim]  # [input_dim]
                
                # For lora_A: [rank, input_dim]
                grad_A = reshaped_grads.repeat(self.rank, 1)  # Repeat to match rank dimension
                
                # For lora_B: [input_dim, rank]
                grad_B = reshaped_grads.unsqueeze(1).repeat(1, self.rank)  # Add rank dimension
                
                # Update weights with proper dimensions
                self.lora_A.weight.data -= learning_rate * grad_A
                self.lora_B.weight.data -= learning_rate * grad_B
            else:
                avg_gradients = gradients.mean(dim=[0, 1])
                grad_A = avg_gradients.view(1, -1)[:, :self.input_dim].view(self.rank, -1)
                grad_B = avg_gradients.view(1, -1)[:, :self.input_dim].view(self.input_dim, -1)
                self.lora_A.weight.data -= grad_A
                self.lora_B.weight.data -= grad_B.t()
                
    def get_stats(self) -> Dict[str, float]:
        """
        Get adapter statistics.
        
        Returns:
            Dictionary containing adapter statistics
        """
        with torch.no_grad():
            stats = {
                'rank': self.rank,
                'alpha': self.alpha,
                'input_dim': self.input_dim,
                'param_count': sum(p.numel() for p in self.parameters()),
                'a_norm': torch.norm(self.lora_A.weight).item(),
                'b_norm': torch.norm(self.lora_B.weight).item()
            }
        return stats
