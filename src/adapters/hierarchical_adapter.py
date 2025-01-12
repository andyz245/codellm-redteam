"""
Hierarchical Adapter Module for AdaptiFin

Implements the three-layer adaptation structure (market, risk, compliance)
with controlled interaction and different update frequencies.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np

from .lora_adapter import LoRAAdapter
from ..feedback.market_feedback_buffer import MarketFeedbackBuffer


class LayerType(Enum):
    """Types of adaptation layers with their update frequencies."""
    MARKET = "market"      # Highest frequency
    RISK = "risk"         # Medium frequency
    COMPLIANCE = "compliance"  # Lowest frequency


@dataclass
class LayerConfig:
    """Configuration for each adaptation layer."""
    update_frequency: int  # Update every N steps
    lora_rank: int        # Rank for LoRA adaptation
    learning_rate: float  # Layer-specific learning rate


class HierarchicalAdapter(nn.Module):
    """
    Implements hierarchical fine-tuning with three-layer adaptation structure
    and controlled interaction between layers.
    """
    
    def __init__(
        self,
        model_dim: int,
        layer_configs: Optional[Dict[LayerType, LayerConfig]] = None
    ):
        """
        Initialize hierarchical adapter.
        
        Args:
            model_dim: Dimension of model hidden states
            layer_configs: Optional configurations for each layer
        """
        super().__init__()
        
        # Default layer configurations if not provided
        self.layer_configs = layer_configs or {
            LayerType.MARKET: LayerConfig(
                update_frequency=1,    # Update every step
                lora_rank=8,
                learning_rate=1e-4
            ),
            LayerType.RISK: LayerConfig(
                update_frequency=10,   # Update every 10 steps
                lora_rank=4,
                learning_rate=5e-5
            ),
            LayerType.COMPLIANCE: LayerConfig(
                update_frequency=100,  # Update every 100 steps
                lora_rank=2,
                learning_rate=1e-5
            )
        }
        
        # Initialize LoRA adapters for each layer
        self.adapters = nn.ModuleDict({
            layer_type.value: LoRAAdapter(
                input_dim=model_dim,
                rank=config.lora_rank,
                layer_name=layer_type.value
            )
            for layer_type, config in self.layer_configs.items()
        })
        
        # Layer-specific update tracking
        self.update_steps = {
            layer_type: 0
            for layer_type in LayerType
        }
        
        # Initialize interaction controls
        self.layer_weights = nn.Parameter(
            torch.ones(len(LayerType)) / len(LayerType)
        )
        
        # Initialize market feedback buffer
        self.market_feedback = MarketFeedbackBuffer(
            window_size=1000,
            decay_factor=0.95
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_mask: Optional[List[LayerType]] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical adaptation layers.
        
        Args:
            hidden_states: Input hidden states
            layer_mask: Optional list of layers to use
            
        Returns:
            Adapted hidden states
        """
        adapted_states = hidden_states
        active_layers = layer_mask or list(LayerType)
        
        # Apply each active layer's adaptation with residual connection
        for idx, layer_type in enumerate(active_layers):
            layer_output = self.adapters[layer_type.value](adapted_states)
            weight = torch.sigmoid(self.layer_weights[idx])  # Ensure positive scaling
            adapted_states = adapted_states + weight * layer_output
            # Add layer-specific normalization to prevent information loss
            layer_norm = nn.LayerNorm(adapted_states.size(-1), device=adapted_states.device)
            adapted_states = layer_norm(adapted_states)
            
        return adapted_states
        
    def update_layer(
        self,
        layer_type: LayerType,
        gradients: torch.Tensor,
        market_feedback: Optional[torch.Tensor] = None
    ):
        """
        Update a specific layer if its update frequency condition is met.
        
        Args:
            layer_type: Type of layer to update
            gradients: Computed gradients
            market_feedback: Optional market performance feedback
        """
        config = self.layer_configs[layer_type]
        self.update_steps[layer_type] += 1
        
        # Check if update frequency condition is met
        if self.update_steps[layer_type] % config.update_frequency == 0:
            # Scale gradients by layer-specific learning rate
            scaled_gradients = gradients * config.learning_rate
            
            # Get weighted market feedback
            if market_feedback is not None:
                self.market_feedback.add_feedback(
                    market_metrics=market_feedback,
                    impact=torch.norm(gradients).item()
                )
                
            weighted_feedback, impact = self.market_feedback.get_weighted_feedback()
            
            # Apply market feedback to market layer with proper broadcasting
            if layer_type == LayerType.MARKET:
                # Reshape feedback for proper broadcasting across all dimensions
                feedback_shape = [-1] + [1] * (scaled_gradients.dim() - 1)
                reshaped_feedback = weighted_feedback.view(*feedback_shape)
                scaled_gradients *= (1 + reshaped_feedback)
            
            # Update layer parameters with decay-adjusted learning rate
            effective_lr = config.learning_rate
            if impact > 0:
                effective_lr *= (1 + 0.1 * impact)
                
            self.adapters[layer_type.value].update_weights(
                scaled_gradients,
                learning_rate=effective_lr
            )
            
    def get_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each layer.
        
        Returns:
            Dictionary containing layer statistics
        """
        stats = {}
        for layer_type in LayerType:
            adapter = self.adapters[layer_type.value]
            config = self.layer_configs[layer_type]
            
            # Get layer index for weight access
            layer_idx = list(LayerType).index(layer_type)
            
            stats[layer_type.value] = {
                'update_frequency': config.update_frequency,
                'learning_rate': config.learning_rate,
                'weight': self.layer_weights[layer_idx].item(),
                'updates_performed': self.update_steps[layer_type]
            }
            
        return stats
        
    def reset_update_tracking(self):
        """Reset update step counters for all layers."""
        self.update_steps = {
            layer_type: 0
            for layer_type in LayerType
        }
