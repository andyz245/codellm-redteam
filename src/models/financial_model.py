"""
Financial Model Implementation for AdaptiFin

This module implements a concrete financial language model with dynamic parameter
selection and market-aware updates.
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .base_model import BaseModel


class FinancialModel(BaseModel):
    """
    Financial language model with dynamic parameter selection and
    market-aware updates.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        sensitivity_threshold: float = 0.1,
        accumulation_steps: int = 4,
        learning_rate: float = 1e-4
    ):
        """
        Initialize financial model.
        
        Args:
            model_name: Name of the pre-trained model to use
            batch_size: Size of mini-batches for sensitivity computation
            sensitivity_threshold: Threshold for parameter importance
            accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Base learning rate for updates
        """
        super().__init__(
            batch_size=batch_size,
            sensitivity_threshold=sensitivity_threshold,
            accumulation_steps=accumulation_steps
        )
        
        # Initialize underlying transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.learning_rate = learning_rate
        
        # Market-aware projection layer
        self.market_projection = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size
        )
        
        # Initialize parameter selection
        self.setup_parameter_selection()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        market_metrics: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass with market-aware computation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            market_metrics: Optional market performance metrics
            **kwargs: Additional arguments for transformer
            
        Returns:
            Model outputs with market awareness
        """
        # Get base transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply market-aware projection
        hidden_states = outputs.last_hidden_state
        if market_metrics is not None:
            # Incorporate market metrics into hidden states
            market_projection = self.market_projection(hidden_states)
            # Properly broadcast market metrics across sequence length and hidden dimension
            expanded_metrics = market_metrics.unsqueeze(1).unsqueeze(2).expand(-1, hidden_states.size(1), hidden_states.size(2))
            hidden_states = hidden_states + expanded_metrics * market_projection
            
        return hidden_states
    
    def get_learning_rate(self) -> float:
        """Get current learning rate for parameter updates."""
        return self.learning_rate
        
    def process_market_batch(
        self,
        input_ids: torch.Tensor,
        market_metrics: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of market data efficiently.
        
        Args:
            input_ids: Input token IDs
            market_metrics: Market performance metrics
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing processed outputs and metrics
        """
        # Forward pass with market metrics
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            market_metrics=market_metrics
        )
        
        # Compute loss incorporating market metrics
        loss = self._compute_market_aware_loss(outputs, market_metrics)
        
        # Update parameters with market feedback
        self.update_parameters(loss, market_metrics)
        
        return {
            'loss': loss,
            'outputs': outputs,
            'market_impact': self._compute_market_impact(outputs, market_metrics)
        }
        
    def _compute_market_aware_loss(
        self,
        outputs: torch.Tensor,
        market_metrics: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss incorporating market performance.
        
        Args:
            outputs: Model outputs
            market_metrics: Market performance metrics
            
        Returns:
            Combined loss value
        """
        # Basic loss computation
        base_loss = torch.mean(outputs)
        
        # Incorporate market metrics with proper broadcasting
        # Reshape market metrics to match output dimensions [batch, seq_len, hidden]
        expanded_metrics = market_metrics.unsqueeze(1).unsqueeze(2).expand(-1, outputs.size(1), 1)
        market_loss = torch.mean(outputs * expanded_metrics)
        
        return base_loss + 0.5 * market_loss
        
    def _compute_market_impact(
        self,
        outputs: torch.Tensor,
        market_metrics: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute impact of market metrics on model outputs.
        
        Args:
            outputs: Model outputs [batch, seq_len, hidden]
            market_metrics: Market performance metrics [batch]
            
        Returns:
            Market impact scores [batch]
        """
        with torch.no_grad():
            # Properly broadcast market metrics
            expanded_metrics = market_metrics.unsqueeze(1).unsqueeze(2).expand(-1, outputs.size(1), 1)
            # Compute impact scores across sequence length and hidden dimensions
            impact = torch.mean(torch.sum(outputs * expanded_metrics, dim=-1), dim=1)
        return impact
