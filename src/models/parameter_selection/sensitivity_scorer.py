"""
Sensitivity Scorer Module for AdaptiFin

This module implements lightweight parameter sensitivity analysis for efficient
parameter selection in financial language models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

class SensitivityScorer:
    """
    Implements efficient parameter sensitivity scoring for selective fine-tuning.
    Uses small batches of market data to identify high-impact parameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        score_threshold: float = 0.1,
        memory_efficient: bool = True
    ):
        """
        Initialize the sensitivity scorer.
        
        Args:
            model: The underlying language model
            batch_size: Size of mini-batches for sensitivity computation
            score_threshold: Threshold for parameter importance
            memory_efficient: Whether to use memory-efficient computation
        """
        self.model = model
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.memory_efficient = memory_efficient
        self.parameter_scores: Dict[str, float] = {}
        self.gradient_accumulator = GradientAccumulator(model)
        
    def compute_parameter_sensitivity(
        self,
        batch_data: torch.Tensor,
        market_metrics: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute sensitivity scores for model parameters using a small batch.
        
        Args:
            batch_data: Input data batch
            market_metrics: Optional market performance metrics
            
        Returns:
            Dictionary mapping parameter names to sensitivity scores
        """
        self.model.zero_grad()
        
        # Forward pass with market-aware loss computation
        outputs = self.model(batch_data)
        loss = self._compute_market_aware_loss(outputs, market_metrics)
        
        # Efficient backward pass with gradient accumulation
        loss.backward()
        
        # Compute parameter sensitivities
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Compute importance score based on gradient magnitude
                    score = torch.norm(param.grad).item()
                    
                    # Update running average of parameter sensitivity
                    if name in self.parameter_scores:
                        self.parameter_scores[name] = 0.9 * self.parameter_scores[name] + 0.1 * score
                    else:
                        self.parameter_scores[name] = score
        
        return self.parameter_scores
    
    def select_important_parameters(self) -> List[str]:
        """
        Select parameters for update based on sensitivity scores.
        
        Returns:
            List of parameter names that exceed the importance threshold
        """
        if not self.parameter_scores:
            return []
            
        # Normalize scores
        scores = np.array(list(self.parameter_scores.values()))
        normalized_scores = (scores - scores.mean()) / scores.std()
        
        # Select parameters above threshold
        important_params = [
            name for name, score in zip(self.parameter_scores.keys(), normalized_scores)
            if score > self.score_threshold
        ]
        
        return important_params
    
    def _compute_market_aware_loss(
        self,
        model_outputs: torch.Tensor,
        market_metrics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss incorporating market performance metrics if available.
        
        Args:
            model_outputs: Output from the model forward pass
            market_metrics: Optional market performance metrics
            
        Returns:
            Combined loss value incorporating market awareness
        """
        base_loss = self._compute_base_loss(model_outputs)
        
        if market_metrics is not None:
            # Incorporate market metrics into loss computation
            market_loss = self._compute_market_loss(model_outputs, market_metrics)
            return base_loss + 0.5 * market_loss
        
        return base_loss
    
    def _compute_base_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute base model loss."""
        # Implement based on specific model architecture
        return torch.mean(outputs)
    
    def _compute_market_loss(
        self,
        outputs: torch.Tensor,
        market_metrics: torch.Tensor
    ) -> torch.Tensor:
        """Compute market-aware component of loss."""
        # Implement market-specific loss computation
        return torch.mean(outputs * market_metrics)


class GradientAccumulator:
    """
    Implements efficient gradient accumulation for focused parameter updates.
    """
    
    def __init__(self, model: nn.Module, accumulation_steps: int = 4):
        """
        Initialize gradient accumulator.
        
        Args:
            model: The underlying model
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_grads: Dict[str, torch.Tensor] = {}
        
    def accumulate_gradients(self, parameter_mask: Optional[List[str]] = None):
        """
        Accumulate gradients for specified parameters.
        
        Args:
            parameter_mask: Optional list of parameter names to accumulate
        """
        self.current_step += 1
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if parameter_mask is None or name in parameter_mask:
                        if name not in self.accumulated_grads:
                            self.accumulated_grads[name] = param.grad.clone()
                        else:
                            self.accumulated_grads[name] += param.grad
                            
        if self.current_step >= self.accumulation_steps:
            self._apply_accumulated_gradients()
            
    def _apply_accumulated_gradients(self):
        """Apply accumulated gradients and reset accumulator."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.accumulated_grads:
                    param.grad = self.accumulated_grads[name] / self.accumulation_steps
                    
        self.accumulated_grads.clear()
        self.current_step = 0
