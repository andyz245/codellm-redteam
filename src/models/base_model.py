from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from .parameter_selection.sensitivity_scorer import SensitivityScorer
from ..utils.metrics.parameter_metrics import ParameterMetricsTracker


class BaseModel(ABC, nn.Module):
    """
    Base class for all models in AdaptiFin framework.
    Implements dynamic parameter selection and market-aware updates.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        sensitivity_threshold: float = 0.1,
        accumulation_steps: int = 4
    ):
        """
        Initialize base model with dynamic parameter selection.
        
        Args:
            batch_size: Size of mini-batches for sensitivity computation
            sensitivity_threshold: Threshold for parameter importance
            accumulation_steps: Number of steps to accumulate gradients
        """
        super().__init__()
        self.batch_size = batch_size
        self.sensitivity_threshold = sensitivity_threshold
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
        # Initialize parameter selection components
        self.sensitivity_scorer = None  # Initialized in setup_parameter_selection
        self.metrics_tracker = ParameterMetricsTracker()
        
    def setup_parameter_selection(self):
        """Initialize parameter selection after model parameters are defined."""
        self.sensitivity_scorer = SensitivityScorer(
            model=self,
            batch_size=self.batch_size,
            score_threshold=self.sensitivity_threshold
        )
    
    @abstractmethod
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
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model outputs
        """
        pass
    
    def update_parameters(
        self,
        loss: torch.Tensor,
        market_metrics: Optional[torch.Tensor] = None
    ):
        """
        Update parameters using dynamic selection and gradient accumulation.
        
        Args:
            loss: Current batch loss
            market_metrics: Optional market performance metrics
        """
        if self.sensitivity_scorer is None:
            self.setup_parameter_selection()
            return  # Skip this update to ensure proper initialization
            
        # Ensure sensitivity scorer is initialized
        assert self.sensitivity_scorer is not None, "Sensitivity scorer not initialized"
            
        # Compute parameter sensitivities
        try:
            sensitivity_scores = self.sensitivity_scorer.compute_parameter_sensitivity(
                loss,
                market_metrics
            )
            
            # Update parameter metrics
            for name, score in sensitivity_scores.items():
                self.metrics_tracker.update_metrics(name, score)
            
            # Accumulate gradients for important parameters
            important_params = self.sensitivity_scorer.select_important_parameters()
            if hasattr(self.sensitivity_scorer, 'gradient_accumulator'):
                self.sensitivity_scorer.gradient_accumulator.accumulate_gradients(
                    parameter_mask=important_params
                )
        except Exception as e:
            print(f"Error during parameter update: {e}")
            return
        
        self.current_step += 1
        
        # Apply accumulated gradients when ready
        if self.current_step >= self.accumulation_steps:
            self._apply_updates()
            self.current_step = 0
    
    def _apply_updates(self):
        """Apply accumulated gradients to selected parameters."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    importance = self.metrics_tracker.get_parameter_importance(name)
                    # Scale update by parameter importance
                    param.data -= param.grad * importance * self.get_learning_rate()
    
    @abstractmethod
    def get_learning_rate(self) -> float:
        """Get current learning rate for parameter updates."""
        pass
    
    def get_parameter_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all parameters.
        
        Returns:
            Dictionary mapping parameter names to their statistics
        """
        stats = {}
        for name, _ in self.named_parameters():
            param_stats = self.metrics_tracker.get_parameter_stats(name)
            if param_stats:
                stats[name] = param_stats
        return stats
    
    def get_top_parameters(self, k: int = 10) -> List[str]:
        """
        Get names of top-k most important parameters.
        
        Args:
            k: Number of parameters to return
            
        Returns:
            List of parameter names
        """
        return self.metrics_tracker.get_top_parameters(k)
