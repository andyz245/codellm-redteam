"""
Market Feedback Buffer Module for AdaptiFin

Implements sliding window mechanism with exponential decay for efficient
market feedback integration and memory management.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketFeedback:
    """Container for market feedback data with timestamp."""
    timestamp: float
    metrics: torch.Tensor
    impact: float


class MarketFeedbackBuffer:
    """
    Implements sliding window buffer with exponential decay for
    market feedback integration.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        decay_factor: float = 0.95,
        memory_efficient: bool = True
    ):
        """
        Initialize market feedback buffer.
        
        Args:
            window_size: Maximum size of sliding window
            decay_factor: Exponential decay factor for historical data
            memory_efficient: Whether to use memory-efficient storage
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.memory_efficient = memory_efficient
        
        # Initialize sliding window buffer
        self.feedback_buffer: deque[MarketFeedback] = deque(maxlen=window_size)
        self.current_timestamp = 0.0
        
    def add_feedback(
        self,
        market_metrics: torch.Tensor,
        impact: float,
        timestamp: Optional[float] = None
    ):
        """
        Add new market feedback to buffer.
        
        Args:
            market_metrics: Current market performance metrics
            impact: Impact score of the feedback
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        # Update current timestamp
        self.current_timestamp = max(self.current_timestamp, timestamp)
        
        # Add new feedback to buffer
        feedback = MarketFeedback(
            timestamp=timestamp,
            metrics=market_metrics.detach().clone(),
            impact=impact
        )
        
        self.feedback_buffer.append(feedback)
        
        # Prune old data if memory efficiency is enabled
        if self.memory_efficient and len(self.feedback_buffer) == self.window_size:
            self._prune_old_data()
            
    def get_weighted_feedback(
        self,
        current_time: Optional[float] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Get weighted average of feedback using exponential decay.
        
        Args:
            current_time: Optional current timestamp
            
        Returns:
            Tuple of (weighted market metrics, weighted impact)
        """
        if not self.feedback_buffer:
            return torch.zeros_like(self.feedback_buffer[0].metrics), 0.0
            
        if current_time is None:
            current_time = self.current_timestamp
            
        # Compute weights using exponential decay
        weights = torch.tensor([
            self.decay_factor ** ((current_time - feedback.timestamp) / 3600)
            for feedback in self.feedback_buffer
        ])
        weights = weights / weights.sum()
        
        # Compute weighted averages
        weighted_metrics = torch.zeros_like(self.feedback_buffer[0].metrics)
        weighted_impact = 0.0
        
        for feedback, weight in zip(self.feedback_buffer, weights):
            weighted_metrics += feedback.metrics * weight
            weighted_impact += feedback.impact * weight.item()
            
        return weighted_metrics, weighted_impact
        
    def _prune_old_data(self, threshold: float = 0.01):
        """
        Remove old data with negligible impact.
        
        Args:
            threshold: Minimum weight threshold for keeping data
        """
        current_time = self.current_timestamp
        min_timestamp = current_time - (
            -np.log(threshold) / np.log(self.decay_factor) * 3600
        )
        
        # Remove feedback older than minimum timestamp
        while (
            self.feedback_buffer and
            self.feedback_buffer[0].timestamp < min_timestamp
        ):
            self.feedback_buffer.popleft()
            
    def get_stats(self) -> Dict[str, float]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary containing buffer statistics
        """
        stats = {
            'buffer_size': len(self.feedback_buffer),
            'window_size': self.window_size,
            'decay_factor': self.decay_factor,
            'memory_usage': sum(
                feedback.metrics.nelement() * feedback.metrics.element_size()
                for feedback in self.feedback_buffer
            )
        }
        
        if self.feedback_buffer:
            stats.update({
                'oldest_timestamp': self.feedback_buffer[0].timestamp,
                'newest_timestamp': self.feedback_buffer[-1].timestamp,
                'avg_impact': np.mean([
                    feedback.impact
                    for feedback in self.feedback_buffer
                ])
            })
            
        return stats
