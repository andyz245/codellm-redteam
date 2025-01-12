"""
Parameter Metrics Module for AdaptiFin

This module provides utilities for tracking and analyzing parameter importance
metrics during model fine-tuning.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from dataclasses import dataclass
from collections import deque


@dataclass
class ParameterMetrics:
    """Stores metrics for parameter importance tracking."""
    sensitivity_score: float
    update_frequency: int
    impact_history: List[float]


class ParameterMetricsTracker:
    """
    Tracks and analyzes parameter importance metrics over time.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        decay_factor: float = 0.95
    ):
        """
        Initialize parameter metrics tracker.
        
        Args:
            window_size: Size of sliding window for metric history
            decay_factor: Exponential decay factor for historical metrics
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.metrics: Dict[str, ParameterMetrics] = {}
        
    def update_metrics(
        self,
        parameter_name: str,
        sensitivity_score: float,
        impact_score: Optional[float] = None
    ):
        """
        Update metrics for a parameter.
        
        Args:
            parameter_name: Name of the parameter
            sensitivity_score: Current sensitivity score
            impact_score: Optional impact score from recent updates
        """
        if parameter_name not in self.metrics:
            self.metrics[parameter_name] = ParameterMetrics(
                sensitivity_score=sensitivity_score,
                update_frequency=0,
                impact_history=[]
            )
        
        metrics = self.metrics[parameter_name]
        
        # Update sensitivity score with exponential decay
        metrics.sensitivity_score = (
            self.decay_factor * metrics.sensitivity_score +
            (1 - self.decay_factor) * sensitivity_score
        )
        
        # Update impact history
        if impact_score is not None:
            if len(metrics.impact_history) >= self.window_size:
                metrics.impact_history.pop(0)
            metrics.impact_history.append(impact_score)
        
        metrics.update_frequency += 1
        
    def get_parameter_importance(self, parameter_name: str) -> float:
        """
        Get overall importance score for a parameter.
        
        Args:
            parameter_name: Name of the parameter
            
        Returns:
            Combined importance score based on sensitivity and impact
        """
        if parameter_name not in self.metrics:
            return 0.0
            
        metrics = self.metrics[parameter_name]
        
        # Combine sensitivity score with impact history
        importance = float(metrics.sensitivity_score)
        
        if metrics.impact_history:
            recent_impact = float(np.mean(metrics.impact_history[-10:]))
            importance *= (1 + recent_impact)
            
        return float(importance)
        
    def get_top_parameters(self, k: int = 10) -> List[str]:
        """
        Get top-k most important parameters.
        
        Args:
            k: Number of parameters to return
            
        Returns:
            List of parameter names sorted by importance
        """
        parameter_scores = [
            (name, self.get_parameter_importance(name))
            for name in self.metrics.keys()
        ]
        
        # Sort by importance score in descending order
        parameter_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in parameter_scores[:k]]
        
    def get_parameter_stats(self, parameter_name: str) -> Dict[str, float]:
        """
        Get detailed statistics for a parameter.
        
        Args:
            parameter_name: Name of the parameter
            
        Returns:
            Dictionary containing parameter statistics
        """
        if parameter_name not in self.metrics:
            return {}
            
        metrics = self.metrics[parameter_name]
        
        stats = {
            'sensitivity_score': metrics.sensitivity_score,
            'update_frequency': metrics.update_frequency,
            'avg_impact': np.mean(metrics.impact_history) if metrics.impact_history else 0.0,
            'recent_impact': np.mean(metrics.impact_history[-10:]) if metrics.impact_history else 0.0
        }
        
        return stats
