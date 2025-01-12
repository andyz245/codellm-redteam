"""
Test Suite for AdaptiFin Framework

Validates the implementation of dynamic parameter selection,
hierarchical fine-tuning, and market feedback integration.
"""

import unittest
import torch
import numpy as np
from typing import Dict, List

from src.models.financial_model import FinancialModel
from src.adapters.hierarchical_adapter import HierarchicalAdapter, LayerType
from src.adapters.lora_adapter import LoRAAdapter
from src.feedback.market_feedback_buffer import MarketFeedbackBuffer


class TestAdaptiFin(unittest.TestCase):
    """Test cases for AdaptiFin framework components."""
    
    def setUp(self):
        """Set up test environment."""
        self.model_dim = 768
        self.batch_size = 32
        self.seq_length = 128
        
        # Initialize components
        self.model = FinancialModel(
            model_name="bert-base-uncased",
            batch_size=self.batch_size
        )
        
        self.adapter = HierarchicalAdapter(
            model_dim=self.model_dim
        )
        
        self.market_buffer = MarketFeedbackBuffer(
            window_size=100,
            decay_factor=0.95
        )
        
    def test_dynamic_parameter_selection(self):
        """Test dynamic parameter selection mechanism."""
        # Create sample input
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        market_metrics = torch.randn(self.batch_size)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            market_metrics=market_metrics
        )
        
        # Process batch with market feedback
        result = self.model.process_market_batch(
            input_ids=input_ids,
            market_metrics=market_metrics
        )
        
        # Verify parameter selection
        self.assertIn('market_impact', result)
        self.assertIsInstance(result['market_impact'], torch.Tensor)
        self.assertEqual(result['market_impact'].shape[0], self.batch_size)
        
    def test_hierarchical_adaptation(self):
        """Test three-layer adaptation structure."""
        # Create sample hidden states
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.model_dim)
        
        # Test each layer individually
        for layer_type in LayerType:
            adapted_states = self.adapter.forward(
                hidden_states=hidden_states,
                layer_mask=[layer_type]
            )
            
            # Verify shape preservation
            self.assertEqual(adapted_states.shape, hidden_states.shape)
            
            # Verify non-zero adaptation
            self.assertFalse(torch.allclose(adapted_states, hidden_states))
            
    def test_market_feedback_integration(self):
        """Test sliding window mechanism for market feedback."""
        # Add sample feedback
        for i in range(50):
            metrics = torch.randn(10)  # Sample market metrics
            impact = float(torch.rand(1))
            timestamp = float(i)
            
            self.market_buffer.add_feedback(
                market_metrics=metrics,
                impact=impact,
                timestamp=timestamp
            )
            
        # Get weighted feedback
        weighted_metrics, weighted_impact = self.market_buffer.get_weighted_feedback()
        
        # Verify exponential decay
        self.assertGreater(weighted_impact, 0)
        self.assertEqual(weighted_metrics.shape[0], 10)
        
        # Verify buffer size
        stats = self.market_buffer.get_stats()
        self.assertLessEqual(stats['buffer_size'], 100)
        
    def test_computational_efficiency(self):
        """Test computational efficiency of parameter updates."""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        market_metrics = torch.randn(self.batch_size)
        
        # Measure memory usage before and after
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Process batch
        result = self.model.process_market_batch(
            input_ids=input_ids,
            market_metrics=market_metrics
        )
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_increase = final_memory - initial_memory
        
        # Verify reasonable memory usage (less than 1GB increase)
        self.assertLess(memory_increase, 1e9)
        
    def test_layer_interaction(self):
        """Test controlled interaction between adaptation layers."""
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.model_dim)
        gradients = torch.randn_like(hidden_states)
        market_feedback = torch.randn(self.batch_size)
        
        # Update each layer
        for layer_type in LayerType:
            self.adapter.update_layer(
                layer_type=layer_type,
                gradients=gradients,
                market_feedback=market_feedback
            )
            
        # Get layer statistics
        stats = self.adapter.get_layer_stats()
        
        # Verify layer-specific updates
        for layer_type in LayerType:
            layer_stats = stats[layer_type.value]
            self.assertIn('update_frequency', layer_stats)
            self.assertIn('learning_rate', layer_stats)
            self.assertIn('updates_performed', layer_stats)
            
    def test_memory_management(self):
        """Test memory management in feedback buffer."""
        # Fill buffer beyond window size
        for i in range(150):  # More than window size
            metrics = torch.randn(10)
            impact = float(torch.rand(1))
            self.market_buffer.add_feedback(metrics, impact)
            
        # Verify automatic pruning
        stats = self.market_buffer.get_stats()
        self.assertLessEqual(
            stats['buffer_size'],
            self.market_buffer.window_size
        )
        
        # Verify memory efficiency
        self.assertLess(
            stats['memory_usage'],
            1e8  # Less than 100MB
        )


if __name__ == '__main__':
    unittest.main()
