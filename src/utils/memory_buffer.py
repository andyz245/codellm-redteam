import numpy as np

class SlidingWindowBuffer:
    """Sliding window buffer with exponential decay for market feedback."""
    
    def __init__(self, window_size: int, decay_factor: float = 0.95):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.buffer = []
