import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional

class ImportanceTracker:
    """Tracks importance of concepts/tokens during inference"""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.importance_scores: Dict[int, float] = {}  # token_position -> importance
        
    def mark_important(self, positions: List[int], concept_name: str, weight: float = 1.0):
        """Mark specific positions as important for a concept"""
        for pos in positions:
            self.importance_scores[pos] = weight
            
    def get_importance_mask(self, seq_length: int) -> mx.array:
        """Get importance mask for sequence"""
        mask = mx.ones(seq_length)
        for pos, weight in self.importance_scores.items():
            if pos < seq_length:
                mask = mask.at[pos].set(weight)
        return mask

class GeometricTransform(nn.Module):
    """Applies geometric transformations to embeddings"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.importance_tracker = ImportanceTracker(hidden_size)
        
    def __call__(self, x: mx.array) -> mx.array:
        """Apply geometric transformation to input"""
        # Start with identity transform
        return x
