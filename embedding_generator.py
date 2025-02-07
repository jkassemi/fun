"""
Generate meaningful embeddings from text concepts instead of random values.
"""

import mlx.core as mx
from mlx_lm import load, generate
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class ConceptPair:
    """A pair of related concepts for generating embeddings"""
    primary: str
    contrast: str = None

class EmbeddingGenerator:
    """Generate embeddings from meaningful text"""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model, self.tokenizer = load(model_name)
        # Get model dimension from sample output
        sample_ids = self.tokenizer.encode("test")
        sample_output = self.model(mx.array([sample_ids]))
        self.hidden_size = sample_output.shape[-1]
        
        # Define concept pairs for different semantic dimensions
        self.concept_pairs = {
            "technical": ConceptPair(
                "code algorithms software engineering",
                "poetry art music emotion"
            ),
            "emotional": ConceptPair(
                "joy love happiness peace",
                "anger fear sadness hate"
            ),
            "abstract": ConceptPair(
                "truth logic reason fact",
                "illusion chaos random noise"
            ),
            "temporal": ConceptPair(
                "now present immediate current",
                "past ancient history memory"
            )
        }
        
    def get_embedding(self, text: str) -> mx.array:
        """Get embedding for text"""
        ids = self.tokenizer.encode(text)
        output = self.model(mx.array([ids]))
        # Use mean of token embeddings
        return mx.mean(output, axis=1)[0]
    
    def random_embedding(self, use_meaningful: bool = True) -> mx.array:
        """Get random embedding, either meaningful or pure random"""
        if not use_meaningful:
            return mx.random.normal((self.hidden_size,))
            
        # Pick random concept pair
        pair_name = np.random.choice(list(self.concept_pairs.keys()))
        pair = self.concept_pairs[pair_name]
        
        # Get embedding for primary concept
        return self.get_embedding(pair.primary)
    
    def concept_embedding(self, concept_name: str) -> Tuple[mx.array, Optional[mx.array]]:
        """Get embeddings for a specific concept pair"""
        if concept_name not in self.concept_pairs:
            raise KeyError(f"Concept {concept_name} not found")
            
        pair = self.concept_pairs[concept_name]
        primary = self.get_embedding(pair.primary)
        contrast = self.get_embedding(pair.contrast) if pair.contrast else None
        
        return primary, contrast
