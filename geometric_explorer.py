"""
Explore geometric transformations of embeddings.
Command-line interface for rapid experimentation.
"""

import mlx.core as mx
from mlx_lm import load, generate
import numpy as np
from typing import List, Tuple, Optional
from geometric_core import GeometricCore, TransformationField, TransformationType

class GeometricExplorer:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model, self.tokenizer = load(model_name)
        # Get hidden size from model's embedding dimension
        sample_text = "test"
        sample_ids = self.tokenizer.encode(sample_text)
        sample_embedding = self.model.embed(mx.array([sample_ids]))
        hidden_size = sample_embedding.shape[-1]
        self.core = GeometricCore(hidden_size)
        
    def get_embeddings(self, text: str) -> Tuple[List[Tuple[str, int]], mx.array]:
        """Get token IDs and embeddings for text"""
        ids = self.tokenizer.encode(text)
        tokens = [(self.tokenizer.decode([id]), id) for id in ids]
        return tokens, self.model.embed(mx.array([ids]))
    
    def add_transformation(self, 
                         center: Optional[mx.array] = None,
                         direction: Optional[mx.array] = None,
                         transform_type: TransformationType = TransformationType.ROTATION,
                         strength: float = 1.0) -> None:
        """Add a transformation field"""
        if center is None:
            center = mx.random.normal((self.model.config.hidden_size,))
            
        field = TransformationField(
            center=center,
            direction=direction,
            transform_type=transform_type,
            strength=strength
        )
        self.core.add_field(field)
    
    def explore(self, text: str):
        """Interactive exploration of transformations"""
        tokens, embeddings = self.get_embeddings(text)
        
        print(f"\nInput text: {text}")
        print(f"Tokens: {tokens}")
        
        # Add some example transformations
        self.add_transformation(transform_type=TransformationType.ROTATION)
        self.add_transformation(transform_type=TransformationType.SCALING)
        
        # Apply transformations
        transformed = self.core.apply_all_fields(embeddings)
        
        # Generate from transformed embeddings
        output_ids = self.model.generate(transformed, max_tokens=20)
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        
        print(f"\nTransformed output: {output_text}")
        
        # Show transformation effects
        print("\nTransformation analysis:")
        print(f"Original norm: {mx.linalg.norm(embeddings)}")
        print(f"Transformed norm: {mx.linalg.norm(transformed)}")
        
        # Calculate overall change
        delta = mx.linalg.norm(transformed - embeddings) / mx.linalg.norm(embeddings)
        print(f"Relative change: {delta:.2%}")

def main():
    explorer = GeometricExplorer()
    
    while True:
        try:
            text = input("\nEnter text to transform (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
            explorer.explore(text)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
