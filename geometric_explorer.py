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
        # Get hidden size from model output
        sample_output = self.model(mx.array([sample_ids]))
        self.hidden_size = sample_output.shape[-1]
        self.core = GeometricCore(self.hidden_size)
        
    def get_embeddings(self, text: str) -> Tuple[List[Tuple[str, int]], mx.array]:
        """Get token IDs and embeddings for text"""
        ids = self.tokenizer.encode(text)
        tokens = [(self.tokenizer.decode([id]), id) for id in ids]
        return tokens, self.model(mx.array([ids]))
    
    def add_transformation(self, 
                         center: Optional[mx.array] = None,
                         direction: Optional[mx.array] = None,
                         transform_type: TransformationType = TransformationType.ROTATION,
                         strength: float = 1.0) -> None:
        """Add a transformation field"""
        if center is None:
            center = mx.random.normal((self.hidden_size,))
            
        field = TransformationField(
            center=center,
            direction=direction,
            transform_type=transform_type,
            strength=strength
        )
        self.core.add_field(field)
    
    def add_concept(self, name: str, text: str) -> None:
        """Add a concept by getting its embedding"""
        _, embeddings = self.get_embeddings(text)
        # Use mean embedding as concept center
        concept_embedding = mx.mean(embeddings, axis=1)[0]
        self.core.add_concept(name, concept_embedding)
        
    def generate_from_embeddings(self, embeddings: mx.array) -> str:
        """Generate text from embeddings"""
        # Use model to generate from the transformed embeddings
        output_ids = generate(self.model, embeddings, self.tokenizer)
        return self.tokenizer.decode(output_ids)

    def explore(self, text: str):
        """Interactive exploration of transformations"""
        tokens, embeddings = self.get_embeddings(text)
        
        print(f"\nInput text: {text}")
        print(f"Tokens: {tokens}")
        
        # Add some example concepts if none exist
        if not self.core.concept_centers:
            self.add_concept("positive", "excellent amazing wonderful")
            self.add_concept("negative", "terrible horrible awful")
            self.add_concept("technical", "computer software code")
            self.add_concept("emotional", "happy sad angry love")
            
        # Show concept similarities
        print("\nConcept Analysis:")
        for concept in self.core.concept_centers.keys():
            sim = self.core.get_concept_similarity(embeddings[0], concept)
            print(f"{concept:>10}: {float(mx.mean(sim)):>+.2%}")
            
        # Add transformations based on concepts
        self.add_transformation(
            center=self.core.concept_centers["technical"],
            transform_type=TransformationType.PROJECTION,
            strength=0.5
        )
        
        # Apply transformations
        transformed = self.core.apply_all_fields(embeddings)
        
        # Analyze token patterns
        print("\nToken Analysis:")
        for i, (token, token_id) in enumerate(tokens):
            print(f"{i:2d}. '{token}' (ID: {token_id})")
            # Compare original vs transformed embeddings for this token
            orig_norm = mx.linalg.norm(embeddings[0,i])
            trans_norm = mx.linalg.norm(transformed[0,i])
            delta = (trans_norm - orig_norm) / orig_norm
            print(f"    Δnorm: {delta:+.2%}")
        
        # Show transformation effects
        print("\nTransformation analysis:")
        print(f"Original norm: {mx.linalg.norm(embeddings)}")
        print(f"Transformed norm: {mx.linalg.norm(transformed)}")
        
        # Calculate overall change
        delta = mx.linalg.norm(transformed - embeddings) / mx.linalg.norm(embeddings)
        print(f"Relative change: {delta:.2%}")
        
        # Generate and show transformed text
        print("\nTransformed text:")
        transformed_text = self.generate_from_embeddings(transformed)
        print(transformed_text)

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
