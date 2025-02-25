"""
Automated tests for geometric transformations of embeddings.
Generates results for paper analysis.
"""

import mlx.core as mx
import numpy as np
from typing import List, Tuple
from geometric_core import GeometricCore, TransformationField, TransformationType
from embedding_generator import EmbeddingGenerator
import json
from pathlib import Path

def create_test_embeddings(generator: EmbeddingGenerator, num_vectors: int, 
                         use_meaningful: bool = True) -> mx.array:
    """Create test embeddings with known properties"""
    embeddings = []
    for _ in range(num_vectors):
        emb = generator.random_embedding(use_meaningful)
        embeddings.append(emb)
    
    # Stack and normalize
    embeddings = mx.stack(embeddings)
    embeddings = embeddings.reshape(1, num_vectors, -1)
    norms = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norms + 1e-6)

def run_transformation_tests(save_dir: str = "results"):
    """Run suite of geometric transformation tests"""
    Path(save_dir).mkdir(exist_ok=True)
    results = {}
    
    # Test parameters
    num_vectors = 10
    
    # Create generators and test data
    generator = EmbeddingGenerator()
    core = GeometricCore(generator.hidden_size)
    embeddings = create_test_embeddings(generator, num_vectors)
    
    # Test each transformation type
    print("\nTesting transformations...")
    results["transformations"] = {}
    
    test_text = "The meaning of life is understanding and growth"
    base_embedding = generator.get_embedding(test_text)
    
    for transform_type in TransformationType:
        print(f"\nTesting {transform_type.value}...")
        results["transformations"][transform_type.value] = {}
        
        for strength in [0.1, 0.5, 0.9]:
            center = generator.get_embedding("truth and wisdom")
            direction = generator.get_embedding("progress and evolution") if transform_type in [
                TransformationType.ROTATION, 
                TransformationType.TRANSLATION
            ] else None
            
            field = TransformationField(
                center=center,
                direction=direction,
                transform_type=transform_type,
                strength=strength
            )
        
            transformed = core.apply_field(embeddings, field)
            
            # Calculate metrics
            orig_norms = mx.linalg.norm(embeddings, axis=-1)
            trans_norms = mx.linalg.norm(transformed, axis=-1)
            delta_norms = (trans_norms - orig_norms) / orig_norms
            
            # Calculate semantic similarity changes
            orig_sim = mx.matmul(embeddings[0], base_embedding) / (
                mx.linalg.norm(embeddings[0], axis=-1) * mx.linalg.norm(base_embedding) + 1e-6
            )
            trans_sim = mx.matmul(transformed[0], base_embedding) / (
                mx.linalg.norm(transformed[0], axis=-1) * mx.linalg.norm(base_embedding) + 1e-6
            )
            
            results["transformations"][transform_type.value][f"strength_{strength}"] = {
                "mean_delta_norm": float(mx.mean(delta_norms)),
                "std_delta_norm": float(mx.std(delta_norms)),
                "mean_semantic_shift": float(mx.mean(trans_sim - orig_sim)),
                "max_semantic_shift": float(mx.max(mx.abs(trans_sim - orig_sim)))
            }
    
    # Test 2: Semantic Lenses
    print("\nTesting semantic lenses...")
    results["lenses"] = {}
    
    # Get meaningful concept embeddings
    truth_emb, lie_emb = generator.concept_embedding("truth")
    joy_emb, despair_emb = generator.concept_embedding("emotion")
    power_emb, powerless_emb = generator.concept_embedding("power")
    connected_emb, alone_emb = generator.concept_embedding("connection")
    
    concepts = {
        "truth": truth_emb,
        "deception": lie_emb,
        "joy": joy_emb,
        "despair": despair_emb,
        "power": power_emb,
        "powerless": powerless_emb,
        "connected": connected_emb,
        "isolation": alone_emb
    }
    
    # Create and test technical lens
    core.create_lens(
        "technical", 
        ["base"], 
        strength=0.5,
        contrast_concepts=["opposite"]
    )
    
    for name, embedding in concepts.items():
        core.add_concept(name, embedding)
    
    # Apply lens and measure effects
    lens_transformed = core.apply_lens(embeddings, "technical")
    
    # Calculate how lens affects similarity to concepts
    for concept in concepts:
        before_sim = core.get_concept_similarity(embeddings[0], concept)
        after_sim = core.get_concept_similarity(lens_transformed[0], concept)
        
        results["lenses"][f"{concept}_effect"] = {
            "before_mean_sim": float(mx.mean(before_sim)),
            "after_mean_sim": float(mx.mean(after_sim)),
            "delta": float(mx.mean(after_sim - before_sim))
        }
    
    # Save results
    print(f"\nSaving results to {save_dir}/transformation_results.json")
    with open(f"{save_dir}/transformation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = run_transformation_tests()
    
    # Print summary
    print("\nTest Results Summary:")
    print("=====================")
    
    print("\nTransformation Tests:")
    for transform_type in TransformationType:
        print(f"\n{transform_type.value.title()} Tests:")
        for strength, metrics in results["transformations"][transform_type.value].items():
            print(f"\nStrength {strength}:")
            print(f"Mean Δnorm: {metrics['mean_delta_norm']:+.2%}")
            print(f"Std Δnorm:  {metrics['std_delta_norm']:.2%}")
            print(f"Mean semantic shift: {metrics['mean_semantic_shift']:+.2%}")
            print(f"Max semantic shift:  {metrics['max_semantic_shift']:+.2%}")
    
    print("\nLens Effect Tests:")
    for concept, metrics in results["lenses"].items():
        print(f"\nEffect on {concept}:")
        print(f"Before: {metrics['before_mean_sim']:+.2f}")
        print(f"After:  {metrics['after_mean_sim']:+.2f}")
        print(f"Delta:  {metrics['delta']:+.2f}")
