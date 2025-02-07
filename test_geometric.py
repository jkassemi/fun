"""
Automated tests for geometric transformations of embeddings.
Generates results for paper analysis.
"""

import mlx.core as mx
import numpy as np
from typing import List, Tuple
from geometric_core import GeometricCore, TransformationField, TransformationType
import json
from pathlib import Path

def create_test_embeddings(dim: int, num_vectors: int) -> mx.array:
    """Create test embeddings with known properties"""
    embeddings = mx.random.normal((1, num_vectors, dim))
    # Normalize each vector
    norms = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norms + 1e-6)

def run_transformation_tests(save_dir: str = "results"):
    """Run suite of geometric transformation tests"""
    Path(save_dir).mkdir(exist_ok=True)
    results = {}
    
    # Test parameters
    dim = 64  # Small dimension for quick testing
    num_vectors = 10
    
    # Create core and test data
    core = GeometricCore(dim)
    embeddings = create_test_embeddings(dim, num_vectors)
    
    # Test 1: Basic Projections
    print("\nTesting basic projections...")
    results["projections"] = {}
    
    for strength in [0.1, 0.5, 0.9]:
        center = mx.random.normal((dim,))
        field = TransformationField(
            center=center,
            transform_type=TransformationType.PROJECTION,
            strength=strength
        )
        
        transformed = core.apply_field(embeddings, field)
        
        # Calculate metrics
        orig_norms = mx.linalg.norm(embeddings, axis=-1)
        trans_norms = mx.linalg.norm(transformed, axis=-1)
        delta_norms = (trans_norms - orig_norms) / orig_norms
        
        results["projections"][f"strength_{strength}"] = {
            "mean_delta_norm": float(mx.mean(delta_norms)),
            "std_delta_norm": float(mx.std(delta_norms))
        }
    
    # Test 2: Semantic Lenses
    print("\nTesting semantic lenses...")
    results["lenses"] = {}
    
    # Create related concept pairs
    base = mx.random.normal((dim,))
    concepts = {
        "base": base,
        "similar": base + 0.1 * mx.random.normal((dim,)),
        "opposite": -base + 0.1 * mx.random.normal((dim,))
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
    
    print("\nProjection Tests:")
    for strength, metrics in results["projections"].items():
        print(f"\nStrength {strength}:")
        print(f"Mean Δnorm: {metrics['mean_delta_norm']:+.2%}")
        print(f"Std Δnorm:  {metrics['std_delta_norm']:.2%}")
    
    print("\nLens Effect Tests:")
    for concept, metrics in results["lenses"].items():
        print(f"\nEffect on {concept}:")
        print(f"Before: {metrics['before_mean_sim']:+.2f}")
        print(f"After:  {metrics['after_mean_sim']:+.2f}")
        print(f"Delta:  {metrics['delta']:+.2f}")
