"""
Core geometric transformation operations for embedding spaces.
Built for scale and performance.
"""

import mlx.core as mx
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

class TransformationType(Enum):
    ROTATION = "rotation"
    SCALING = "scaling"
    TRANSLATION = "translation"
    REFLECTION = "reflection"
    PROJECTION = "projection"

@dataclass
class TransformationField:
    """Defines a geometric transformation field in embedding space"""
    center: mx.array  # Center point of transformation
    direction: Optional[mx.array] = None  # Direction vector for asymmetric transforms
    strength: float = 1.0
    decay_rate: float = 1.0
    transform_type: TransformationType = TransformationType.ROTATION

@dataclass
class ConceptLens:
    """A lens that emphasizes certain semantic aspects"""
    name: str
    primary_concepts: List[str]
    strength: float = 0.5
    contrast_concepts: List[str] = None

class GeometricCore:
    """Core geometric operations on embedding spaces"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.fields: List[TransformationField] = []
        self.concept_centers: Dict[str, mx.array] = {}
        self.lenses: Dict[str, ConceptLens] = {}
    
    def create_lens(self, name: str, concepts: List[str], strength: float = 0.5, 
                   contrast_concepts: List[str] = None) -> None:
        """Create a semantic lens from concepts"""
        self.lenses[name] = ConceptLens(name, concepts, strength, contrast_concepts)
    
    def apply_lens(self, embeddings: mx.array, lens_name: str) -> mx.array:
        """Apply a semantic lens transformation"""
        if lens_name not in self.lenses:
            raise KeyError(f"Lens {lens_name} not found")
            
        lens = self.lenses[lens_name]
        result = embeddings
        
        # Enhance primary concepts
        for concept in lens.primary_concepts:
            if concept in self.concept_centers:
                field = TransformationField(
                    center=self.concept_centers[concept],
                    strength=lens.strength,
                    transform_type=TransformationType.PROJECTION
                )
                result = self.apply_field(result, field)
        
        # Reduce contrast concepts if specified
        if lens.contrast_concepts:
            for concept in lens.contrast_concepts:
                if concept in self.concept_centers:
                    field = TransformationField(
                        center=self.concept_centers[concept],
                        strength=-lens.strength * 0.5,  # Softer contrast
                        transform_type=TransformationType.PROJECTION
                    )
                    result = self.apply_field(result, field)
        
        return result
        
    def add_field(self, field: TransformationField) -> None:
        """Add a transformation field"""
        self.fields.append(field)
        
    def add_concept(self, name: str, embedding: mx.array) -> None:
        """Register a concept's center in embedding space"""
        self.concept_centers[name] = embedding / (mx.linalg.norm(embedding) + 1e-6)
    
    def apply_field(self, embeddings: mx.array, field: TransformationField) -> mx.array:
        """Apply a single transformation field"""
        # Normalize vectors
        center = field.center / (mx.linalg.norm(field.center) + 1e-6)
        
        if field.transform_type == TransformationType.ROTATION:
            # Compute rotation in the plane defined by center and direction
            if field.direction is not None:
                direction = field.direction / (mx.linalg.norm(field.direction) + 1e-6)
                plane_normal = mx.linalg.cross(center, direction)
                plane_normal = plane_normal / (mx.linalg.norm(plane_normal) + 1e-6)
                
                # Project embeddings onto rotation plane
                proj = embeddings - mx.matmul(embeddings, plane_normal.reshape(-1, 1)) * plane_normal
                
                # Apply rotation
                angle = field.strength
                cos_angle = mx.cos(angle)
                sin_angle = mx.sin(angle)
                
                rotated = cos_angle * proj + sin_angle * mx.linalg.cross(plane_normal, proj)
                return embeddings + (rotated - proj)
            
            # Simple rotation around center
            sim = mx.matmul(embeddings, center) / (mx.linalg.norm(embeddings, axis=-1) + 1e-6)
            return embeddings + field.strength * sim.reshape(-1, 1) * center
            
        elif field.transform_type == TransformationType.SCALING:
            # Scale along center direction
            sim = mx.matmul(embeddings, center) / (mx.linalg.norm(embeddings, axis=-1) + 1e-6)
            scale_factor = 1.0 + field.strength * sim.reshape(-1, 1)
            return embeddings * scale_factor
            
        elif field.transform_type == TransformationType.TRANSLATION:
            # Translate in direction
            if field.direction is not None:
                direction = field.direction / (mx.linalg.norm(field.direction) + 1e-6)
                return embeddings + field.strength * direction
            return embeddings + field.strength * center
            
        elif field.transform_type == TransformationType.REFLECTION:
            # Reflect across hyperplane defined by center normal
            sim = mx.matmul(embeddings, center) / (mx.linalg.norm(embeddings, axis=-1) + 1e-6)
            return embeddings - 2 * field.strength * sim.reshape(-1, 1) * center
            
        elif field.transform_type == TransformationType.PROJECTION:
            # Project onto subspace defined by center while preserving some original
            sim = mx.matmul(embeddings, center) / (mx.linalg.norm(embeddings, axis=-1) + 1e-6)
            projection = sim.reshape(-1, 1) * center
            return (1 - field.strength) * embeddings + field.strength * projection
            
        return embeddings
    
    def apply_all_fields(self, embeddings: mx.array) -> mx.array:
        """Apply all transformation fields"""
        result = embeddings
        for field in self.fields:
            result = self.apply_field(result, field)
        return result
    
    def get_concept_similarity(self, embeddings: mx.array, concept: str) -> mx.array:
        """Get similarity between embeddings and a concept"""
        if concept not in self.concept_centers:
            raise KeyError(f"Concept {concept} not found")
            
        center = self.concept_centers[concept]
        return mx.matmul(embeddings, center) / (mx.linalg.norm(embeddings, axis=-1) + 1e-6)
