# Copyright James Kassemi <james@kassemi.org> 2025 US

# ---- 
"""
The code includes utility functions to:

Create code-specific lenses (using as triggers)
Create emotion-based concept relationships
Inspect embeddings for debugging

The main purpose seems to be extending a language
model (like DeepSeek or Llama) with geometric
transformations that can modify the model's behavior
based on:

Token importance
Concept relationships
Context detection
Diffusion fields for emphasis/dampening of certain concepts


"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import numpy.typing as npt

class AdaptiveGeometricTransform(GeometricTransform):
    def __init__(
        self, original_embed: nn.Embedding, num_reference_points: int = 4
    ) -> None:
        super().__init__(original_embed, num_reference_points)
        self.context_lenses: Dict[str, ContextLens] = {}
        self.active_lens = None
        self.context_window = []
        self.window_size = 5

    def add_context_lens(
        self, name: str, trigger_embeddings: List[mx.array], threshold: float = 0.85
    ) -> None:
        """Add a new context-specific lens"""
        self.context_lenses[name] = ContextLens(name, trigger_embeddings, threshold)

    def detect_context(self, embedded: mx.array) -> Optional[str]:
        """Detect which context lens should be active based on embedding patterns"""
        max_similarity = 0.0
        best_lens = None

        # Normalize input embedding
        embedded_norm = embedded / (mx.linalg.norm(embedded) + 1e-6)

        for lens_name, lens in self.context_lenses.items():
            # Check similarity with trigger embeddings
            for trigger in lens.trigger_embeddings:
                trigger_norm = trigger / (mx.linalg.norm(trigger) + 1e-6)
                similarity = mx.sum(embedded_norm * trigger_norm)

                if similarity > max_similarity and similarity > lens.threshold:
                    max_similarity = similarity
                    best_lens = lens_name

        return best_lens

    def apply_context_specific_transform(
        self, embedded: mx.array, context: str
    ) -> mx.array:
        """Apply context-specific transformation"""
        lens = self.context_lenses[context]

        # Get the context-specific transformation
        if context == "code":
            # Apply code-specific transformations (e.g., syntax awareness)
            transformed = self._apply_code_lens(embedded, lens)
        elif context == "technical":
            # Apply technical writing transformations
            transformed = self._apply_technical_lens(embedded, lens)
        else:
            # Default transformation
            transformed = embedded

        return transformed

    def _apply_code_lens(self, embedded: mx.array, lens: ContextLens) -> mx.array:
        """Apply code-specific transformations"""
        # Enhance code-like patterns
        strength = lens.transform_strength

        # Look for code-specific patterns (indentation, symbols, etc)
        pattern_scores = mx.zeros_like(embedded)
        for trigger in lens.trigger_embeddings:
            similarity = mx.matmul(embedded, trigger)
            pattern_scores += similarity.reshape(-1, 1) * trigger

        return embedded + strength * pattern_scores

    def __call__(self, x: mx.array) -> mx.array:
        embedded = super().__call__(x)
        batch_size, seq_len, hidden_size = embedded.shape

        # Update context window
        self.context_window.append(embedded[:, -1:, :])
        if len(self.context_window) > self.window_size:
            self.context_window.pop(0)

        # Detect context for current token
        current_context = self.detect_context(embedded[:, -1:, :])

        if current_context:
            # Apply context-specific transformation
            transformed = self.apply_context_specific_transform(
                embedded, current_context
            )

            # Smooth transition between contexts
            if self.active_lens != current_context:
                # Blend transformations at context boundaries
                alpha = 0.8  # Transition smoothness
                transformed = alpha * transformed + (1 - alpha) * embedded

            self.active_lens = current_context
        else:
            transformed = embedded

        return transformed


class ImportanceTracker:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size: int = hidden_size
        self.importance_scores: Dict[int, float] = {}  # token_position -> importance
        self.concept_mappings: Dict[str, List[int]] = {}  # concept_name -> [positions]

    def mark_important(
        self, positions: List[int], concept_name: str, weight: float = 1.0
    ) -> None:
        for pos in positions:
            self.importance_scores[pos] = weight
        self.concept_mappings[concept_name] = positions

    def get_importance_mask(self, seq_length: int) -> mx.array:
        # Create mask directly using array operations
        mask: mx.array = mx.zeros((seq_length,))

        # Convert to array for vectorized operations
        positions = []
        weights = []
        for pos, weight in self.importance_scores.items():
            if pos < seq_length:
                positions.append(pos)
                weights.append(weight)

        if positions:
            # Create a mask array where each position gets its corresponding weight
            indices = mx.array(positions)
            values = mx.array(weights)

            # Use array indexing to set values
            mask = mx.zeros((seq_length,))
            for idx, value in zip(positions, weights):
                mask = mask.at[idx].add(value)

        return mask


class DiffusionField:
    def __init__(
        self, center: mx.array, strength: float, decay_rate: float = 1.0
    ) -> None:
        self.center: mx.array = center / (
            mx.linalg.norm(center) + 1e-6
        )  # normalized center
        self.strength: float = strength
        self.decay_rate: float = decay_rate

    def apply(self, embeddings: mx.array) -> mx.array:
        # Calculate cosine similarities to center
        similarities: mx.array = mx.matmul(embeddings, self.center)

        # Apply decay function (can be modified for different decay patterns)
        decay: mx.array = mx.exp(-self.decay_rate * (1 - similarities))

        # Calculate influence
        return self.strength * decay.reshape(-1, 1) * self.center


class GeometricTransform(nn.Module):
    def __init__(
        self, original_embed: nn.Embedding, num_reference_points: int = 4
    ) -> None:
        super().__init__()
        self.original_embed: nn.Embedding = original_embed

        self.reference_points: mx.array = mx.random.normal((num_reference_points, 3584))
        self.strengths: mx.array = mx.ones((num_reference_points,))

        # Initialize importance tracking
        self.importance_tracker: ImportanceTracker = ImportanceTracker(3584)

        # Add concept basis vectors
        self.concept_embeddings: Dict[str, mx.array] = {}
        self.diffusion_fields: List[DiffusionField] = []

    def add_concept(self, name: str, embedding: mx.array) -> None:
        """Add a concept embedding to use as a reference point"""
        normalized_embedding: mx.array = embedding / (mx.linalg.norm(embedding) + 1e-6)
        self.concept_embeddings[name] = normalized_embedding

    def apply_importance(self, embedded: mx.array, seq_len: int) -> mx.array:
        """Apply importance weighting to embeddings"""
        importance_mask: mx.array = self.importance_tracker.get_importance_mask(seq_len)
        return embedded * (1 + importance_mask.reshape(-1, 1))

    def add_diffusion_field(
        self, center: mx.array, strength: float, decay_rate: float = 1.0
    ) -> None:
        """Add a new diffusion field to influence embeddings"""
        field: DiffusionField = DiffusionField(center, strength, decay_rate)
        self.diffusion_fields.append(field)

    def __call__(self, x: mx.array) -> mx.array:
        embedded: mx.array = self.original_embed(x)
        batch_size, seq_len, hidden_size = embedded.shape
        embedded_flat: mx.array = embedded.reshape(-1, hidden_size)

        embedded_norm: mx.array = embedded_flat / (
            mx.linalg.norm(embedded_flat, axis=-1, keepdims=True) + 1e-6
        )
        ref_norm: mx.array = self.reference_points / (
            mx.linalg.norm(self.reference_points, axis=-1, keepdims=True) + 1e-6
        )

        similarities: mx.array = mx.matmul(embedded_norm, ref_norm.transpose())

        # apply geometric transformation
        transformed: mx.array = embedded_flat + mx.matmul(
            similarities, self.strengths[:, None] * self.reference_points
        )

        # apply importance weighting
        transformed = self.apply_importance(transformed, seq_len)

        # apply concept-based transformations
        for concept_name, concept_embedding in self.concept_embeddings.items():
            concept_similarity: mx.array = mx.matmul(embedded_norm, concept_embedding)
            transformed = (
                transformed + concept_similarity.reshape(-1, 1) * concept_embedding
            )

        return transformed.reshape(batch_size, seq_len, hidden_size)

    def as_linear(self, x: mx.array) -> mx.array:
        return self.original_embed.as_linear(x)


class ConceptLens:
    """Defines a transformation lens based on concept relationships"""

    def __init__(self, name: str, concepts: List[str]) -> None:
        self.name: str = name
        self.concepts: List[str] = concepts
        self.weights: Dict[Tuple[str, str], float] = {}

    def add_relationship(self, concept1: str, concept2: str, weight: float) -> None:
        """Define relationship strength between concepts"""
        self.weights[(concept1, concept2)] = weight

    def apply_to_transform(self, geometric_transform: GeometricTransform) -> None:
        """Apply this lens to a geometric transform layer"""
        for (c1, c2), weight in self.weights.items():
            if (
                c1 in geometric_transform.concept_embeddings
                and c2 in geometric_transform.concept_embeddings
            ):
                e1: mx.array = geometric_transform.concept_embeddings[c1]
                e2: mx.array = geometric_transform.concept_embeddings[c2]
                # Create a weighted combination
                combined: mx.array = (e1 + weight * e2) / (1 + weight)
                geometric_transform.add_concept(f"{self.name}_{c1}_{c2}", combined)


class ContextLens:
    def __init__(
        self,
        name: str,
        trigger_embeddings: List[mx.array],
        activation_threshold: float = 0.85,
    ) -> None:
        self.name = name
        self.trigger_embeddings = (
            trigger_embeddings  # Embeddings that activate this lens
        )
        self.threshold = activation_threshold
        self.transform_strength = mx.array([1.0])  # Learnable parameter

        # Cache for quick pattern matching
        self.pattern_cache = {}



# Example setup code:
def create_code_lens(model, tokenizer) -> List[mx.array]:
    """Create trigger embeddings for code context"""
    code_tokens = [
        "def",
        "class",
        "return",
        "import",
        "for",
        "    ",
        "#",
        "():",
        "self",
        "->",
    ]
    trigger_embeddings = []

    for token in code_tokens:
        ids = tokenizer.encode(token)
        if len(ids) > 0:
            # Get embedding for first token
            embedding = model.transformer.wte(mx.array([[ids[0]]]))[0, 0]
            trigger_embeddings.append(embedding)

    return trigger_embeddings


def create_emotion_lens() -> ConceptLens:
    lens = ConceptLens(
        "emotional_balance", ["anxiety", "confidence", "catastrophizing", "perspective"]
    )
    lens.add_relationship("anxiety", "perspective", 0.7)
    lens.add_relationship("catastrophizing", "confidence", 0.5)
    return lens


def inspect_embeddings(
    model: nn.Module, tokenizer: AutoTokenizer, text: str = "Hello world"
) -> mx.array:
    # Tokenize
    tokens: List[int] = tokenizer.encode(text)
    print(f"Tokens: {tokens}")

    # Get embeddings
    embeddings: mx.array = model.transformer.wte(mx.array([tokens]))
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Look at first token's embedding
    first_token_embedding: mx.array = embeddings[0, 0]
    print(f"\nFirst token embedding (first 10 dimensions):")
    print(first_token_embedding[:10])

    return embeddings


import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple


class LayerGeometricTransform(nn.Module):
    """Geometric transform that can be applied at any layer in the network"""

    def __init__(
        self,
        hidden_size: int,
        layer_index: int,
        num_reference_points: int = 4,
        reference_init: str = "random",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_index = layer_index

        # Initialize reference points based on strategy
        if reference_init == "random":
            self.reference_points = mx.random.normal(
                (num_reference_points, hidden_size)
            )
        else:
            self.reference_points = mx.zeros((num_reference_points, hidden_size))

        self.strengths = mx.ones((num_reference_points,))

        # Layer-specific parameters
        self.layer_scale = mx.array([1.0])  # Learnable layer-specific scaling
        self.concept_embeddings: Dict[str, mx.array] = {}
        self.attention_biases: Dict[str, mx.array] = {}

    def transform_hidden_states(self, hidden_states: mx.array) -> mx.array:
        """Apply geometric transformation to hidden states"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_size)

        # Normalize hidden states and reference points
        hidden_norm = hidden_flat / (
            mx.linalg.norm(hidden_flat, axis=-1, keepdims=True) + 1e-6
        )
        ref_norm = self.reference_points / (
            mx.linalg.norm(self.reference_points, axis=-1, keepdims=True) + 1e-6
        )

        # Calculate similarities and apply transformation
        similarities = mx.matmul(hidden_norm, ref_norm.transpose())
        transformed = hidden_flat + self.layer_scale * mx.matmul(
            similarities, self.strengths[:, None] * self.reference_points
        )

        # Apply concept modifications
        for concept_embedding in self.concept_embeddings.values():
            concept_similarity = mx.matmul(hidden_norm, concept_embedding)
            transformed = (
                transformed + concept_similarity.reshape(-1, 1) * concept_embedding
            )

        return transformed.reshape(batch_size, seq_len, hidden_size)

    def transform_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Apply geometric transformations to attention components"""
        # Apply concept-based attention biases
        for bias in self.attention_biases.values():
            # Reshape bias for broadcasting
            bias = bias.reshape(1, 1, -1)  # [1, 1, hidden_size]

            # Modify query and key projections
            query = query + query * bias
            key = key + key * bias

        return query, key, value


class MultiLayerGeometricTransform(nn.Module):
    """Manages geometric transformations across multiple layers"""

    def __init__(self, model: nn.Module, num_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers

        # Create transforms for each layer
        self.layer_transforms = [
            LayerGeometricTransform(hidden_size=hidden_size, layer_index=i)
            for i in range(num_layers)
        ]

        # Track layer-wise concepts
        self.layer_concepts: Dict[int, Dict[str, mx.array]] = {
            i: {} for i in range(num_layers)
        }

        # Initialize hooks
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Setup hooks to intercept and transform layer outputs"""

        def create_hook(layer_idx: int) -> callable:
            def hook(
                module: nn.Module, inputs: Tuple[mx.array, ...], output: mx.array
            ) -> mx.array:
                # Apply geometric transformation
                transformed = self.layer_transforms[layer_idx].transform_hidden_states(
                    output
                )
                return transformed

            return hook

        # Attach hooks to transformer layers
        for i, layer in enumerate(self.model.transformer.h):
            # Note: This assumes the model follows standard transformer architecture
            # You might need to adjust the path to layers based on your model
            layer.register_forward_hook(create_hook(i))

    def add_concept_to_layer(
        self, concept_name: str, concept_embedding: mx.array, layer_idx: int
    ) -> None:
        """Add a concept to a specific layer"""
        if 0 <= layer_idx < self.num_layers:
            self.layer_transforms[layer_idx].concept_embeddings[
                concept_name
            ] = concept_embedding
            self.layer_concepts[layer_idx][concept_name] = concept_embedding

    def add_attention_bias(
        self, bias_name: str, bias_vector: mx.array, layer_idx: int
    ) -> None:
        """Add attention bias to a specific layer"""
        if 0 <= layer_idx < self.num_layers:
            self.layer_transforms[layer_idx].attention_biases[bias_name] = bias_vector


def apply_multilayer_transform(model: nn.Module) -> nn.Module:
    """Helper function to apply multi-layer transformation to a model"""
    # Get model dimensions
    hidden_size = model.config.hidden_size
    num_layers = len(model.transformer.h)

    # Create and apply multi-layer transform
    transform = MultiLayerGeometricTransform(
        model=model, num_layers=num_layers, hidden_size=hidden_size
    )

    # Wrap the model's forward pass
    original_forward = model.forward

    def transformed_forward(*args, **kwargs):
        # Apply embedding transformation first
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            model.transformer.wte = GeometricTransform(model.transformer.wte)

        # Then proceed with normal forward pass with layer transformations
        return original_forward(*args, **kwargs)

    model.forward = transformed_forward
    return model


if __name__ == "__main__":
    # checkpoint: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    checkpoint: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    model, tokenizer = load(path_or_hf_repo=checkpoint)

    model.model.embed_tokens = GeometricTransform(model.model.embed_tokens)
    emotional_lens = create_emotion_lens()
    emotional_lens.apply_to_transform(model.model.embed_tokens)

    # mark importance concepts
    model.model.embed_tokens.importance_tracker.mark_important(
        positions=[0, 1, 2],  # Replace with actual token positions
        concept_name="key_requirement",
        weight=1.5,
    )

    # # create dampening field
    # dampening_vector = mx.array([...])  # Vector representing concept to dampen
    # model.model.embed_tokens.add_diffusion_field(
    #     center=dampening_vector,
    #     strength=-0.2,  # Negative strength for dampening
    #     decay_rate=0.5  # Controls how quickly the effect falls off
    # )
    #
    # # create emphasizing field
    # emphasis_vector = mx.array([...])  # Vector representing concept to emphasize
    # model.model.embed_tokens.add_diffusion_field(
    #     center=emphasis_vector,
    #     strength=0.3,   # Positive strength for emphasis
    #     decay_rate=1.0  # Sharper falloff
    # )

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=tokenizer.encode("this is"),
        max_tokens=100,
        verbose=True,
    )

    print(response)


"""

Ah, I see the issue - MLX doesn't have a direct scatter operation 
like some other frameworks. Let's modify the get_importance_mask method
to use available MLX operations instead.

"""


# see, with quotes like the one above, i'd like to instead of suggesting action
# after seeing the issue, express 2 to 3 levels of reasoning on what
# the user is asking, confirmation, and ways to test. with the understanding
# that code models should call tools to test syntax and such as they work
# think through the limited environment you're in. use bash commands.
