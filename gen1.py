# Copyright James Kassemi <james@kassemi.org> 2025 US

# ---- 

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from transformers import AutoTokenizer

checkpoint = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model, tokenizer = load(path_or_hf_repo=checkpoint)


class ImportanceTracker:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.importance_scores = {}  # token_position -> importance
        self.concept_mappings = {}  # concept_name -> [positions]

    def mark_important(self, positions, concept_name, weight=1.0):
        for pos in positions:
            self.importance_scores[pos] = weight
        self.concept_mappings[concept_name] = positions

    def get_importance_mask(self, seq_length):
        # Create base mask
        mask = mx.zeros((seq_length,))

        # Create updates for important positions
        updates = []
        indices = []
        for pos, weight in self.importance_scores.items():
            if pos < seq_length:
                indices.append(pos)
                updates.append(weight)

        if indices:
            # Update mask all at once using scatter
            mask = mx.scatter(mask, mx.array(indices), mx.array(updates), axis=0)

        return mask


class DiffusionField:
    def __init__(self, center, strength, decay_rate=1.0):
        self.center = center / (mx.linalg.norm(center) + 1e-6)  # normalized center
        self.strength = strength
        self.decay_rate = decay_rate

    def apply(self, embeddings):
        # Calculate cosine similarities to center
        similarities = mx.matmul(embeddings, self.center)

        # Apply decay function (can be modified for different decay patterns)
        decay = mx.exp(-self.decay_rate * (1 - similarities))

        # Calculate influence
        return self.strength * decay.reshape(-1, 1) * self.center


# fun: we could potentially, with traditional training,
# find some text or concept that we find useful or interesting or nice:
#
#   > https://github.com/dwhinham/mt32-pi/blob/075b52809e77420c6e80828825fe42430336b369/README.md#%EF%B8%8F-contributing
#   >
#   > This project is generally quite stable and very usable, but still considered by its author to be in early stages of development.
#   > Hence, please DO NOT work on large features and open pull requests without prior discussion. There is a strong possibility that work-in-progress code for proposed features already exists, but may not yet be public, and your work will have to be rejected.
#   > Trivial changes to the code that fix issues are always welcome, as are improvements to documentation, and hardware/software compatibility reports.
# 
# and we could push the model toward that, or could push parts of the model toward that. 
# some existing parts of the model may associate with it in ways they shouldn't. we should be able 
# to explore that, as well. really need some interface to evaluate logits - autocomplete interface.
#
#
# maybe the future code editor is actually just a basic code editor. lol neovim

class GeometricTransform(nn.Module):
    def __init__(self, original_embed, num_reference_points=4):
        super().__init__()
        self.original_embed = original_embed

        self.reference_points = mx.random.normal((num_reference_points, 3072))
        self.strengths = mx.ones((num_reference_points,))

        # Initialize importance tracking
        self.importance_tracker = ImportanceTracker(3072)

        # Add concept basis vectors
        self.concept_embeddings = {}
        self.diffusion_fields = []

    def add_concept(self, name, embedding):
        """Add a concept embedding to use as a reference point"""
        normalized_embedding = embedding / (mx.linalg.norm(embedding) + 1e-6)
        self.concept_embeddings[name] = normalized_embedding

    def apply_importance(self, embedded, seq_len):
        """Apply importance weighting to embeddings"""
        importance_mask = self.importance_tracker.get_importance_mask(seq_len)
        return embedded * (1 + importance_mask.reshape(-1, 1))

    def add_diffusion_field(self, center, strength, decay_rate=1.0):
        """Add a new diffusion field to influence embeddings"""
        field = DiffusionField(center, strength, decay_rate)
        self.diffusion_fields.append(field)


    def __call__(self, x):
        embedded = self.original_embed(x)
        batch_size, seq_len, hidden_size = embedded.shape
        embedded_flat = embedded.reshape(-1, hidden_size)

        embedded_norm = embedded_flat / (
            mx.linalg.norm(embedded_flat, axis=-1, keepdims=True) + 1e-6
        )
        ref_norm = self.reference_points / (
            mx.linalg.norm(self.reference_points, axis=-1, keepdims=True) + 1e-6
        )

        similarities = mx.matmul(embedded_norm, ref_norm.transpose())

        # apply geometric transformation
        transformed = embedded_flat + mx.matmul(
            similarities, self.strengths[:, None] * self.reference_points
        )

        # apply importance weighting
        transformed = self.apply_importance(transformed, seq_len)

        # apply concept-based transformations
        for concept_name, concept_embedding in self.concept_embeddings.items():
            concept_similarity = mx.matmul(embedded_norm, concept_embedding)
            transformed = (
                transformed + concept_similarity.reshape(-1, 1) * concept_embedding
            )

        return transformed.reshape(batch_size, seq_len, hidden_size)

    def as_linear(self, x):
        return self.original_embed.as_linear(x)


class ConceptLens:
    """Defines a transformation lens based on concept relationships"""

    def __init__(self, name, concepts):
        self.name = name
        self.concepts = concepts
        self.weights = {}

    def add_relationship(self, concept1, concept2, weight):
        """Define relationship strength between concepts"""
        self.weights[(concept1, concept2)] = weight

    def apply_to_transform(self, geometric_transform):
        """Apply this lens to a geometric transform layer"""
        for (c1, c2), weight in self.weights.items():
            if (
                c1 in geometric_transform.concept_embeddings
                and c2 in geometric_transform.concept_embeddings
            ):
                e1 = geometric_transform.concept_embeddings[c1]
                e2 = geometric_transform.concept_embeddings[c2]
                # Create a weighted combination
                combined = (e1 + weight * e2) / (1 + weight)
                geometric_transform.add_concept(f"{self.name}_{c1}_{c2}", combined)


# Example usage:
def create_emotion_lens():
    lens = ConceptLens(
        "emotional_balance", ["anxiety", "confidence", "catastrophizing", "perspective"]
    )
    lens.add_relationship("anxiety", "perspective", 0.7)
    lens.add_relationship("catastrophizing", "confidence", 0.5)
    return lens


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

def inspect_embeddings(model, tokenizer, text="Hello world"):
    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"Tokens: {tokens}")

    # Get embeddings
    embeddings = model.transformer.wte(mx.array([tokens]))
    print(f"\nEmbedding shape: {embeddings.shape}")
    # shape will be [batch_size, sequence_length, embedding_dimension]

    # Look at first token's embedding
    first_token_embedding = embeddings[0, 0]
    print(f"\nFirst token embedding (first 10 dimensions):")
    print(first_token_embedding[:10])

    return embeddings


# Usage:
# embeddings = inspect_embeddings(model, tokenizer)

# class GeometricTransform(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.hidden_size = model.args.hidden_size
#         self.embed = model.model.embed_tokens
#
#     def __call__(self, x):
#         embedded = self.embed(x)
#         return embedded
#
#         # Normalize embeddings
#         x_norm = x / mx.linalg.norm(x, axis=-1, keepdims=True)
#         ref_norm = self.reference_points / mx.linalg.norm(
#             self.reference_points, axis=-1, keepdims=True
#         )
#
#         # Calculate similarities
#         similarities = mx.matmul(x_norm, ref_norm.T)
#
#         # Apply transformations
#         transformed = x
#         for i in range(len(self.transform_strengths)):
#             contribution = similarities[:, :, i : i + 1] * self.transform_strengths[i]
#             delta = mx.matmul(contribution, self.reference_points[i : i + 1])
#             transformed = transformed + delta
#
#         return transformed
#
#
# def apply_transform(model, transform_layer):
#     def transformed_forward(x):
#         # Get original embeddings
#         embedded = model.transformer.wte(x)
#         # Apply transformation
#         transformed = transform_layer(embedded)
#         # Continue with model forward pass
#         return model.forward_with_embedding(transformed)
#
#     return transformed_forward
#


# If I get lost in here, you'll have to decide whether
# to come looking for me. I don't advise it unless you're
# into low wages, bitter cold, long months of complete
# darkness, constant danger, and doubtful return.
