import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Tuple, Optional
import logging
import json
import util
import subprocess
import tempfile
from pathlib import Path
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


class Day1(nn.Module):
    def __init__(self):
        super().__init__()
        self.formless = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.embed_tokens = self.formless.model.embed_tokens
        self.spectrum_components = []  # Store spectrum components and weights

    def add_vector_shift(
        self, start_embed: torch.Tensor, end_embed: torch.Tensor, weight=1.0
    ):
        direction = (end_embed - start_embed) / torch.norm(end_embed - start_embed)
        self.spectrum_components.append((direction, weight))
        self.combined_direction = self.calculate_combined_direction()

    def add_cosine_shift(
        self, start_embed: torch.Tensor, end_embed: torch.Tensor, weight=1.0
    ):
        cos_sim = F.cosine_similarity(start_embed, end_embed)
        direction = cos_sim * end_embed / torch.norm(end_embed)
        self.spectrum_components.append((direction, weight))
        self.combined_direction = self.calculate_combined_direction()

    def emb(self, token: str):
        # TODO: this is super fragile to anything more than a single token
        # will consider some sort of averaging or centering here, but we might
        # just want to let the model work through it.
        return self.embed_tokens(
            tokenizer.encode(token, return_tensors="pt", add_special_tokens=False).to(
                device
            )
        )[0]

    def calculate_combined_direction(self):
        emb_dim = self.embed_tokens.weight.shape[1]
        combined_direction = torch.zeros(1, emb_dim, device=device)

        for direction, weight in self.spectrum_components:
            combined_direction += direction * weight

        if torch.norm(combined_direction) > 0:
            combined_direction = combined_direction / torch.norm(combined_direction)
        return combined_direction

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.formless(inputs, output_hidden_states=True)
            prelogits = outputs.hidden_states[-1]
            modified_prelogits = prelogits + prelogits * self.combined_direction
            outputs.logits = self.formless.lm_head(modified_prelogits)
            return outputs


def generate_response(model, input_ids, max_tokens, should_print: bool = True):
    generated_tokens = []
    current_input_ids = input_ids.clone()

    for _ in range(max_tokens):
        outputs = model.forward(current_input_ids)
        next_token_logits = outputs.logits[:, -1, :]

        # Get probability distribution through softmax
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        # Optional: Print top-k tokens and probabilities
        topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices[0])

        # Select next token
        next_token = torch.argmax(next_token_logits, dim=-1)

        # Check for EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

        if should_print:
            print(
                tokenizer.decode(next_token.item(), skip_special_tokens=False),
                flush=True,
                end="",
            )

        # Append the new token to generated sequence
        generated_tokens.append(next_token.item())
        current_input_ids = torch.cat(
            [current_input_ids, next_token.unsqueeze(0)], dim=1
        )

    return generated_tokens


def main():
    # Initialize model
    model = Day1().to(device)

    def get_embedding_for_token(token: str):
        """Get the numerical token identifier for the given token

        Args:
            token: the token representation as a string
        """
        return model.emb(token)

    def add_vector_shift(start_token: str, end_token: str, weight: float = 1.0):
        """Adds embedding space vector shift to preclassifier

        Args:
            start_token: a single token to define the starting point of the vector
            end_token: a single token to define the ending point of the vector
            weight: the weight to be applied to the transformation
        """
        start_emb = model.emb(start_token)
        end_emb = model.emb(end_token)
        model.add_vector_shift(start_emb, end_emb, weight)

    def add_cosine_shift(start_token: str, end_token: str, weight: float = 1.0):
        """Adds cosine vector shift to preclassifier

        Args:
            start_token: a single token to define the starting point of the vector
            end_token: a single token to define the ending point of the vector
            weight: the weight to be applied to the transformation
        """
        start_emb = model.emb(start_token)
        end_emb = model.emb(end_token)
        model.add_cosine_shift(start_emb, end_emb, weight)

    tools = [get_embedding_for_token, add_vector_shift, add_cosine_shift]

    messages = []
    model.add_cosine_shift(model.emb("red"), model.emb("blue"), weight=1.0)

    try:
        while True:
            text = util.wrapped_input("> ")

            messages.append({"role": "user", "content": text})
            logger.info("prompt %s", json.dumps(messages))

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                tools=tools,
            ).to(device)

            generated_tokens = generate_response(model, inputs, max_tokens=100)
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            messages.append({"role": "assistant", "content": response})
            logger.info("response %s", json.dumps(response))
            print()  # newline at the end of the response

    except KeyboardInterrupt:
        print("\nExiting chatbot...")
    except Exception as e:
        logger.error("Error occurred: %s", str(e))
        raise


if __name__ == "__main__":
    main()
