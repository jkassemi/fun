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

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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
        self.light = nn.Linear(in_features=152064, out_features=1, bias=False)
        self.embed_tokens = self.formless.model.embed_tokens

        # will transform the embeddings along a direction between two tokens
        self.embed_transform_start = self.embed_tokens(
            tokenizer.encode("red", return_tensors="pt").to(device)
        )
        self.embed_transform_stop = self.embed_tokens(
            tokenizer.encode("blue", return_tensors="pt").to(device)
        )
        self.direction = (
            (self.embed_transform_stop - self.embed_transform_start)
            / torch.norm(self.embed_transform_stop - self.embed_transform_start)
        ).to(device)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.formless(inputs, output_hidden_states=True)

            # Get the last layer's hidden states
            hidden_states = outputs.hidden_states[-1]

            # Get the embeddings for the last token
            last_token_embeddings = hidden_states[:, -1, :]

            # Apply your transformation (example)
            alpha = 200  # Strength of modification
            modified_embeddings = (
                last_token_embeddings + alpha * self.direction
            )  # Simplified

            # Replace the original embeddings with the modified ones for the generation part
            outputs.logits = self.formless.lm_head(
                modified_embeddings
            )  # Replace the logits calculation

            return outputs


def main():
    # Initialize model
    model = Day1().to(device)
    model.add_spectrum("wood_to_metal", "wood", "metal", weight=5.0)
    model.add_spectrum("cold_to_hot", "cold", "hot", weight=10.0)

    messages = []

    try:
        while True:
            text = util.wrapped_input("> ")

            messages.append({"role": "user", "content": text})
            logger.info("prompt %s", json.dumps(messages))

            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            # Get model outputs with transformed embeddings
            outputs = model.forward(inputs)

            # Get the logits from the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Get probability distribution through softmax
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices[0])

            print("Top-k tokens and probabilities:")
            for i in range(50):
                print(f"{topk_tokens[i]}: {topk_probs[0][i].item():.4f}")

            next_token = torch.argmax(next_token_logits, dim=-1)

            # Decode the token to text
            response = tokenizer.decode(next_token, skip_special_tokens=True)

            messages.append({"role": "assistant", "content": response})
            logger.info("response %s", json.dumps(response))
            print(response)

    except KeyboardInterrupt:
        print("\nExiting chatbot...")
    except Exception as e:
        logger.error("Error occurred: %s", str(e))
        raise


if __name__ == "__main__":
    main()
