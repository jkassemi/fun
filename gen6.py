
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

class ModifiedEmbedding(nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.original_embedding = original_embedding
        self.spectrum_components = {}
        self.combined_direction = None
        
    def add_spectrum(self, name, start_token, end_token, weight=1.0):
        start_embed = self.original_embedding(torch.tensor([start_token]).to(device))
        end_embed = self.original_embedding(torch.tensor([end_token]).to(device))
        direction = (end_embed - start_embed) / torch.norm(end_embed - start_embed)
        self.spectrum_components[name] = {"direction": direction, "weight": weight}
        self.combined_direction = self.calculate_combined_direction()
        
    def calculate_combined_direction(self):
        if not self.spectrum_components:
            return None
        combined_direction = torch.zeros_like(list(self.spectrum_components.values())[0]["direction"])
        for spectrum in self.spectrum_components.values():
            combined_direction += spectrum["direction"] * spectrum["weight"]
        return combined_direction / torch.norm(combined_direction) if torch.norm(combined_direction) > 0 else combined_direction
        
    def forward(self, input_ids):
        embeddings = self.original_embedding(input_ids)
        if self.combined_direction is not None:
            embeddings = embeddings * self.combined_direction
        return embeddings

class Day1(nn.Module):
    def __init__(self):
        super().__init__()
        self.formless = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        print(self.formless.model.embed_tokens)
        self.modified_embedding = ModifiedEmbedding(self.formless.model.embed_tokens)
        print(self.modified_embedding)
        self.formless.model.embed_tokens = self.modified_embedding

    def add_spectrum(self, name, start_token, end_token, weight=1.0):
        start_id = tokenizer.encode(start_token)[0]
        end_id = tokenizer.encode(end_token)[0]
        self.modified_embedding.add_spectrum(name, start_id, end_id, weight)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.formless(inputs)
            return outputs


def main():
    # Initialize model
    model = Day1().to(device)
    messages = []

    model.add_spectrum("wood_to_metal", "wood", "metal", weight=5.0)
    model.add_spectrum("red_to_blue", "red", "blue", weight=50.0)
    model.add_spectrum("start_to_stop", "start", "stop", weight=50.0)

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
