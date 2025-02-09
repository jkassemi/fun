import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Tuple, Optional
import logging
import json
import util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


class Day1(nn.Module):
    def __init__(self):
        super().__init__()
        self.formless = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.light = nn.Linear(in_features=152064, out_features=1, bias=False)
        self.past_key_values = None

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.formless(
                inputs, past_key_values=self.past_key_values, use_cache=True
            )
            # Store past key values for potential continued generation
            self.past_key_values = outputs.past_key_values
            return outputs


def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize model
    model = Day1().to(device)

    messages = []

    try:
        while True:
            text = util.wrapped_input("> ")
            if not text:
                continue

            messages.append({"role": "user", "content": text})
            logger.info("prompt %s", json.dumps(messages))

            # Prepare input
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            # Get model outputs
            outputs = model.forward(inputs)

            # Get the logits from the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Sample from the logits (you can modify this sampling strategy)
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
