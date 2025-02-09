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
            
            # Get probability distribution through softmax
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Get full distribution and indices
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            
            # Create temporary file with distribution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for prob, idx in zip(sorted_probs[:50], sorted_indices[:50]):  # Top 50 tokens
                    token = tokenizer.decode([idx])
                    f.write(f"{token!r}  [{prob:.4f}]\n")
                temp_path = f.name

            # Open vim for editing
            try:
                subprocess.run(['vim', temp_path], check=True)
                
                # Read back the reordered tokens
                with open(temp_path) as f:
                    lines = f.readlines()
                
                # Parse reordered tokens
                reordered_tokens = []
                for line in lines:
                    if line.strip():
                        token_str = line.split('[')[0].strip()
                        # Remove quotes from token string
                        token_str = token_str[1:-1] if token_str.startswith('"') else token_str
                        reordered_tokens.append(token_str)
                
                # Convert reordered tokens back to indices
                reordered_indices = []
                for token in reordered_tokens:
                    token_id = tokenizer.encode(token, add_special_tokens=False)[0]
                    reordered_indices.append(token_id)
                
                # Create new probability distribution (using softmax with temperature)
                temperature = 2.0  # Adjust this to control how "sharp" the distribution is
                positions = torch.arange(len(reordered_indices), device=device, dtype=torch.float)
                new_logits = -positions / temperature
                new_probs = torch.nn.functional.softmax(new_logits, dim=0)
                
                # Select next token based on new distribution
                next_token = torch.tensor([reordered_indices[0]], device=device)
                
            except subprocess.CalledProcessError:
                # If vim exits with error, fall back to original argmax
                next_token = torch.argmax(next_token_logits, dim=-1)
            finally:
                # Clean up temp file
                Path(temp_path).unlink()
            
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
