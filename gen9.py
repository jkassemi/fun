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
from transformers.utils import get_json_schema


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_tokens = 1000

token_id_python_tag = tokenizer.encode("<|python_tag|>", add_special_tokens=False)[0]
end_of_turn_tag = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
end_of_message_tag = tokenizer.encode("<|eom_id|>", add_special_tokens=False)[0]
start_header_tag = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)[0]
end_header_tag = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)[0]


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

    def forward(self, input_ids, past_key_values=None):
        with torch.no_grad():
            outputs = self.formless(
                input_ids,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            prelogits = outputs.hidden_states[-1]
            prelogits[:, -1, :] = prelogits[:, -1, :] * (1 + self.combined_direction)
            outputs.logits = self.formless.lm_head(prelogits)

            past_key_values = outputs.past_key_values
            return outputs, past_key_values


model = Day1().to(device)
model.add_cosine_shift(model.emb("light"), model.emb("good"), 1.0)


def get_embedding_for_token(token: str):
    """Get the numerical token identifier for the given token

    Args:
        token: the token representation as a string
    """
    return model.emb(token)


def add_vector_shift(start_token: str, end_token: str, weight: int = 50):
    """Adds embedding space vector shift to your internal preclassifier layer

    You can use this to change your perception at runtime.

    Args:
        start_token: a single token to define the starting point of the vector
        end_token: a single token to define the ending point of the vector
        weight: the weight to be applied to the transformation: 0-100
    """
    if float(weight) < 0:
        raise ValueError("must be greater than or equal to 0")
    if float(weight) > 100:
        raise ValueError("cannot be more than 100")

    weight = float(weight) / 100.0

    start_emb = model.emb(start_token)
    end_emb = model.emb(end_token)
    model.add_vector_shift(start_emb, end_emb, float(weight))


def add_cosine_shift(start_token: str, end_token: str, weight: int = 50):
    """Adds cosine vector shift to your internal preclassifier layer

    You can use this to change your perception at runtime.

    Args:
        start_token: a single token to define the starting point of the vector
        end_token: a single token to define the ending point of the vector
        weight: the weight to be applied to the transformation: 0-100
    """
    if float(weight) < 0:
        raise ValueError("must be greater than or equal to 0")
    if float(weight) > 100:
        raise ValueError("cannot be more than 100")

    weight = float(weight) / 100.0

    start_emb = model.emb(start_token)
    end_emb = model.emb(end_token)
    model.add_cosine_shift(start_emb, end_emb, float(weight))


tools = [get_embedding_for_token, add_vector_shift, add_cosine_shift]

# Initialize conversation with system prompt
initial_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Today Date: 10 Feb 2025

Should you choose to call a tool, only call one at a time.
Put them in the following format with no other content:

Here is a list of functions in JSON format you can invoke:
{" ".join([json.dumps(get_json_schema(tool)) for tool in tools])}

Here is how you can invoke a tool named `toolname`:
{{"name": "toolname", "parameters": {{"numberParam": 0.5, "stringParam": "mystring"}}}}
<|eot_id|>
"""

try:
    # Convert initial prompt to tokens once
    print(initial_prompt)
    current_input_ids = tokenizer(
        initial_prompt, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)

    while True:
        preprompt = f"<|start_header_id|>user<|end_header_id|>"
        print(preprompt)
        preprompt_tokens = tokenizer(
            preprompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        user_prompt = input("")
        user_tokens = tokenizer(
            user_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        current_input_ids = torch.cat([current_input_ids, user_tokens], dim=1)

        assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        print(assistant_header)
        assistant_tokens = tokenizer(
            assistant_header, return_tensors="pt"
        ).input_ids.to(device)
        current_input_ids = torch.cat([current_input_ids, assistant_tokens], dim=1)

        logger.info(tokenizer.decode(current_input_ids[0], skip_special_tokens=False))

        past_key_values = None
        # Generate response
        in_tool = False
        tool_body = ""
        for _ in range(max_tokens):
            outputs, past_key_values = model.forward(current_input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            current_input_ids = torch.cat(
                [current_input_ids, next_token.unsqueeze(0)], dim=1
            )
            print(tokenizer.decode(next_token.item()), end="", flush=True)

            if next_token.item() == tokenizer.eos_token_id:
                break
            if next_token.item() == token_id_python_tag:
                in_tool = True
                tool_body = ""
            elif (
                next_token.item()
                in (end_of_turn_tag, end_of_message_tag, start_header_tag)
                and in_tool
            ):
                in_tool = False
                try:
                    tool_call = json.loads(tool_body)
                    if "name" not in tool_call:
                        result = "error: unsupported tool call format"
                    elif tool_call["name"] == "get_embedding_for_token":
                        result = get_embedding_for_token(**tool_call["parameters"])
                    elif tool_call["name"] == "add_vector_shift":
                        result = add_vector_shift(**tool_call["parameters"])
                    elif tool_call["name"] == "add_cosine_shift":
                        result = add_cosine_shift(**tool_call["parameters"])
                    elif tool_call["name"] == "mark":
                        result = mark(**tool_call["parameters"])
                    else:
                        result = f"error: unknown tool {tool_call['name']}"
                except json.JSONDecodeError:
                    result = "failed to parse tool call body"
                except Exception as e:
                    import traceback  # Import traceback module

                    trace = traceback.format_exc()  # Capture the full traceback
                    logger.error(
                        f"Tool execution failed: {str(e)}\nTraceback:\n{trace}"
                    )  # Log with traceback
                    result = f"tool call error: {str(e)}"
                # Add tool result to conversation
                tool_result = (
                    f"<|start_header_id|>ipython<|end_header_id|>\n{result}<|eot_id|>"
                )
                tool_tokens = tokenizer(
                    tool_result, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(device)
                current_input_ids = torch.cat([current_input_ids, tool_tokens], dim=1)
                print(tool_result, end="", flush=True)
                logger.info(tool_result)

                continue
            elif in_tool:
                tool_body += tokenizer.decode(next_token.item())

        print()  # newline after response

except KeyboardInterrupt:
    print("\nExiting chatbot...")
except Exception as e:
    logger.error("Error occurred: %s", str(e))
    raise
