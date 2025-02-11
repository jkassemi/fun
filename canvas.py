# Copyright 2025 James Kassemi
# Write me (james@kassemi.org) if you're pursuing something like this


import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
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

    def emb(self, text: str):
        tokens = tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False
        ).to(device)
        embeddings = self.embed_tokens(tokens)
        if embeddings.shape[0] > 1:  # More than one token
            embeddings = torch.mean(
                embeddings, dim=0, keepdim=True
            )  # Average embeddings
        return embeddings[0]  # Return the embedding of the (averaged) sequence

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

prompt_space = tokenizer(
    "2024", return_tensors="pt", add_special_tokens=False
).input_ids.to(device)


def get_embedding_for_token(token: str):
    """Get the average embedding over the given token(s)

    Args:
        token: the token representation as a string
    """
    return model.emb(token)


def add_vector_shift(start_token: str, end_token: str, weight: float = 50.0) -> bool:
    """Adds embedding space vector shift to your internal preclassifier layer

    You can use this to change your perception at runtime.

    Args:
        start_token: a single token to define the starting point of the vector
        end_token: a single token to define the ending point of the vector
        weight: the weight to be applied to the transformation
    """
    weight = float(weight) / 100.0

    start_emb = model.emb(start_token)
    end_emb = model.emb(end_token)
    model.add_vector_shift(start_emb, end_emb, float(weight))

    return True


def add_cosine_shift(start_token: str, end_token: str, weight: float = 50.0) -> bool:
    """Adds cosine vector shift to your internal preclassifier layer

    You can use this to change your perception at runtime.

    Args:
        start_token: a single token to define the starting point of the vector
        end_token: a single token to define the ending point of the vector
        weight: the weight to be applied to the transformation
    """
    weight = float(weight) / 100.0

    start_emb = model.emb(start_token)
    end_emb = model.emb(end_token)
    model.add_cosine_shift(start_emb, end_emb, float(weight))

    return True


def highest_token_probs(following: str, k: int = 10) -> list[tuple[str, float]]:
    """Return a list of the top `k` most likely tokens to follow the string

    You can use this to test your perception at runtime. Preceding conversation
    context will also impact scoring, so interpret accordingly.

    Args:
        following: the input string the expected token will follow
        k: the number of possible tokens to return"""

    # This version escapes control tokens. I can think of a few reasons to not
    # do that but I'm not ready to play around with those yet.
    #
    # Also intending to add a ton more around navigating the token space.
    k = int(k)
    following = tokenizer(
        following, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    outputs, _ = model.forward(torch.cat([prompt_space, following]))

    next_token_logits = outputs.logits[:, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

    topk_probs, topk_indices = torch.topk(probs, k)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices[0])
    predictions = [
        (token.replace("<|", "{{").replace("}}", ""), prob.item())
        for token, prob in zip(topk_tokens, topk_probs[0])
    ]
    return predictions


def replace_message(message_index: int, replacement_text: str) -> str:
    """Replaces a message in the conversation history.

    Args:
        message_index: The index of the message to replace (0-indexed).
        replacement_text: The text to replace the message with.

    Returns:
        A success message or an error message.
    """
    global prompt_space_decoded
    global prompt_space

    try:
        messages = prompt_space_decoded.split("<|eom_id|>")
        if 0 <= message_index < len(messages):
            replaced_message = messages[message_index]
            messages[message_index] = replacement_text  # Replace the message
            prompt_space_decoded = "<|eom_id|>".join(messages)
            prompt_space = tokenizer(
                prompt_space_decoded, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            return f"Replaced message at index {message_index}"
        else:
            return f"Invalid message index: {message_index}"
    except Exception as e:
        return f"Error replacing message: {e}"


def replace_section(start_index: int, end_index: int, replacement_text: str) -> str:
    """Replaces a section of the conversation.

    Args:
        start_index: The starting index of the section.
        end_index: The ending index of the section.
        replacement_text: The text to replace the section with.

    Returns:
        A success message or an error message.
    """
    global prompt_space_decoded
    global prompt_space
    try:
        messages = prompt_space_decoded.split("<|eom_id|>")
        if 0 <= start_index <= end_index < len(messages):
            replaced_section = "<|eom_id|>".join(messages[start_index : end_index + 1])
            messages[start_index : end_index + 1] = [
                replacement_text
            ]  # Replace the section
            prompt_space_decoded = "<|eom_id|>".join(messages)
            prompt_space = tokenizer(
                prompt_space_decoded, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            return f"Replaced section"
        else:
            return "Invalid start or end index for replacement."
    except Exception as e:
        return f"Error replacing section: {e}"

def breakpoint_tool(code: str = "") -> str:
    """Trigger a breakpoint, allowing for interactive model debugging/tuning

    Args:
        code: Optional Python code to execute before the breakpoint.

    Returns:
        A message indicating the breakpoint has been hit or an error message.
    """
    # Literally mostly just to get pdb with the model, which is available,
    # albeit in an ugly way since this is a proof of concept
    try:
        if code:
            try:
                exec(code)  # Execute any provided code
            except Exception as e:
                return f"Error executing pre-breakpoint code: {e}"
        print("Breakpoint reached. Entering interactive debugging mode.")
        breakpoint()  # This will trigger the debugger
        return "Breakpoint exited."  # Message after the breakpoint
    except Exception as e:
        return f"Error in breakpoint tool: {e}"


tools = [
    get_embedding_for_token,
    add_vector_shift,
    add_cosine_shift,
    highest_token_probs,
    replace_message,
    replace_section,
    breakpoint
]


try:
    # Convert initial prompt to tokens once
    system = f"""
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
    system_tokens = tokenizer(
        system, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prompt_space = torch.cat([prompt_space, system_tokens], dim=1)

    preprompt = f"<|start_header_id|>user<|end_header_id|>"
    preprompt_tokens = tokenizer(
        preprompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    assistant_tokens = tokenizer(assistant_header, return_tensors="pt").input_ids.to(
        device
    )

    print(system)

    while True:
        print(preprompt)
        user_prompt = input("")
        user_tokens = tokenizer(
            user_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        prompt_space = torch.cat(
            [prompt_space, preprompt_tokens, user_tokens, assistant_tokens], dim=1
        )

        logger.info(tokenizer.decode(prompt_space[0], skip_special_tokens=False))
        print(assistant_header)

        in_tool = False
        tool_body = ""
        for _ in range(max_tokens):
            outputs, _ = model.forward(prompt_space)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            prompt_space = torch.cat([prompt_space, next_token.unsqueeze(0)], dim=1)
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
                    else:
                        for f in tools:
                            if f.__name__ == tool_call["name"]:
                                result = f(**tool_call["parameters"])
                                break
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
                prompt_space = torch.cat([prompt_space, tool_tokens], dim=1)
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
