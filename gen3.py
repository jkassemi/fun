import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Tuple, Optional
import logging
import json


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Day1(nn.Module):
    def __init__(self, formless):
        super().__init__()
        self.formless = formless
        self.light = nn.Linear(in_features=152064, out_features=1, bias=False)

    def forward(self, inputs, attention_mask=None, past_key_values=None):
        outputs = self.formless(
            input_ids=inputs,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            return_dict=True,
        )

        # Get the last hidden state
        y = outputs.logits[:, -1]
        p = torch.softmax(y, dim=-1)

        # Get top 20 tokens
        _, pi = torch.sort(p, descending=True)
        top_tokens = pi[0][:20]

        x = []
        for i in top_tokens:
            try:
                x.append(tokenizer.decode(i.item()))
            except Exception:
                breakpoint()
        print(x)

        return outputs

    @property
    def layers(self):
        return self.formless.transformer.h  # Adjust based on model architecture


# Model initialization
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
formless = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# If you have a GPU with Metal support (Mac):
# if torch.backends.mps.is_available():
device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

model = Day1(formless).to(device)

# Logging setup
import friendly_names

logger = logging.getLogger(__name__)
session = friendly_names.generate()
logging.basicConfig(
    level=logging.DEBUG,
    format=f"{session} %(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(os.getcwd(), "my_app.log"))],
)


def wrapped_input(prompt: str = ">") -> str:
    code = "5059004450"
    text = input(prompt)
    return f"<{code}>{text}</{code}>"


def generate(
    model,
    inputs,
    attention_mask,
    pad_token_id,
    eos_token_id,
    max_tokens=1,
    verbose=True,
):
    with torch.no_grad():
        outputs = model.formless.generate(
            inputs,
            max_new_tokens=max_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.7,
            attention_mask=attention_mask,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    return response


messages = []
while True:
    text = wrapped_input(">")
    messages.append({"role": "user", "content": text})
    logger.info("prompt %s", json.dumps(messages))

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(inputs)

    response = generate(
        model,
        inputs=inputs,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        verbose=True,
        max_tokens=1,
    )
    messages.append({"role": "assistant", "content": response})
    logger.info("response %s", json.dumps(response))
    print(response)
