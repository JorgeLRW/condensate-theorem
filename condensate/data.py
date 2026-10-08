"""WikiText-2 prompt construction for the survival sweep."""

import torch
from datasets import load_dataset


def load_wikitext2_text():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return " ".join(
        row["text"].strip()
        for row in dataset
        if len(row["text"].strip()) > 30 and not row["text"].strip().startswith("=")
    )


def make_prompts(tokenizer, text, contexts, prompt_count, max_new_tokens, device):
    """Non-overlapping natural-text prompts; each context gets `prompt_count` distinct prefixes."""
    original_limit = tokenizer.model_max_length
    tokenizer.model_max_length = 1_000_000_000
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    finally:
        tokenizer.model_max_length = original_limit

    prompts = {}
    for context in contexts:
        stride = context + max_new_tokens + 1024
        prompts[context] = []
        for index in range(prompt_count):
            start = index * stride
            chunk = tokens[start : start + context]
            if len(chunk) != context:
                raise ValueError(
                    f"WikiText-2 is too short for {prompt_count} distinct {context}-token prompts"
                )
            prompts[context].append(torch.tensor([chunk], device=device, dtype=torch.long))
    return prompts
