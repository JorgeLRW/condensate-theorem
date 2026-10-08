#!/usr/bin/env python3
"""Paired dense-vs-sparse survival sweep for the condensate selector.

Defaults reproduce the paper's grid: Qwen2-0.5B, contexts 2K/8K/16K, supports 97/193/385/769,
five WikiText-2 prefixes per context, R=1, 128 greedy tokens (60 paired runs).

    python scripts/run_survival.py --output results/survival.json
"""

import argparse
import gc
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Settings that must match for a partial run to be resumed; mixing them would mix incompatible rows.
RESUME_KEYS = (
    "model", "dtype", "device", "torch", "transformers", "contexts", "supports",
    "reuse_intervals", "prompts_per_context", "max_new_tokens",
)

from condensate.data import load_wikitext2_text, make_prompts  # noqa: E402
from condensate.decode import SparseDecodeController, dense_reference, evaluate_decode  # noqa: E402


def parse_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def save_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temp.replace(path)


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    modeling = importlib.import_module(
        f"transformers.models.{model.config.model_type}.modeling_{model.config.model_type}"
    )
    return model, tokenizer, modeling.apply_rotary_pos_emb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--contexts", type=parse_ints, default=[2048, 8192, 16384])
    parser.add_argument("--supports", type=parse_ints, default=[97, 193, 385, 769])
    parser.add_argument("--reuse", type=parse_ints, default=[1])
    parser.add_argument("--prompts", type=int, default=5, help="prefixes per context")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default=str(ROOT / "results" / "survival.json"))
    parser.add_argument("--resume", action="store_true",
                        help="continue a partial run already saved to --output; starts fresh if it does not exist")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; the harness runs in float16 on a GPU.")

    model, tokenizer, apply_rope = load_model(args.model)
    text = load_wikitext2_text()
    prompts = make_prompts(tokenizer, text, args.contexts, args.prompts, args.max_new_tokens, model.device)

    document = {
        "metadata": {
            "model": args.model,
            "task": "survival",
            "dtype": "float16",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "contexts": args.contexts,
            "supports": args.supports,
            "algorithms": ["kv_group"],
            "reuse_intervals": args.reuse,
            "prompts_per_context": args.prompts,
            "max_new_tokens": args.max_new_tokens,
            "dense_prefill": True,
            "sparse_policy": "layer-local query and KV cache; per-layer additive head mask during decode",
            "teacher_forcing": (
                "common dense-prefill first token is excluded; scores later reference tokens "
                "from the preceding reference token"
            ),
            "selector_budget": (
                "support is per query head; unique distant KV positions per KV head are separately recorded"
            ),
            "timing_claim": "none; research harness only",
        },
        "survival": [],
    }

    completed = set()
    output = Path(args.output)
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        for key in RESUME_KEYS:
            if previous["metadata"].get(key) != document["metadata"][key]:
                raise SystemExit(
                    f"cannot resume {output}: {key} differs "
                    f"({previous['metadata'].get(key)!r} vs {document['metadata'][key]!r}); use a new --output"
                )
        document["survival"] = previous["survival"]
        completed = {(r["context"], r["prompt_index"], r["support"], r["reuse"]) for r in document["survival"]}
        print(f"resuming: {len(completed)} rows already in {output}", flush=True)

    controller = SparseDecodeController(model, apply_rope)
    try:
        for context in args.contexts:
            for prompt_index, prompt in enumerate(prompts[context]):
                pending = [
                    (reuse, support)
                    for reuse in args.reuse
                    for support in args.supports
                    if (context, prompt_index, support, reuse) not in completed
                ]
                if not pending:
                    continue
                started = time.time()
                base_cache, reference_tokens, reference_losses = dense_reference(
                    model, prompt, args.max_new_tokens
                )
                for reuse in args.reuse:
                    for support in args.supports:
                        if (context, prompt_index, support, reuse) in completed:
                            continue
                        result = evaluate_decode(
                            model, base_cache, reference_tokens, reference_losses,
                            controller, support, reuse,
                        )
                        document["survival"].append({
                            "context": context,
                            "prompt_index": prompt_index,
                            "support": support,
                            "algorithm": "kv_group",
                            "reuse": reuse,
                            **result,
                        })
                        save_json(args.output, document)
                        print(
                            f"N={context} prompt={prompt_index} S={support} R={reuse} "
                            f"Tdiv={result['first_divergence']} "
                            f"TF={result['teacher_forced_match']:.1f}% "
                            f"dPPL={result['delta_ppl_percent']:+.1f}%",
                            flush=True,
                        )
                del base_cache
                gc.collect()
                torch.cuda.empty_cache()
                print(f"completed context={context} prompt={prompt_index} "
                      f"elapsed={time.time() - started:.1f}s", flush=True)
    finally:
        controller.close()
    print(f"wrote {len(document['survival'])} rows to {args.output}")


if __name__ == "__main__":
    main()
