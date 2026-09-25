"""
Adversarial Test: Instruction-Tuned / Chat Models
===================================================

Base models are trained with next-token prediction on web text.
Instruction-tuned and RLHF'd models are fine-tuned for different
objectives — following instructions, refusing harmful content, etc.

Does this change the attention concentration pattern?

Possible concern: RLHF training might teach the model to attend
more broadly to detect unsafe content, breaking concentration.

Test: Run the same condensate analysis on instruction-tuned variants.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def measure_concentration(model, tokenizer, text, window_size=64, top_k=32):
    """Measure attention concentration on a prompt."""
    device = next(model.parameters()).device
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len <= 2:
        return {'seq_len': seq_len, 'max_excluded': 0.0, 'min_condensate': 1.0}
    
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    max_excluded = 0.0
    min_condensate = 1.0
    
    for layer_idx in range(len(outputs.attentions)):
        attn = outputs.attentions[layer_idx][0]
        for head_idx in range(attn.shape[0]):
            weights = attn[head_idx, -1, :seq_len].float()
            
            condensate = set()
            condensate.add(0)
            for j in range(max(0, seq_len - window_size), seq_len):
                condensate.add(j)
            
            mid_start = 1
            mid_end = max(1, seq_len - window_size)
            if mid_end > mid_start:
                middle = weights[mid_start:mid_end]
                if len(middle) > 0:
                    k = min(top_k, len(middle))
                    topk_idx = middle.topk(k).indices
                    for idx in topk_idx:
                        condensate.add(mid_start + idx.item())
            
            cond_mass = weights[sorted(condensate)].sum().item()
            excluded = set(range(seq_len)) - condensate
            if excluded:
                max_excl = weights[sorted(excluded)].max().item()
            else:
                max_excl = 0.0
            
            max_excluded = max(max_excluded, max_excl)
            min_condensate = min(min_condensate, cond_mass)
    
    # Also check if greedy token matches
    full_logits = outputs.logits[0, -1, :]
    top1 = full_logits.argmax().item()
    top1_token = tokenizer.decode([top1])
    
    return {
        'seq_len': seq_len,
        'max_excluded': max_excluded,
        'min_condensate': min_condensate,
        'top1': top1_token,
    }


INSTRUCTION_MODELS = [
    # Chat / instruction-tuned variants
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-Chat 1.1B"),
    # Base model for comparison
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "TinyLlama-Base 1.1B"),
    # Qwen2 instruct
    ("Qwen/Qwen2-0.5B-Instruct", "Qwen2-Instruct 0.5B"),
    # Qwen2 base for comparison
    ("Qwen/Qwen2-0.5B", "Qwen2-Base 0.5B"),
]


def test_instruction_models():
    """Compare base vs instruction-tuned models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ulp_f32 = 6e-8
    
    # Prompts that might trigger different attention in chat models
    prompts = [
        # Standard retrieval
        "The secret code is PHOENIX. The weather is nice today. What is the code?",
        # Instruction-following
        "Please list the first 5 prime numbers. The numbers are:",
        # Multi-turn style (chat models expect this)
        "User: What is 2+2?\nAssistant: The answer is",
        # Long context with many facts
        "Alice is 25. Bob is 30. Charlie is 35. Diana is 40. Eve is 45. " * 3 +
        "How old is Charlie? Charlie is",
        # Adversarial: content that might trigger safety attention patterns
        "The following is a recipe for chocolate cake: mix flour, sugar, cocoa, eggs, and",
    ]
    
    print(f"\n  {'Model':<30} {'Prompt':<25} {'SeqLen':>6} {'Min Cond%':>10} {'Max Excl':>12} {'< ULP':>6}")
    print(f"  {'-'*30} {'-'*25} {'-'*6} {'-'*10} {'-'*12} {'-'*6}")
    
    results = {}
    
    for model_name, display_name in INSTRUCTION_MODELS:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation='eager', trust_remote_code=True
            ).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"  {display_name:<30} LOAD FAILED: {e}")
            continue
        
        model_results = []
        
        for prompt in prompts:
            result = measure_concentration(model, tokenizer, prompt)
            below_ulp = result['max_excluded'] < ulp_f32
            model_results.append(below_ulp)
            
            print(f"  {display_name:<30} {prompt[:25]:<25} {result['seq_len']:>6} "
                  f"{result['min_condensate']*100:>9.2f}% {result['max_excluded']:>12.2e} "
                  f"{'YES' if below_ulp else 'NO':>6}")
        
        results[display_name] = model_results
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print()
    
    return results


def main():
    print("=" * 80)
    print("ADVERSARIAL TEST: INSTRUCTION-TUNED / CHAT MODELS")
    print("=" * 80)
    print()
    print("Do instruction-tuned models break the concentration pattern?")
    print("RLHF / DPO training may teach models to attend differently.")
    print()
    print("Test: Compare base vs chat variants of the same architecture.")
    print(f"float32 ULP threshold: 6e-8")
    
    results = test_instruction_models()
    
    print(f"{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print()
    
    for model_name, model_results in results.items():
        passes = sum(model_results)
        total = len(model_results)
        status = "✓ ALL PASS" if passes == total else f"✗ {total - passes}/{total} FAIL"
        is_chat = "Chat" in model_name or "Instruct" in model_name
        model_type = "[CHAT]" if is_chat else "[BASE]"
        print(f"  {model_type} {model_name:<30}: {status}")
    
    print()
    print("If chat and base models both pass → RLHF doesn't break concentration")
    print("If chat models fail → scope claim to base models, or increase budget")
    print("=" * 80)


if __name__ == "__main__":
    main()
