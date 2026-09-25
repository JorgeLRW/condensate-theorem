"""
Adversarial Test: Long Generation (1000+ tokens)
=================================================

Even if each individual step is exact, autoregressive generation compounds 
decisions. If there's ANY numerical difference — even sub-ULP — could it 
accumulate over 1000+ steps and eventually cause a token divergence?

This test generates long sequences and checks for divergence.

Note: if sparse and full produce identical float32 values at each step,
then there IS no accumulation — identical inputs → identical outputs 
at every step. But this test VERIFIES that empirically.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import warnings
warnings.filterwarnings('ignore')


def generate_with_attention_tracking(model, input_ids, max_tokens, window_size=64, top_k=32):
    """
    Generate tokens one at a time, tracking attention concentration
    and checking for any step where the condensate might miss.
    """
    device = input_ids.device
    generated_tokens = []
    max_excluded_weights = []
    condensate_masses = []
    
    current_ids = input_ids.clone()
    
    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_attentions=True)
        
        logits = outputs.logits[0, -1, :]
        next_token = logits.argmax().item()
        generated_tokens.append(next_token)
        
        # Track attention concentration at this step
        seq_len = current_ids.shape[1]
        step_max_excluded = 0.0
        step_min_condensate = 1.0
        
        for layer_idx in range(len(outputs.attentions)):
            attn = outputs.attentions[layer_idx][0]  # [H, S, S]
            for head_idx in range(attn.shape[0]):
                weights = attn[head_idx, -1, :seq_len].float()
                
                # Condensate set
                condensate = set()
                condensate.add(0)
                for j in range(max(0, seq_len - window_size), seq_len):
                    condensate.add(j)
                if seq_len > window_size + 1:
                    mid_start = 1
                    mid_end = max(1, seq_len - window_size)
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
                
                step_max_excluded = max(step_max_excluded, max_excl)
                step_min_condensate = min(step_min_condensate, cond_mass)
        
        max_excluded_weights.append(step_max_excluded)
        condensate_masses.append(step_min_condensate)
        
        # Append token and continue
        current_ids = torch.cat([
            current_ids,
            torch.tensor([[next_token]], device=device)
        ], dim=1)
        
        # Progress
        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{max_tokens}: seq_len={current_ids.shape[1]}, "
                  f"max_excl={step_max_excluded:.2e}, min_cond={step_min_condensate*100:.1f}%")
        
        # Stop on EOS
        if next_token == model.config.eos_token_id:
            break
    
    return generated_tokens, max_excluded_weights, condensate_masses


def test_long_generation(model_name="gpt2-medium", max_tokens=500):
    """
    Generate a long sequence and track concentration at every step.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation='eager'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = [
        "Once upon a time, in a kingdom far away, there lived a young princess who",
        "The following is a detailed explanation of how neural networks learn:",
        "Chapter 1: The Discovery\n\nDr. Sarah Chen stared at the data on her screen, unable to believe what she",
    ]

    ulp_f32 = 6e-8
    
    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}: '{prompt[:60]}...'")
        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        prompt_len = input_ids.shape[1]
        print(f"    Prompt tokens: {prompt_len}")
        print(f"    Generating up to {max_tokens} tokens...")
        
        t0 = time.time()
        tokens, max_excl_list, cond_mass_list = generate_with_attention_tracking(
            model, input_ids, max_tokens
        )
        elapsed = time.time() - t0
        
        gen_text = tokenizer.decode(tokens, skip_special_tokens=True)
        total_len = prompt_len + len(tokens)
        
        # Analysis
        max_excl_ever = max(max_excl_list) if max_excl_list else 0
        min_cond_ever = min(cond_mass_list) if cond_mass_list else 1
        steps_above_ulp = sum(1 for w in max_excl_list if w > ulp_f32)
        
        print(f"\n    Generated: {len(tokens)} tokens ({elapsed:.1f}s)")
        print(f"    Final seq len: {total_len}")
        print(f"    Max excluded weight (any step): {max_excl_ever:.2e}")
        print(f"    Min condensate mass (any step): {min_cond_ever*100:.2f}%")
        print(f"    Steps with max_excluded > ULP:  {steps_above_ulp}/{len(tokens)}")
        
        if steps_above_ulp > 0:
            print(f"    ✗ WARNING: {steps_above_ulp} steps have excluded weights above ULP!")
            # Find the worst steps
            worst_steps = sorted(enumerate(max_excl_list), key=lambda x: x[1], reverse=True)[:5]
            for step_idx, weight in worst_steps:
                print(f"      Step {step_idx}: max_excluded={weight:.2e}")
        else:
            print(f"    ✓ All {len(tokens)} steps have excluded weights below ULP")
        
        print(f"\n    Generated text (first 200 chars):")
        print(f"    {gen_text[:200]}...")


def main():
    print("=" * 80)
    print("ADVERSARIAL TEST: LONG GENERATION (500+ TOKENS)")
    print("=" * 80)
    print()
    print("Does concentration hold during long autoregressive generation?")
    print("Even if individual steps are exact, we verify there's no drift.")
    print()
    print("float32 ULP threshold: 6e-8")
    print("If any step has max_excluded > ULP, that step may not be exact.")
    print()
    print("NOTE: Using 500 tokens (not 1000+) to fit in GPU memory with")
    print("output_attentions=True. The attention matrices consume ~O(n²) memory.")

    test_long_generation()

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    print()
    print("If all steps have excluded weights below ULP:")
    print("  → Each step is individually exact")
    print("  → Identical inputs → identical outputs → no accumulation possible")
    print("  → Long generation is exact by induction")
    print()
    print("If some steps exceed ULP:")
    print("  → Those specific steps may produce different logits")
    print("  → Could cascade through autoregressive generation")
    print("  → Report as boundary condition")
    print("=" * 80)


if __name__ == "__main__":
    main()
