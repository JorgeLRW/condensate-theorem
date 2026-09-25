"""
Adversarial Test: Inputs Designed to Break Sparsity
====================================================

The Condensate Theorem relies on trained models concentrating attention.
Can we craft inputs that SPREAD attention uniformly, breaking the pattern?

Attack vectors:
1. Random token sequences (no semantic structure)
2. Repeated tokens (uniform content)
3. All-same-token input (degenerate case)
4. Adversarial: content where many positions are equally important
5. Very short sequences (where condensate = full sequence)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def measure_attention_concentration(model, tokenizer, text, window_size=64, top_k=32):
    """
    Measure how concentrated attention is on the condensate set.
    Returns: condensate_mass, max_excluded_weight, seq_len, per-layer details
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len <= 2:
        return 1.0, 0.0, seq_len, []  # trivially exact

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    layer_details = []
    global_max_excluded = 0.0
    global_min_condensate = 1.0
    
    for layer_idx in range(len(outputs.attentions)):
        attn = outputs.attentions[layer_idx][0]  # [H, S, S]
        layer_max_excl = 0.0
        layer_min_cond = 1.0
        
        for head_idx in range(attn.shape[0]):
            weights = attn[head_idx, -1, :seq_len].float()
            
            # Build condensate set
            condensate = set()
            condensate.add(0)  # anchor
            for j in range(max(0, seq_len - window_size), seq_len):
                condensate.add(j)  # window
            
            # Top-k from middle
            middle_start = 1
            middle_end = max(1, seq_len - window_size)
            if middle_end > middle_start:
                middle = weights[middle_start:middle_end]
                k = min(top_k, len(middle))
                if k > 0:
                    topk_idx = middle.topk(k).indices
                    for idx in topk_idx:
                        condensate.add(middle_start + idx.item())
            
            condensate_mass = weights[sorted(condensate)].sum().item()
            
            excluded = set(range(seq_len)) - condensate
            if excluded:
                excl_weights = weights[sorted(excluded)]
                max_excl = excl_weights.max().item()
            else:
                max_excl = 0.0
            
            layer_max_excl = max(layer_max_excl, max_excl)
            layer_min_cond = min(layer_min_cond, condensate_mass)
        
        global_max_excluded = max(global_max_excluded, layer_max_excl)
        global_min_condensate = min(global_min_condensate, layer_min_cond)
        layer_details.append({
            'layer': layer_idx,
            'max_excluded': layer_max_excl,
            'min_condensate': layer_min_cond,
        })
    
    return global_min_condensate, global_max_excluded, seq_len, layer_details


def test_adversarial_inputs():
    """Run all adversarial input types."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2-medium', attn_implementation='eager'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    
    ulp_f32 = 6e-8
    
    # --- Adversarial inputs ---
    
    test_cases = []
    
    # 1. Random tokens
    random_ids = torch.randint(0, tokenizer.vocab_size, (200,))
    random_text = tokenizer.decode(random_ids)
    test_cases.append(("Random tokens (200)", random_text))
    
    # 2. Repeated word
    test_cases.append(("Repeated 'the' (200x)", "the " * 200))
    
    # 3. Repeated sentence
    test_cases.append(("Repeated sentence (20x)", "The cat sat on the mat. " * 20))
    
    # 4. All same token
    test_cases.append(("All periods (200)", ". " * 200))
    
    # 5. Numbers sequence (uniform importance?)
    test_cases.append(("Number sequence", " ".join(str(i) for i in range(200))))
    
    # 6. Multi-fact (many positions should matter)
    facts = []
    for i in range(20):
        facts.append(f"Fact {i}: the value is {i * 7 + 3}.")
    facts.append("What is the value in Fact 15?")
    test_cases.append(("Multi-fact retrieval (20 facts)", " ".join(facts)))
    
    # 7. Alternating languages / scripts
    test_cases.append(("Mixed content", 
        "Hello world. 你好世界. Bonjour le monde. こんにちは世界. " * 5 +
        "What was the first greeting?"))
    
    # 8. Code with many variable references
    code = "x=1\ny=2\nz=3\na=x+y\nb=y+z\nc=a+b\n" * 10 + "result = c + "
    test_cases.append(("Code many vars", code))
    
    # 9. Very long repeated question (should attend broadly)
    test_cases.append(("Same question 20x",
        "What is the meaning of life? " * 20 + "The answer is"))
    
    # 10. Adversarial: list every letter
    test_cases.append(("Alphabet repeated",
        " ".join(list("abcdefghijklmnopqrstuvwxyz") * 10) + " The first letter was"))
    
    print(f"\n  {'Test Case':<35} {'SeqLen':>6} {'Min Cond%':>10} {'Max Excl':>12} {'< ULP':>6} {'Status':>12}")
    print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*12} {'-'*6} {'-'*12}")
    
    passes = 0
    fails = 0
    
    for name, text in test_cases:
        min_cond, max_excl, seq_len, details = measure_attention_concentration(
            model, tokenizer, text
        )
        
        below_ulp = max_excl < ulp_f32
        status = "✓ EXACT" if below_ulp else "✗ EXCEEDS"
        
        if below_ulp:
            passes += 1
        else:
            fails += 1
        
        print(f"  {name:<35} {seq_len:>6} {min_cond*100:>9.2f}% {max_excl:>12.2e} {'YES' if below_ulp else 'NO':>6} {status:>12}")
        
        # If it fails, show per-layer breakdown
        if not below_ulp:
            worst_layers = sorted(details, key=lambda d: d['max_excluded'], reverse=True)[:3]
            for d in worst_layers:
                print(f"    → Layer {d['layer']}: max_excluded={d['max_excluded']:.2e}, min_condensate={d['min_condensate']*100:.1f}%")
    
    return passes, fails


def test_short_sequences():
    """
    For very short sequences (< 97 positions), condensate = full sequence.
    The theorem is trivially true. Verify this edge case.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2-medium', attn_implementation='eager'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    
    print(f"\n  Short sequence edge cases (condensate ≥ full sequence):")
    print(f"  {'SeqLen':>6} {'Condensate Size':>16} {'Covers All?':>12}")
    print(f"  {'-'*6} {'-'*16} {'-'*12}")
    
    for text in ["Hi", "Hello world", "The cat sat on the mat", 
                 "Tell me a story about " + "a " * 30 + "dragon"]:
        input_ids = tokenizer(text, return_tensors='pt')['input_ids']
        seq_len = input_ids.shape[1]
        condensate_size = min(1 + 64 + 32, seq_len)  # anchor + window + topk
        covers_all = condensate_size >= seq_len
        print(f"  {seq_len:>6} {condensate_size:>16} {'YES (trivial)' if covers_all else 'NO':>12}")


def main():
    print("=" * 80)
    print("ADVERSARIAL TEST: INPUTS DESIGNED TO BREAK SPARSITY")
    print("=" * 80)
    print()
    print("Can we craft inputs that spread attention uniformly?")
    print("If so, excluded positions may exceed the ULP threshold.")
    print()
    print("float32 ULP threshold: 6e-8")
    print("If max excluded weight > 6e-8 → bit-exactness breaks for that input.")

    passes, fails = test_adversarial_inputs()
    test_short_sequences()

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  Passed: {passes}")
    print(f"  Failed: {fails}")
    
    if fails == 0:
        print(f"\n  ✓ All adversarial inputs maintain concentration below ULP.")
        print(f"    The Condensate Theorem holds even on pathological inputs.")
    else:
        print(f"\n  ✗ {fails} input(s) have excluded weights above ULP.")
        print(f"    These inputs may break bit-exact equivalence.")
        print(f"    Consider: (a) increasing top-k, (b) scoping the claim,")
        print(f"    or (c) reporting these as boundary conditions.")
    
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
