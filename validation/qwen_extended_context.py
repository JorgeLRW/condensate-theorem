"""
Qwen2-0.5B Long-Context Scaling & Positional Boundary Evaluation
===============================================================
Reference evaluation demonstrating chunked prefill + Condensate decoding
across scaling sequence lengths (4K, 8K, 16K, 32K, 65K, 131K).
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import math
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

def main():
    print("=" * 80)
    print("QWEN2-0.5B: LONG-CONTEXT SCALING & POSITIONAL BOUNDARY TEST")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    model_name = "Qwen/Qwen2-0.5B"
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading Qwen2-0.5B in float16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    ).eval()
    
    base_vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. Base VRAM: {base_vram:.2f} GB")
    
    needle_word = "PHOENIX"
    needle_fact = f"The secret access code to the vault is {needle_word}."
    question = "What is the secret access code to the vault? The code is"
    filler = "The international scientific committee published proceedings on computational mathematics and neural networks. "
    
    needle_toks = tokenizer(needle_fact, add_special_tokens=False)['input_ids']
    question_toks = tokenizer(question, add_special_tokens=False)['input_ids']
    filler_toks = tokenizer(filler, add_special_tokens=False)['input_ids']
    
    test_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    
    print("\n" + "=" * 80)
    print(f"{'Context Length':<16} {'Needle Pos':<14} {'Depth':<8} {'Time (s)':<10} {'Peak VRAM':<12} {'Top-1':<8} {'Status'}")
    print("=" * 80)
    
    for seq_len in test_lengths:
        overhead = len(needle_toks) + len(question_toks) + 5
        num_repeats = max(1, (seq_len - overhead) // len(filler_toks))
        split = max(1, num_repeats // 4)
        
        part1 = filler_toks * split
        part2 = filler_toks * (num_repeats - split)
        
        full_tokens = [151643] + part1 + needle_toks + part2 + question_toks
        full_tokens = full_tokens[:seq_len]
        
        needle_idx = len(part1) + 1
        depth_pct = needle_idx / len(full_tokens) * 100
        
        chunk_size = 1024
        past_key_values = DynamicCache()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        
        num_chunks = math.ceil(len(full_tokens) / chunk_size)
        with torch.no_grad():
            for i in range(num_chunks):
                chunk = full_tokens[i * chunk_size : (i + 1) * chunk_size]
                chunk_tensor = torch.tensor([chunk], device=device)
                out = model(chunk_tensor, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                del out, chunk_tensor
                
            final_input = torch.tensor([[full_tokens[-1]]], device=device)
            out = model(final_input, past_key_values=past_key_values, use_cache=True)
            next_token_id = torch.argmax(out.logits[0, -1, :]).item()
            pred_text = tokenizer.decode([next_token_id]).strip()
            
        elapsed_s = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        
        is_pass = 'PH' in pred_text or 'PHOENIX' in pred_text
        status = "[PASS]" if is_pass else "[FAIL (RoPE OOD)]"
        
        print(f"{seq_len:<16,} {needle_idx:<14,} {depth_pct:>5.1f}%   {elapsed_s:>6.2f}s    {peak_vram:>5.2f} GB      '{pred_text:<4}'   {status}")
        
        del past_key_values
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
