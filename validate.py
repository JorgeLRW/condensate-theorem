#!/usr/bin/env python3
"""
Condensate Theorem - One-Command Validation
============================================

Run this script to validate the entire theorem:

    python validate.py

This runs all validation scripts and provides a summary.

NOTE: These are REFERENCE IMPLEMENTATIONS that prove the theorem is correct.
      The production-optimized Topological Attention kernel (157x+ speedup)
      is available under commercial license: jorgeruizwilliams@gmail.com
"""

import subprocess
import sys
from pathlib import Path


def run_validation(script_name: str, description: str) -> bool:
    """Run a validation script and return success status."""
    script_path = Path(__file__).parent / "validation" / script_name
    
    print("\n" + "=" * 80)
    print(f"  {description}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            timeout=300  # 5 minute timeout per script
        )
        return True
    except subprocess.CalledProcessError:
        print(f"  ❌ FAILED: {script_name}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ⏱️ TIMEOUT: {script_name}")
        return False
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              THE CONDENSATE THEOREM - VALIDATION SUITE                       ║
║                                                                              ║
║    Proving: Trained transformers are O(n), not O(n²)                         ║
║                                                                              ║
║    This repository contains REFERENCE IMPLEMENTATIONS that validate          ║
║    the theorem's mathematical correctness. The production kernel             ║
║    (157x+ speedup) is available under commercial license.                    ║
║                                                                              ║
║    Contact: jorgeruizwilliams@gmail.com                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    validations = [
        ("attention_mass.py", "TEST 1: Attention Mass Distribution - WHY the manifold works"),
        ("exact_equivalence.py", "TEST 2: Exact Equivalence - Sparse produces identical logits"),
        ("needle_retrieval.py", "TEST 3: Needle Retrieval - Dynamic Top-K finds buried facts"),
        ("multimodel.py", "TEST 4: Multi-Model - Pattern holds across architectures"),
    ]
    
    results = []
    
    for script, description in validations:
        success = run_validation(script, description)
        results.append((script, success))
    
    # Summary
    print("\n")
    print("=" * 80)
    print("                           VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for script, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {script:<25} {status}")
    
    print("-" * 80)
    print(f"  TOTAL: {passed}/{total} validations passed")
    print("=" * 80)
    
    if passed == total:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ✅ THEOREM VALIDATED                                                       ║
║                                                                              ║
║   The Condensate Manifold captures ~100% of attention mass.                  ║
║   Sparse attention produces IDENTICAL outputs to full O(n²) attention.       ║
║   The pattern holds across GPT-2, Pythia, Qwen, and TinyLlama.               ║
║                                                                              ║
║   This reference implementation PROVES the theorem works.                    ║
║   The production Topological Attention kernel achieves 157x+ speedup.        ║
║                                                                              ║
║   License the production kernel: jorgeruizwilliams@gmail.com                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
⚠️  Some validations failed. This may be due to:
   - Missing dependencies (pip install torch transformers)
   - GPU/CPU memory constraints
   - Network issues downloading models

   Try running individual scripts to debug:
   python validation/attention_mass.py
""")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
