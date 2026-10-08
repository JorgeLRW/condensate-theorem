#!/usr/bin/env python3
"""Repository checks.

    python validate.py              # CPU unit tests for the selector and the omitted-mass identity
    python validate.py --smoke      # also rerun 2K / prompt 0 / S=97,769 on a CUDA GPU and compare to the archive

The smoke rerun is a single paired case from the 60-run sweep, not the full sweep.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "results" / "qwen_survival_clean.json"
COMPARED_FIELDS = (
    "first_divergence",
    "teacher_forced_match",
    "delta_ppl_percent",
    "mean_unique_distant_positions_per_kv",
    "max_unique_distant_positions_per_kv",
)
TOLERANCE = 1e-3


def run_unit_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    return unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()


def run_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "smoke.json"
        subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "run_survival.py"),
                "--contexts", "2048", "--prompts", "1", "--supports", "97,769",
                "--reuse", "1", "--max-new-tokens", "128", "--output", str(output),
            ],
            check=True,
        )
        rerun = json.loads(output.read_text(encoding="utf-8"))["survival"]

    archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))["survival"]
    all_match = True
    for row in rerun:
        reference = next(
            r for r in archive
            if r["context"] == row["context"]
            and r["prompt_index"] == row["prompt_index"]
            and r["support"] == row["support"]
            and r["reuse"] == row["reuse"]
        )
        print(f"S={row['support']} R={row['reuse']} N={row['context']} prompt={row['prompt_index']}")
        for field in COMPARED_FIELDS:
            diff = abs(float(row[field]) - float(reference[field]))
            ok = diff <= TOLERANCE
            all_match &= ok
            print(f"  {'MATCH' if ok else 'DIFF '} {field}: rerun={row[field]} archive={reference[field]}")
    return all_match


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="also run the GPU smoke rerun")
    args = parser.parse_args()

    ok = run_unit_tests()
    if args.smoke:
        ok = run_smoke() and ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
