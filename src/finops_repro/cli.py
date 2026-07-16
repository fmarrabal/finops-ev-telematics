from __future__ import annotations

import argparse
from pathlib import Path
from .pipeline import reproduce


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the FinOps EV-telematics R2 analysis.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    summary = reproduce(args.output_dir, make_figures=not args.skip_figures)
    print(f"Outputs written to {Path(args.output_dir).resolve()}")
    print(f"All reference checks passed: {summary['all_reference_checks_passed']}")
    return 0 if summary["all_reference_checks_passed"] else 1
