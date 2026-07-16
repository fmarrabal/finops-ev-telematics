#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finops_repro.pipeline import reproduce


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the R2 FinOps EV-telematics analysis, tables, tests, and figures."
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated artifacts.")
    parser.add_argument("--skip-figures", action="store_true", help="Generate numerical outputs only.")
    args = parser.parse_args()
    summary = reproduce(args.output_dir, make_figures=not args.skip_figures)
    print(f"Outputs written to: {Path(args.output_dir).resolve()}")
    print(f"All reference checks passed: {summary['all_reference_checks_passed']}")
    return 0 if summary["all_reference_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
