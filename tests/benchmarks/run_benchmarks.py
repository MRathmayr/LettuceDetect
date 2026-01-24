#!/usr/bin/env python3
"""Run LettuceDetect benchmarks.

Usage:
    # Quick check (100 samples, no CI)
    python tests/benchmarks/run_benchmarks.py --quick

    # Full benchmark (all samples)
    python tests/benchmarks/run_benchmarks.py --full

    # Component benchmarks only
    python tests/benchmarks/run_benchmarks.py --components

    # Specific dataset
    python tests/benchmarks/run_benchmarks.py --dataset halueval_qa --limit 500
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run LettuceDetect benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 100 samples, no CI computation",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full mode: all samples, full metrics",
    )
    parser.add_argument(
        "--components",
        action="store_true",
        help="Run component benchmarks only",
    )
    parser.add_argument(
        "--stages",
        action="store_true",
        help="Run stage benchmarks only",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Run cascade benchmarks only",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ragtruth",
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per dataset",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Determine test path
    if args.components:
        cmd.append("tests/benchmarks/components/")
    elif args.stages:
        cmd.append("tests/benchmarks/stages/")
    elif args.cascade:
        cmd.append("tests/benchmarks/cascade/")
    else:
        cmd.append("tests/benchmarks/")

    # Add options
    cmd.append("--ignore-glob=")  # Don't use default ignore

    if args.quick:
        cmd.extend(["--quick"])
    if args.full:
        cmd.extend(["--full"])
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    cmd.extend(["-v", "--tb=short", "-s"])  # Verbose, short traceback, no capture

    if args.verbose:
        print(f"Running: {' '.join(cmd)}")

    # Run pytest
    result = subprocess.run(cmd, cwd=".")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
