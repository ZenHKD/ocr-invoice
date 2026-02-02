#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED SYNTHETIC INVOICE GENERATOR

Uses the modular synthetic_data package for generating diverse, realistic invoice data.

Features:
    - Multiple layout types (thermal, VAT, handwritten, cafe)
    - Realistic purchase behavior patterns
    - Visual defects (folds, stains, noise, etc.)
    - Edge cases (partial scans, rotations, multi-receipt)
    - Configurable scenarios for training/validation

Usage:
    # Quick generation with defaults
    python run_synthetic.py

    # Custom generation
    python run_synthetic.py --num 500 --scenario training_hard --output data/train

    # Validation set (cleaner images)
    python run_synthetic.py --num 100 --scenario validation --output data/val
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_data.generator import (
    GenerationScenario,
    generate_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Vietnamese invoice dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 100 balanced training samples
    python run_synthetic.py --num 100 --scenario training_balanced

    # Generate difficult training data
    python run_synthetic.py --num 500 --scenario training_hard --output data/hard

    # Generate clean validation data
    python run_synthetic.py --num 50 --scenario validation --output data/val

    # Focus on edge cases for robustness testing
    python run_synthetic.py --num 200 --scenario edge_cases_focus

Available scenarios:
    training_balanced  - Balanced mix (75% normal, 15% edge cases, 10% bad)
    training_hard      - More challenging (50% normal, 35% edge cases, 15% bad)
    validation         - Clean data (95% normal, 5% edge cases)
    edge_cases_focus   - Heavy edge cases (30% normal, 60% edge cases)
    retail_focus       - Focus on supermarket/convenience store receipts
    restaurant_focus   - Focus on restaurant/cafe bills
    formal_invoices    - Focus on official VAT invoices
        """
    )

    parser.add_argument(
        "-n", "--num",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/train",
        help="Output directory (default: data/train)"
    )

    parser.add_argument(
        "-s", "--scenario",
        type=str,
        default="training_balanced",
        choices=[s.value for s in GenerationScenario],
        help="Generation scenario (default: training_balanced)"
    )

    parser.add_argument(
        "--defect-preset",
        type=str,
        default=None,
        choices=["pristine", "good_scan", "used_receipt", "damaged", "extreme"],
        help="Defect preset to override scenario default"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Print configuration
    if not args.quiet:
        print("=" * 50)
        print("SYNTHETIC INVOICE GENERATOR")
        print("=" * 50)
        print(f"  Samples:    {args.num}")
        print(f"  Output:     {args.output}")
        print(f"  Scenario:   {args.scenario}")
        if args.defect_preset:
            print(f"  Defects:    {args.defect_preset}")
        print("=" * 50)
        print()

    # Generate dataset
    results = generate_dataset(
        output_dir=args.output,
        num_samples=args.num,
        scenario=args.scenario,
        verbose=not args.quiet
    )

    # Final summary
    if not args.quiet:
        print()
        print("=" * 50)
        print(f"SUCCESS: Generated {len(results)} invoices in {args.output}/")
        print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
