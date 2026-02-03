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
    python run_synthetic.py --num 100 --scenario training_balanced --output data/train

    # Validation set (cleaner images)
    python run_synthetic.py --num 100 --scenario validation --output data/val
"""

import argparse
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_data.generator import (
    GenerationScenario,
    SyntheticInvoiceGenerator,
)

# Global generator instance for worker processes
_worker_generator = None

def worker_init(scenario_enum_value, output_dir, num_samples):
    """Initialize the generator in each worker process."""
    global _worker_generator
    try:
        scenario = GenerationScenario(scenario_enum_value)
        _worker_generator = SyntheticInvoiceGenerator.from_scenario(
            scenario, output_dir, num_samples
        )
        import random
        import numpy as np
        seed = (os.getpid() * int(time.time() * 1000)) % 123456789
        random.seed(seed)
        np.random.seed(seed)
    except Exception as e:
        print(f"Worker initialization failed: {e}")

def generate_sample_wrapper(i):
    """Worker function to generate a single sample."""
    global _worker_generator
    if _worker_generator is None:
        return {'error': 'Generator not initialized', 'id': i}
    
    try:
        import random
        dice = random.random()
        config = _worker_generator.config
        
        if dice < config.blank_ratio:
            result = _worker_generator._generate_blank(i)
        elif dice < config.blank_ratio + config.unreadable_ratio:
            result = _worker_generator._generate_unreadable(i)
        elif dice < (config.blank_ratio + config.unreadable_ratio + config.edge_case_ratio):
            result = _worker_generator._generate_with_edge_case(i)
        else:
            result = _worker_generator._generate_realistic(i)
            
        return result
    except Exception as e:
        return {'error': str(e), 'id': i}

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Vietnamese invoice dataset (Parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 100 balanced training samples (auto-parallel)
    python run_synthetic.py --num 100 --scenario training_balanced

    # Generate with specific number of workers
    python run_synthetic.py --num 1000 --workers 4

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
    pure_random_focus  - 100% random text (for debugging)
    pseudo_focus       - 100% pseudo-generated text
        """
    )

    parser.add_argument("-n", "--num", type=int, default=100, help="Number of samples")
    parser.add_argument("-o", "--output", type=str, default="data/train", help="Output directory")
    parser.add_argument("-s", "--scenario", type=str, default="training_balanced",
                        choices=[s.value for s in GenerationScenario], help="Generation scenario")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), 
                        help=f"Number of worker processes (default: {os.cpu_count()})")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (master)")

    args = parser.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using master random seed: {args.seed}")

    if not args.quiet:
        print("=" * 50)
        print("PARALLEL SYNTHETIC INVOICE GENERATOR")
        print("=" * 50)
        print(f"  Samples:    {args.num}")
        print(f"  Output:     {args.output}")
        print(f"  Scenario:   {args.scenario}")
        print(f"  Workers:    {args.workers}")
        print("=" * 50)
        print()

    # Create output directory beforehand to avoid race conditions in workers
    Path(args.output).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor
    with ProcessPoolExecutor(
        max_workers=args.workers, 
        initializer=worker_init,
        initargs=(args.scenario, args.output, args.num)
    ) as executor:
        
        # Submit all tasks
        futures = {executor.submit(generate_sample_wrapper, i): i for i in range(args.num)}
        
        # Process results as they complete
        completed_count = 0
        for future in as_completed(futures):
            res = future.result()
            if 'error' in res:
                if not args.quiet:
                    print(f"Error in sample {res['id']}: {res['error']}")
            else:
                results.append(res)
                completed_count += 1
                if not args.quiet:
                     sample_type = res.get("sample_type", "unknown")
                     print(f"[{completed_count}/{args.num}] Generated {sample_type}: {res.get('id', 'N/A')}")

    elapsed = time.time() - start_time

    if not args.quiet:
        print()
        print("=" * 50)
        print(f"SUCCESS: Generated {len(results)} invoices in {elapsed:.2f}s")
        print("=" * 50)

        # Print summary
        type_counts = {}
        layout_counts = {}
        for r in results:
            t = r.get("sample_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

            layout = r.get("layout_type", "unknown")
            if layout != "unknown":
                layout_counts[layout] = layout_counts.get(layout, 0) + 1

        print(f"\nSample Types:")
        for t, count in sorted(type_counts.items()):
            print(f"  {t}: {count} ({100 * count / len(results):.1f}%)")

        if layout_counts:
            print(f"\nLayout Types:")
            for layout, count in sorted(layout_counts.items()):
                print(f"  {layout}: {count} ({100 * count / len(results):.1f}%)")

    return 0

if __name__ == "__main__":
    if sys.platform == "win32":
        # Windows specifics for multiprocessing
        multiprocessing.freeze_support()
    sys.exit(main())
