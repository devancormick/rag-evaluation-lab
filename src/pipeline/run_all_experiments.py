"""
Script to run all experiments in a directory.
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from .experiment_runner import run_experiment
from .results_analyzer import ResultsAnalyzer


def main():
    """Main entry point for running all experiments."""
    parser = argparse.ArgumentParser(description="Run all RAG evaluation experiments")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="experiments",
        help="Directory containing experiment configuration files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="Glob pattern for configuration files"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comparison report after running experiments"
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"Configuration directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all config files
    config_files = list(config_dir.glob(args.pattern))
    if not config_files:
        print(f"No configuration files found matching pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(config_files)} experiment configurations")
    print(f"Running experiments...\n")

    # Run experiments
    successful = []
    failed = []

    for config_file in tqdm(config_files, desc="Experiments"):
        try:
            print(f"\nRunning: {config_file.name}")
            run_experiment(str(config_file))
            successful.append(config_file)
        except Exception as e:
            print(f"Error running {config_file.name}: {e}", file=sys.stderr)
            failed.append(config_file)

    # Summary
    print("\n" + "=" * 50)
    print("Experiment Summary")
    print("=" * 50)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed experiments:")
        for f in failed:
            print(f"  - {f.name}")

    # Generate report if requested
    if args.generate_report and successful:
        print("\nGenerating comparison report...")
        analyzer = ResultsAnalyzer()
        analyzer.generate_comparison_report()
        analyzer.export_to_csv()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

