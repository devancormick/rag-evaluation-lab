"""
Command-line interface for running experiments.
"""

import argparse
import sys
from pathlib import Path

from .experiment_runner import run_experiment


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file (YAML or JSON)"
    )

    args = parser.parse_args()

    try:
        results = run_experiment(args.config)
        print("\nExperiment completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

