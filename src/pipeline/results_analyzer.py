"""
Results analysis and reporting utilities.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsAnalyzer:
    """Analyzer for experiment results."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize results analyzer.

        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = Path(results_dir)

    def load_results(self, pattern: str = "*_results.json") -> List[Dict[str, Any]]:
        """
        Load all result files matching pattern.

        Args:
            pattern: Glob pattern for result files

        Returns:
            List of result dictionaries
        """
        result_files = list(self.results_dir.glob(pattern))
        results = []

        for file in result_files:
            try:
                with open(file, "r") as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        return results

    def create_comparison_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a comparison DataFrame from multiple results.

        Args:
            results: List of result dictionaries

        Returns:
            DataFrame with experiment metrics
        """
        rows = []
        for result in results:
            row = {
                "experiment": result.get("experiment_name", "unknown"),
                **result.get("metrics", {})
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_comparison_report(
        self,
        output_file: str = "results/comparison_report.html",
        pattern: str = "*_results.json"
    ):
        """
        Generate an HTML comparison report.

        Args:
            output_file: Path to output HTML file
            pattern: Glob pattern for result files
        """
        results = self.load_results(pattern)
        if not results:
            print("No results found to analyze")
            return

        df = self.create_comparison_dataframe(results)

        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                h1 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>RAG Evaluation Comparison Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <h2>Summary Statistics</h2>
            {df.describe().to_html()}
            <h2>Detailed Results</h2>
            {df.to_html(index=False)}
        </body>
        </html>
        """

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        print(f"Report saved to: {output_file}")

    def plot_metrics_comparison(
        self,
        metrics: List[str],
        output_file: str = "results/metrics_comparison.png",
        pattern: str = "*_results.json"
    ):
        """
        Generate comparison plots for specified metrics.

        Args:
            metrics: List of metric names to plot
            output_file: Path to output image file
            pattern: Glob pattern for result files
        """
        results = self.load_results(pattern)
        if not results:
            print("No results found to plot")
            return

        df = self.create_comparison_dataframe(results)

        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            print(f"None of the specified metrics found: {metrics}")
            return

        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, available_metrics):
            df.plot(x="experiment", y=metric, kind="bar", ax=ax, legend=False)
            ax.set_title(f"{metric} Comparison")
            ax.set_xlabel("Experiment")
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")

    def export_to_csv(
        self,
        output_file: str = "results/comparison.csv",
        pattern: str = "*_results.json"
    ):
        """
        Export results to CSV.

        Args:
            output_file: Path to output CSV file
            pattern: Glob pattern for result files
        """
        results = self.load_results(pattern)
        if not results:
            print("No results found to export")
            return

        df = self.create_comparison_dataframe(results)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"CSV exported to: {output_file}")

