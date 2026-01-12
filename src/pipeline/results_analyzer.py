"""
Results analysis and reporting utilities.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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
        pattern: str = "*_results.json",
        interactive: bool = True
    ):
        """
        Generate an HTML comparison report with interactive visualizations.

        Args:
            output_file: Path to output HTML file
            pattern: Glob pattern for result files
            interactive: Whether to include interactive Plotly charts
        """
        results = self.load_results(pattern)
        if not results:
            print("No results found to analyze")
            return

        df = self.create_comparison_dataframe(results)

        # Generate interactive charts if Plotly is available
        charts_html = ""
        if interactive and PLOTLY_AVAILABLE:
            charts_html = self._generate_interactive_charts(df)

        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Comparison Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e8f4f8; }}
                .metric-card {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            </style>
            {'<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' if interactive and PLOTLY_AVAILABLE else ''}
        </head>
        <body>
            <div class="container">
                <h1>RAG Evaluation Comparison Report</h1>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Number of Experiments:</strong> {len(results)}</p>
                
                <h2>Summary Statistics</h2>
                <div class="summary-stats">
                    {self._generate_summary_cards(df)}
                </div>
                {df.describe().to_html(classes='table', escape=False)}
                
                {charts_html}
                
                <h2>Detailed Results</h2>
                {df.to_html(index=False, classes='table', escape=False)}
                
                <h2>Configuration Details</h2>
                {self._generate_config_summary(results)}
            </div>
        </body>
        </html>
        """

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html)

        print(f"Report saved to: {output_file}")

    def _generate_interactive_charts(self, df: pd.DataFrame) -> str:
        """Generate interactive Plotly charts for the dashboard."""
        if not PLOTLY_AVAILABLE:
            return ""
        
        charts = []
        
        # Get metric columns (exclude 'experiment')
        metric_cols = [col for col in df.columns if col != 'experiment']
        
        # Group metrics by category
        retrieval_metrics = [col for col in metric_cols if any(x in col.lower() for x in ['precision', 'recall', 'mrr'])]
        answer_metrics = [col for col in metric_cols if 'answer' in col.lower()]
        faithfulness_metrics = [col for col in metric_cols if 'faithfulness' in col.lower()]
        performance_metrics = [col for col in metric_cols if any(x in col.lower() for x in ['latency', 'throughput'])]
        
        # Retrieval metrics comparison
        if retrieval_metrics:
            fig = make_subplots(rows=1, cols=len(retrieval_metrics), subplot_titles=retrieval_metrics)
            for i, metric in enumerate(retrieval_metrics[:3]):  # Limit to 3 for layout
                fig.add_trace(
                    go.Bar(x=df['experiment'], y=df[metric], name=metric, showlegend=False),
                    row=1, col=i+1
                )
            fig.update_layout(height=400, title_text="Retrieval Metrics Comparison", title_x=0.5)
            charts.append(fig.to_html(include_plotlyjs='cdn', div_id="retrieval_chart"))
        
        # Answer correctness metrics
        if answer_metrics:
            fig = go.Figure()
            for metric in answer_metrics[:5]:  # Limit to 5 metrics
                fig.add_trace(go.Bar(x=df['experiment'], y=df[metric], name=metric))
            fig.update_layout(
                title="Answer Correctness Metrics",
                xaxis_title="Experiment",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            charts.append(fig.to_html(include_plotlyjs='cdn', div_id="answer_chart"))
        
        # Performance metrics
        if performance_metrics:
            fig = go.Figure()
            for metric in performance_metrics:
                fig.add_trace(go.Bar(x=df['experiment'], y=df[metric], name=metric))
            fig.update_layout(
                title="Performance Metrics (Latency & Throughput)",
                xaxis_title="Experiment",
                yaxis_title="Value",
                barmode='group',
                height=400
            )
            charts.append(fig.to_html(include_plotlyjs='cdn', div_id="performance_chart"))
        
        # Overall comparison heatmap
        if len(metric_cols) > 0:
            fig = go.Figure(data=go.Heatmap(
                z=df[metric_cols[:10]].T.values,  # Limit to 10 metrics
                x=df['experiment'].values,
                y=metric_cols[:10],
                colorscale='Viridis',
                colorbar=dict(title="Score")
            ))
            fig.update_layout(
                title="Metrics Heatmap",
                xaxis_title="Experiment",
                yaxis_title="Metric",
                height=500
            )
            charts.append(fig.to_html(include_plotlyjs='cdn', div_id="heatmap_chart"))
        
        return "<h2>Interactive Visualizations</h2>" + "".join(charts)

    def _generate_summary_cards(self, df: pd.DataFrame) -> str:
        """Generate summary statistic cards."""
        cards = []
        metric_cols = [col for col in df.columns if col != 'experiment']
        
        # Key metrics to highlight
        key_metrics = {
            'precision@10': 'Precision@10',
            'recall@10': 'Recall@10',
            'mrr': 'MRR',
            'answer_rouge_l': 'Answer ROUGE-L',
            'mean_latency': 'Mean Latency (s)',
            'throughput': 'Throughput (qps)'
        }
        
        for metric_key, metric_label in key_metrics.items():
            matching_cols = [col for col in metric_cols if metric_key in col.lower()]
            if matching_cols:
                col = matching_cols[0]
                avg_val = df[col].mean()
                max_val = df[col].max()
                best_exp = df.loc[df[col].idxmax(), 'experiment']
                cards.append(f"""
                    <div class="metric-card">
                        <h3>{metric_label}</h3>
                        <p><strong>Average:</strong> {avg_val:.4f}</p>
                        <p><strong>Best:</strong> {max_val:.4f} ({best_exp})</p>
                    </div>
                """)
        
        return "".join(cards) if cards else ""

    def _generate_config_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate configuration summary table."""
        config_rows = []
        for result in results:
            config = result.get('config', {})
            config_rows.append({
                'Experiment': result.get('experiment_name', 'unknown'),
                'Chunking': config.get('chunking', {}).get('strategy', 'N/A'),
                'Embedding': f"{config.get('embedding', {}).get('model_family', 'N/A')}/{config.get('embedding', {}).get('model_name', 'N/A')}",
                'Vector DB': config.get('vectordb', {}).get('backend', 'N/A'),
                'Dataset': config.get('dataset', {}).get('name', 'N/A')
            })
        
        config_df = pd.DataFrame(config_rows)
        return config_df.to_html(index=False, classes='table', escape=False)

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

    def generate_comprehensive_dashboard(
        self,
        output_file: str = "results/dashboard.html",
        pattern: str = "*_results.json"
    ):
        """
        Generate a comprehensive interactive dashboard with all visualizations.

        Args:
            output_file: Path to output HTML file
            pattern: Glob pattern for result files
        """
        self.generate_comparison_report(output_file, pattern, interactive=True)
        print(f"Comprehensive dashboard generated: {output_file}")
