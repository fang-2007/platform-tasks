"""Command-line interface for risk-eval."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from risk_eval import __version__
from risk_eval.core.data_loader import DataLoaderFactory
from risk_eval.core.evaluator import Evaluator
from risk_eval.core.quality_checker import QualityChecker
from risk_eval.core.rule_engine import RuleEngine
from risk_eval.report.chart_generator import ChartGenerator
from risk_eval.report.html_report import HTMLReportGenerator
from risk_eval.utils.config import AppConfig
from risk_eval.utils.logger import configure_logging, get_logger

app = typer.Typer(
    name="risk-eval",
    help="Risk Control Data Quality & Strategy Evaluation Toolkit",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


def _load_data(source: str) -> pd.DataFrame:
    """Load data from source with error handling."""
    console.print(f"[blue]Loading data from {source}...[/blue]")
    df = DataLoaderFactory.load(source)
    console.print(f"[green]Loaded {len(df):,} rows × {len(df.columns)} columns[/green]")
    return df


def _load_rules(path: str) -> RuleEngine:
    """Load rule engine from YAML."""
    console.print(f"[blue]Loading rules from {path}...[/blue]")
    engine = RuleEngine.from_yaml(path)
    console.print(f"[green]Loaded strategy: {engine.config.name}[/green]")
    return engine


@app.command()
def quality_check(
    source: Annotated[str, typer.Argument(help="Data source (file path or connection string)")],
    report: Annotated[Optional[str], typer.Option("--report", "-r", help="Output HTML report path")] = None,
    label: Annotated[Optional[str], typer.Option("--label", "-l", help="Label column name")] = None,
    null_threshold: Annotated[float, typer.Option("--null-threshold", help="Null rate warning threshold")] = 0.5,
    json_logs: Annotated[bool, typer.Option("--json-logs", help="Output JSON logs")] = False,
) -> None:
    """Run data quality checks on a dataset."""
    configure_logging(json_format=json_logs)

    df = _load_data(source)
    checker = QualityChecker(null_threshold=null_threshold)
    result = checker.check(df, label_column=label, dataset_name=Path(source).name)

    # Display summary
    console.print(f"\n[bold]Quality Check Results[/bold]")
    console.print(f"Total rows: {result.total_rows:,}")
    console.print(f"Total columns: {result.total_columns}")
    console.print(f"Warnings: {len(result.warnings)}")

    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in result.warnings:
            console.print(f"  ⚠ {warning}")

    # Display column quality table
    table = Table(title="Column Quality Summary")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Null Rate", justify="right")
    table.add_column("Unique Rate", justify="right")
    table.add_column("Outlier Rate", justify="right")

    for col in result.columns:
        null_color = "red" if col.null_rate > null_threshold else "green"
        outlier_str = f"{col.outlier_rate:.1%}" if col.outlier_rate else "N/A"
        table.add_row(
            col.column,
            col.dtype,
            f"[{null_color}]{col.null_rate:.1%}[/{null_color}]",
            f"{col.unique_rate:.1%}",
            outlier_str,
        )
    console.print(table)

    # Generate report
    if report:
        generator = HTMLReportGenerator()
        generator.generate_quality_report(report, result)
        console.print(f"\n[green]Report saved to {report}[/green]")


@app.command()
def evaluate(
    source: Annotated[str, typer.Argument(help="Data source")],
    rules: Annotated[str, typer.Option("--rules", help="Path to rules YAML file")],
    label: Annotated[str, typer.Option("--label", "-l", help="Label column name")] = "is_bad",
    report: Annotated[Optional[str], typer.Option("--report", "-r", help="Output HTML report path")] = None,
    json_logs: Annotated[bool, typer.Option("--json-logs")] = False,
) -> None:
    """Evaluate a single rule set against labeled data."""
    configure_logging(json_format=json_logs)

    df = _load_data(source)
    engine = _load_rules(rules)

    # Execute rules
    console.print("[blue]Executing rules...[/blue]")
    rule_result = engine.execute(df)

    # Evaluate
    console.print("[blue]Computing metrics...[/blue]")
    evaluator = Evaluator(label_column=label)
    metrics = evaluator.evaluate(
        df,
        rule_result.final_decisions,
        strategy_name=engine.config.name,
    )

    # Display metrics
    console.print(f"\n[bold]Evaluation Results: {engine.config.name}[/bold]")

    table = Table(title="Key Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    metric_rows = [
        ("Sample Size", f"{metrics.sample_size:,}"),
        ("Hit Rate", f"{metrics.hit_rate:.1%}"),
        ("Pass Rate", f"{metrics.pass_rate:.1%}"),
        ("Recall", f"{metrics.recall:.1%}" if metrics.recall is not None else "N/A"),
        ("Precision", f"{metrics.precision:.1%}" if metrics.precision is not None else "N/A"),
        ("False Positive Rate", f"{metrics.false_positive_rate:.1%}" if metrics.false_positive_rate is not None else "N/A"),
        ("False Negative Rate", f"{metrics.false_negative_rate:.1%}" if metrics.false_negative_rate is not None else "N/A"),
        ("Accuracy", f"{metrics.accuracy:.1%}" if metrics.accuracy is not None else "N/A"),
        ("F1 Score", f"{metrics.f1_score:.3f}" if metrics.f1_score is not None else "N/A"),
        ("KS Statistic", f"{metrics.ks_statistic:.3f}" if metrics.ks_statistic is not None else "N/A"),
        ("AUC-ROC", f"{metrics.auc_roc:.3f}" if metrics.auc_roc is not None else "N/A"),
        ("Lift", f"{metrics.lift:.2f}" if metrics.lift is not None else "N/A"),
    ]

    for name, value in metric_rows:
        table.add_row(name, value)
    console.print(table)

    # Display rule hit summary
    rule_table = Table(title="Rule Hit Summary")
    rule_table.add_column("Rule", style="cyan")
    rule_table.add_column("Hits", justify="right")
    rule_table.add_column("Hit Rate", justify="right")
    rule_table.add_column("Action", justify="center")

    for name, summary in rule_result.rule_summaries.items():
        rule_table.add_row(
            name,
            f"{summary['hit_count']:,}",
            f"{summary['hit_rate']:.1%}",
            summary["action"],
        )
    console.print(rule_table)

    # Generate report
    if report:
        chart_gen = ChartGenerator()
        charts = {
            "confusion_matrix": chart_gen.create_confusion_matrix_heatmap(
                metrics.confusion_matrix, f"Confusion Matrix - {engine.config.name}"
            ),
            "metrics_gauge": chart_gen.create_metrics_gauge(metrics, engine.config.name),
            "rule_hits": chart_gen.create_rule_hit_distribution(rule_result.rule_hit_matrix),
        }

        generator = HTMLReportGenerator()
        generator.generate_evaluation_report(report, metrics, rule_result, charts)
        console.print(f"\n[green]Report saved to {report}[/green]")


@app.command()
def compare(
    source: Annotated[str, typer.Argument(help="Data source")],
    rules_a: Annotated[str, typer.Option("--rules-a", help="Path to rules A YAML")],
    rules_b: Annotated[str, typer.Option("--rules-b", help="Path to rules B YAML")],
    label: Annotated[str, typer.Option("--label", "-l", help="Label column name")] = "is_bad",
    report: Annotated[Optional[str], typer.Option("--report", "-r", help="Output HTML report path")] = None,
    json_logs: Annotated[bool, typer.Option("--json-logs")] = False,
) -> None:
    """Compare two rule sets (A/B test)."""
    configure_logging(json_format=json_logs)

    df = _load_data(source)
    engine_a = _load_rules(rules_a)
    engine_b = _load_rules(rules_b)

    # Execute both
    console.print("[blue]Executing strategy A...[/blue]")
    result_a = engine_a.execute(df)
    console.print("[blue]Executing strategy B...[/blue]")
    result_b = engine_b.execute(df)

    # Compare
    console.print("[blue]Computing comparison metrics...[/blue]")
    evaluator = Evaluator(label_column=label)
    comparison = evaluator.compare(
        df,
        result_a.final_decisions,
        result_b.final_decisions,
        name_a=engine_a.config.name,
        name_b=engine_b.config.name,
    )

    # Display comparison
    console.print(f"\n[bold]A/B Comparison: {engine_a.config.name} vs {engine_b.config.name}[/bold]")
    console.print(f"[italic]{comparison.recommendation}[/italic]\n")

    table = Table(title="Metric Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(engine_a.config.name, justify="right")
    table.add_column(engine_b.config.name, justify="right")
    table.add_column("Diff", justify="right")

    for key, diff in comparison.differences.items():
        diff_str = f"{diff['absolute_diff']:+.1%}"
        diff_color = "green" if diff["absolute_diff"] > 0 else "red"
        table.add_row(
            key.replace("_", " ").title(),
            f"{diff['a']:.1%}",
            f"{diff['b']:.1%}",
            f"[{diff_color}]{diff_str}[/{diff_color}]",
        )
    console.print(table)

    # Generate report
    if report:
        chart_gen = ChartGenerator()
        charts = {
            "comparison": chart_gen.create_comparison_bar(
                comparison.metrics_a,
                comparison.metrics_b,
                engine_a.config.name,
                engine_b.config.name,
            ),
        }

        generator = HTMLReportGenerator()
        generator.generate(
            report,
            title=f"A/B Comparison: {engine_a.config.name} vs {engine_b.config.name}",
            comparison=comparison,
            charts=charts,
        )
        console.print(f"\n[green]Report saved to {report}[/green]")


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to run Streamlit on")] = 8501,
) -> None:
    """Launch the Streamlit web interface."""
    import subprocess
    import sys

    web_app = Path(__file__).parent.parent / "web" / "app.py"
    console.print(f"[blue]Starting Streamlit on port {port}...[/blue]")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(web_app), "--server.port", str(port)],
        check=False,
    )


@app.callback()
def main(
    version: Annotated[Optional[bool], typer.Option("--version", "-v", help="Show version")] = None,
) -> None:
    """Risk Control Evaluator - Professional toolkit for risk data quality and strategy evaluation."""
    if version:
        console.print(f"risk-eval version {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
