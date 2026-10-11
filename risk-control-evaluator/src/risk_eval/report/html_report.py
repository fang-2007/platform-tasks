"""HTML report generation for evaluation results."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from jinja2 import Template

from risk_eval.core.evaluator import ComparisonResult, EvaluationMetrics
from risk_eval.core.quality_checker import QualityReport
from risk_eval.core.rule_engine import RuleExecutionSummary
from risk_eval.utils.logger import get_logger

logger = get_logger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-700: #374151;
            --gray-900: #111827;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--gray-50);
            color: var(--gray-900);
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        header {
            background: linear-gradient(135deg, var(--primary), #1e40af);
            color: white;
            padding: 2.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        header h1 { font-size: 1.875rem; font-weight: 700; margin-bottom: 0.5rem; }
        header p { opacity: 0.9; font-size: 0.95rem; }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-success { background: #dcfce7; color: var(--success); }
        .badge-warning { background: #fef3c7; color: var(--warning); }
        .badge-danger { background: #fee2e2; color: var(--danger); }
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .card h2 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--gray-900);
            border-bottom: 2px solid var(--gray-100);
            padding-bottom: 0.5rem;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: var(--gray-50);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid var(--gray-200);
        }
        .metric-value {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--primary);
        }
        .metric-label {
            font-size: 0.875rem;
            color: var(--gray-700);
            margin-top: 0.25rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }
        th {
            background: var(--gray-100);
            font-weight: 600;
            color: var(--gray-700);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }
        tr:hover { background: var(--gray-50); }
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .alert-warning { background: #fef3c7; border-left: 4px solid var(--warning); }
        .alert-danger { background: #fee2e2; border-left: 4px solid var(--danger); }
        .comparison-table th { text-align: center; }
        .comparison-table td { text-align: center; }
        .better { color: var(--success); font-weight: 600; }
        .worse { color: var(--danger); font-weight: 600; }
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray-700);
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p>Generated on {{ generated_at }} | Risk Control Evaluator v{{ version }}</p>
        </header>

        {% if quality_report %}
        <div class="card">
            <h2>Data Quality Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ quality_report.total_rows }}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ quality_report.total_columns }}</div>
                    <div class="metric-label">Total Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ quality_report.warnings | length }}</div>
                    <div class="metric-label">Warnings</div>
                </div>
            </div>
            {% if quality_report.warnings %}
            <div class="alert alert-warning">
                <strong>Warnings:</strong>
                <ul>
                {% for warning in quality_report.warnings %}
                    <li>{{ warning }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}

        {% if rule_summary %}
        <div class="card">
            <h2>Rule Execution Summary</h2>
            <p><strong>Strategy:</strong> {{ rule_summary.config_name }} ({{ rule_summary.config_version }})</p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ rule_summary.total_rows }}</div>
                    <div class="metric-label">Rows Evaluated</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ rule_summary.rule_count }}</div>
                    <div class="metric-label">Rules Executed</div>
                </div>
            </div>
        </div>
        {% endif %}

        {% if metrics %}
        <div class="card">
            <h2>Evaluation Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f" | format(metrics.recall * 100) if metrics.recall else "N/A" }}%</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f" | format(metrics.precision * 100) if metrics.precision else "N/A" }}%</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f" | format(metrics.false_positive_rate * 100) if metrics.false_positive_rate else "N/A" }}%</div>
                    <div class="metric-label">False Positive Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f" | format(metrics.hit_rate * 100) }}%</div>
                    <div class="metric-label">Hit Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f" | format(metrics.pass_rate * 100) }}%</div>
                    <div class="metric-label">Pass Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f" | format(metrics.ks_statistic) if metrics.ks_statistic else "N/A" }}</div>
                    <div class="metric-label">KS Statistic</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f" | format(metrics.auc_roc) if metrics.auc_roc else "N/A" }}</div>
                    <div class="metric-label">AUC-ROC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f" | format(metrics.lift) if metrics.lift else "N/A" }}</div>
                    <div class="metric-label">Lift</div>
                </div>
            </div>
        </div>
        {% endif %}

        {% if comparison %}
        <div class="card">
            <h2>Strategy Comparison: {{ comparison.strategy_a_name }} vs {{ comparison.strategy_b_name }}</h2>
            <p><strong>Recommendation:</strong> {{ comparison.recommendation }}</p>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>{{ comparison.strategy_a_name }}</th>
                        <th>{{ comparison.strategy_b_name }}</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                {% for key, diff in comparison.differences.items() %}
                    <tr>
                        <td style="text-transform: capitalize;">{{ key.replace('_', ' ') }}</td>
                        <td>{{ "%.2f" | format(diff.a * 100) }}%</td>
                        <td>{{ "%.2f" | format(diff.b * 100) }}%</td>
                        <td class="{{ 'better' if diff.absolute_diff > 0 else 'worse' }}">
                            {{ "+%.2f" | format(diff.absolute_diff * 100) if diff.absolute_diff > 0 else "%.2f" | format(diff.absolute_diff * 100) }}%
                        </td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% for chart_id, chart_div in charts.items() %}
        <div class="chart-container">
            {{ chart_div | safe }}
        </div>
        {% endfor %}

        <footer>
            <p>Generated by Risk Control Evaluator &mdash; Professional Risk Data Quality & Strategy Assessment</p>
        </footer>
    </div>
</body>
</html>
"""


class HTMLReportGenerator:
    """Generates standalone HTML reports with embedded Plotly charts."""

    def __init__(self) -> None:
        """Initialize report generator."""
        self.template = Template(HTML_TEMPLATE)

    def generate(
        self,
        output_path: str | Path,
        *,
        title: str = "Risk Control Evaluation Report",
        quality_report: QualityReport | None = None,
        rule_summary: dict[str, Any] | None = None,
        metrics: EvaluationMetrics | None = None,
        comparison: ComparisonResult | None = None,
        charts: dict[str, go.Figure] | None = None,
        version: str = "0.1.0",
    ) -> Path:
        """Generate HTML report and save to file.
        
        Args:
            output_path: Path to save the HTML file.
            title: Report title.
            quality_report: Optional quality check report.
            rule_summary: Optional rule execution summary dict.
            metrics: Optional evaluation metrics.
            comparison: Optional comparison result.
            charts: Optional dictionary of Plotly figures.
            version: Tool version string.

        Returns:
            Path to generated HTML file.
        """
        chart_divs: dict[str, str] = {}
        if charts:
            for chart_id, fig in charts.items():
                chart_divs[chart_id] = fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                )

        context = {
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": version,
            "quality_report": quality_report,
            "rule_summary": rule_summary,
            "metrics": metrics,
            "comparison": comparison,
            "charts": chart_divs,
        }

        html = self.template.render(**context)
        output = Path(output_path)
        output.write_text(html, encoding="utf-8")
        logger.info("HTML report generated", path=str(output))
        return output

    def generate_quality_report(
        self,
        output_path: str | Path,
        report: QualityReport,
        charts: dict[str, go.Figure] | None = None,
    ) -> Path:
        """Generate a quality-only report.

        Args:
            output_path: Output file path.
            report: Quality report.
            charts: Optional charts.

        Returns:
            Path to generated file.
        """
        return self.generate(
            output_path,
            title=f"Data Quality Report - {report.dataset_name or 'Dataset'}",
            quality_report=report,
            charts=charts,
        )

    def generate_evaluation_report(
        self,
        output_path: str | Path,
        metrics: EvaluationMetrics,
        rule_summary: RuleExecutionSummary | None = None,
        charts: dict[str, go.Figure] | None = None,
    ) -> Path:
        """Generate an evaluation-only report.

        Args:
            output_path: Output file path.
            metrics: Evaluation metrics.
            rule_summary: Optional rule execution summary.
            charts: Optional charts.

        Returns:
            Path to generated file.
        """
        summary_dict: dict[str, Any] | None = None
        if rule_summary:
            summary_dict = {
                "config_name": getattr(rule_summary, "config_name", "Unknown"),
                "config_version": getattr(rule_summary, "config_version", "1.0.0"),
                "total_rows": rule_summary.total_rows,
                "rule_count": len(rule_summary.rule_summaries),
            }

        return self.generate(
            output_path,
            title="Strategy Evaluation Report",
            rule_summary=summary_dict,
            metrics=metrics,
            charts=charts,
        )
