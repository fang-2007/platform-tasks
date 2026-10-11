"""Interactive chart generation using Plotly."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

from risk_eval.core.evaluator import ConfusionMatrix, EvaluationMetrics
from risk_eval.utils.logger import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """Generates interactive Plotly charts for evaluation reports."""

    def create_confusion_matrix_heatmap(
        self, cm: ConfusionMatrix, title: str = "Confusion Matrix"
    ) -> go.Figure:
        """Create an annotated confusion matrix heatmap.

        Args:
            cm: Confusion matrix data.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        z = [[cm.true_negative, cm.false_positive], [cm.false_negative, cm.true_positive]]
        labels = [["TN", "FP"], ["FN", "TP"]]
        annotations = [
            [
                f"{labels[i][j]}<br>{z[i][j]}<br>({z[i][j]/cm.total*100:.1f}%)"
                for j in range(2)
            ]
            for i in range(2)
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale="Blues",
                showscale=False,
                text=annotations,
                texttemplate="%{text}",
                textfont={"size": 16},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Prediction",
            yaxis_title="Actual",
            width=500,
            height=450,
        )
        return fig

    def create_metrics_gauge(
        self, metrics: EvaluationMetrics, title: str = "Strategy Performance"
    ) -> go.Figure:
        """Create gauge charts for key metrics.

        Args:
            metrics: Evaluation metrics.
            title: Chart title.

        Returns:
            Plotly figure with subplots.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}],
            ],
            subplot_titles=("Recall", "Precision", "FPR (Misjudgment)", "Pass Rate"),
        )

        metric_values = [
            ("Recall", metrics.recall, 0),
            ("Precision", metrics.precision, 1),
            ("FPR", metrics.false_positive_rate, 2),
            ("Pass Rate", metrics.pass_rate, 3),
        ]

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for (name, value, idx), (row, col) in zip(metric_values, positions):
            if value is None:
                value = 0.0
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=round(value * 100, 1),
                    number={"suffix": "%"},
                    title={"text": name, "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "lightgreen"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 80,
                        },
                    },
                ),
                row=row,
                col=col,
            )

        fig.update_layout(title_text=title, height=600)
        return fig

    def create_roc_curve(
        self,
        df: pd.DataFrame,
        label_column: str,
        scores: pd.Series,
        title: str = "ROC Curve",
    ) -> go.Figure:
        """Create ROC curve.

        Args:
            df: DataFrame with labels.
            label_column: Name of label column.
            scores: Predicted scores/probabilities.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        labels = df[label_column].astype(int)
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.3f})",
                line={"color": "darkblue", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line={"color": "gray", "width": 1, "dash": "dash"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            width=600,
            height=500,
            legend={"x": 0.6, "y": 0.1},
        )
        return fig

    def create_rule_hit_distribution(
        self, rule_hit_matrix: pd.DataFrame, title: str = "Rule Hit Distribution"
    ) -> go.Figure:
        """Create bar chart of rule hit rates.

        Args:
            rule_hit_matrix: DataFrame of rule hits (boolean).
            title: Chart title.

        Returns:
            Plotly figure.
        """
        hit_rates = rule_hit_matrix.mean().sort_values(ascending=True)

        colors = ["green" if r < 0.1 else "orange" if r < 0.3 else "red" for r in hit_rates]

        fig = go.Figure(
            go.Bar(
                x=hit_rates.values * 100,
                y=hit_rates.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in hit_rates.values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Hit Rate (%)",
            yaxis_title="Rule",
            width=700,
            height=max(400, len(hit_rates) * 30),
        )
        return fig

    def create_comparison_bar(
        self,
        metrics_a: EvaluationMetrics,
        metrics_b: EvaluationMetrics,
        name_a: str,
        name_b: str,
        title: str = "Strategy Comparison",
    ) -> go.Figure:
        """Create side-by-side bar chart comparing two strategies.

        Args:
            metrics_a: Metrics for strategy A.
            metrics_b: Metrics for strategy B.
            name_a: Name of strategy A.
            name_b: Name of strategy B.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        fields = [
            ("Recall", "recall"),
            ("Precision", "precision"),
            ("FPR", "false_positive_rate"),
            ("FNR", "false_negative_rate"),
            ("Accuracy", "accuracy"),
            ("F1", "f1_score"),
            ("Hit Rate", "hit_rate"),
            ("Pass Rate", "pass_rate"),
        ]

        labels = []
        values_a = []
        values_b = []

        for label, field in fields:
            val_a = getattr(metrics_a, field)
            val_b = getattr(metrics_b, field)
            if val_a is not None and val_b is not None:
                labels.append(label)
                values_a.append(val_a * 100)
                values_b.append(val_b * 100)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=name_a,
                x=labels,
                y=values_a,
                marker_color="steelblue",
                text=[f"{v:.1f}%" for v in values_a],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                name=name_b,
                x=labels,
                y=values_b,
                marker_color="coral",
                text=[f"{v:.1f}%" for v in values_b],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Metric",
            yaxis_title="Value (%)",
            barmode="group",
            width=800,
            height=500,
            legend={"x": 0.01, "y": 0.99},
        )
        return fig

    def create_ks_plot(
        self,
        df: pd.DataFrame,
        label_column: str,
        scores: pd.Series,
        title: str = "KS Statistic",
    ) -> go.Figure:
        """Create cumulative distribution plot for KS statistic.

        Args:
            df: DataFrame with labels.
            label_column: Name of label column.
            scores: Predicted scores.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        labels = df[label_column]
        pos_scores = scores[labels == True]  # noqa: E712
        neg_scores = scores[labels == False]  # noqa: E712

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for KS plot",
                showarrow=False,
                font={"size": 16},
            )
            return fig

        # Create cumulative distributions
        pos_sorted = np.sort(pos_scores)
        neg_sorted = np.sort(neg_scores)
        all_scores = np.sort(np.concatenate([pos_sorted, neg_sorted]))

        pos_cdf = np.searchsorted(pos_sorted, all_scores, side="right") / len(pos_sorted)
        neg_cdf = np.searchsorted(neg_sorted, all_scores, side="right") / len(neg_sorted)

        diff = np.abs(pos_cdf - neg_cdf)
        ks_idx = np.argmax(diff)
        ks_value = diff[ks_idx]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=all_scores,
                y=pos_cdf,
                mode="lines",
                name="Positive (Bad) CDF",
                line={"color": "red"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=all_scores,
                y=neg_cdf,
                mode="lines",
                name="Negative (Good) CDF",
                line={"color": "blue"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[all_scores[ks_idx], all_scores[ks_idx]],
                y=[pos_cdf[ks_idx], neg_cdf[ks_idx]],
                mode="lines",
                name=f"KS = {ks_value:.3f}",
                line={"color": "green", "dash": "dash", "width": 3},
            )
        )

        fig.update_layout(
            title=f"{title} (KS = {ks_value:.3f})",
            xaxis_title="Score",
            yaxis_title="Cumulative Distribution",
            width=600,
            height=450,
        )
        return fig
