"""Strategy evaluation metrics for risk control rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import auc, roc_auc_score, roc_curve

from risk_eval.utils.exceptions import EvaluationError
from risk_eval.utils.logger import get_logger

logger = get_logger(__name__)


class ConfusionMatrix(BaseModel):
    """Confusion matrix for binary classification."""

    true_positive: int = Field(..., ge=0)
    false_positive: int = Field(..., ge=0)
    true_negative: int = Field(..., ge=0)
    false_negative: int = Field(..., ge=0)

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def positive_actual(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def negative_actual(self) -> int:
        return self.true_negative + self.false_positive

    @property
    def positive_predicted(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def negative_predicted(self) -> int:
        return self.true_negative + self.false_negative


class EvaluationMetrics(BaseModel):
    """Core evaluation metrics for a strategy."""

    sample_size: int
    hit_count: int
    hit_rate: float = Field(..., ge=0.0, le=1.0)
    pass_rate: float = Field(..., ge=0.0, le=1.0)

    # Classification metrics
    recall: float | None = Field(None, ge=0.0, le=1.0)
    precision: float | None = Field(None, ge=0.0, le=1.0)
    false_positive_rate: float | None = Field(None, ge=0.0, le=1.0)
    false_negative_rate: float | None = Field(None, ge=0.0, le=1.0)
    specificity: float | None = Field(None, ge=0.0, le=1.0)
    accuracy: float | None = Field(None, ge=0.0, le=1.0)
    f1_score: float | None = Field(None, ge=0.0, le=1.0)

    # Risk-specific metrics
    bad_rate_overall: float | None = Field(None, ge=0.0, le=1.0)
    bad_rate_in_hits: float | None = Field(None, ge=0.0, le=1.0)
    bad_rate_in_passed: float | None = Field(None, ge=0.0, le=1.0)
    lift: float | None = None

    # Statistical metrics
    ks_statistic: float | None = Field(None, ge=0.0, le=1.0)
    auc_roc: float | None = Field(None, ge=0.0, le=1.0)

    confusion_matrix: ConfusionMatrix | None = None


class ComparisonResult(BaseModel):
    """Result of comparing two strategies."""

    strategy_a_name: str
    strategy_b_name: str
    metrics_a: EvaluationMetrics
    metrics_b: EvaluationMetrics
    differences: dict[str, Any]
    recommendation: str = ""


class Evaluator:
    """Evaluates risk control strategies against labeled data."""

    def __init__(self, label_column: str = "is_bad", positive_label: Any = 1) -> None:
        """Initialize evaluator.

        Args:
            label_column: Name of the target/label column.
            positive_label: Value indicating positive class (bad/default).
        """
        self.label_column = label_column
        self.positive_label = positive_label

    def evaluate(
        self,
        df: pd.DataFrame,
        predictions: pd.Series,
        strategy_name: str = "",
        scores: pd.Series | None = None,
    ) -> EvaluationMetrics:
        """Evaluate a strategy's performance.

        Args:
            df: DataFrame containing features and label.
            predictions: Series of predictions (accept/reject or boolean).
            strategy_name: Optional name for the strategy.
            scores: Optional continuous scores for ROC/KS calculation.

        Returns:
            Computed evaluation metrics.

        Raises:
            EvaluationError: If label column missing or evaluation fails.
        """
        if self.label_column not in df.columns:
            raise EvaluationError(
                f"Label column '{self.label_column}' not found",
                context={"available_columns": list(df.columns)},
            )

        labels = df[self.label_column]

        # Convert predictions to boolean (True = rejected/hit)
        pred_bool = self._convert_to_boolean(predictions)
        label_bool = labels == self.positive_label

        # Handle missing labels
        valid_mask = labels.notna()
        if not valid_mask.all():
            logger.warning(
                "Dropping rows with missing labels",
                dropped=int((~valid_mask).sum()),
            )
            pred_bool = pred_bool[valid_mask]
            label_bool = label_bool[valid_mask]
            labels = labels[valid_mask]
            if scores is not None:
                scores = scores[valid_mask]

        cm = self._compute_confusion_matrix(label_bool, pred_bool)
        metrics = self._compute_metrics(cm, label_bool, pred_bool, scores)
        metrics.confusion_matrix = cm

        logger.info(
            "Evaluation complete",
            strategy=strategy_name or "unnamed",
            recall=metrics.recall,
            fpr=metrics.false_positive_rate,
        )
        return metrics

    def compare(
        self,
        df: pd.DataFrame,
        predictions_a: pd.Series,
        predictions_b: pd.Series,
        name_a: str = "Strategy A",
        name_b: str = "Strategy B",
        scores_a: pd.Series | None = None,
        scores_b: pd.Series | None = None,
    ) -> ComparisonResult:
        """Compare two strategies side by side.

        Args:
            df: DataFrame with features and label.
            predictions_a: Predictions from strategy A.
            predictions_b: Predictions from strategy B.
            name_a: Name of strategy A.
            name_b: Name of strategy B.
            scores_a: Optional scores for strategy A.
            scores_b: Optional scores for strategy B.

        Returns:
            Comparison result with differences and recommendation.
        """
        metrics_a = self.evaluate(df, predictions_a, name_a, scores_a)
        metrics_b = self.evaluate(df, predictions_b, name_b, scores_b)

        differences = self._compute_differences(metrics_a, metrics_b)
        recommendation = self._generate_recommendation(
            metrics_a, metrics_b, differences, name_a, name_b
        )

        return ComparisonResult(
            strategy_a_name=name_a,
            strategy_b_name=name_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            differences=differences,
            recommendation=recommendation,
        )

    def _convert_to_boolean(self, predictions: pd.Series) -> pd.Series:
        """Convert various prediction formats to boolean (True = rejected/hit)."""
        if predictions.dtype == bool:
            return predictions

        unique_vals = predictions.dropna().unique()
        if set(unique_vals).issubset({"accept", "reject", "review"}):
            return predictions.isin({"reject", "review"})

        # Assume 0/1 or other binary where non-zero is positive
        return predictions != 0

    def _compute_confusion_matrix(
        self, labels: pd.Series, predictions: pd.Series
    ) -> ConfusionMatrix:
        """Compute confusion matrix from boolean labels and predictions."""
        tp = int((labels & predictions).sum())
        fp = int((~labels & predictions).sum())
        tn = int((~labels & ~predictions).sum())
        fn = int((labels & ~predictions).sum())

        return ConfusionMatrix(
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
        )

    def _compute_metrics(
        self,
        cm: ConfusionMatrix,
        labels: pd.Series,
        predictions: pd.Series,
        scores: pd.Series | None,
    ) -> EvaluationMetrics:
        """Compute all evaluation metrics from confusion matrix."""
        n = cm.total
        hit_count = cm.positive_predicted
        hit_rate = hit_count / n if n > 0 else 0.0
        pass_rate = 1 - hit_rate

        # Classification metrics
        recall = cm.true_positive / cm.positive_actual if cm.positive_actual > 0 else None
        precision = cm.true_positive / cm.positive_predicted if cm.positive_predicted > 0 else None
        fpr = cm.false_positive / cm.negative_actual if cm.negative_actual > 0 else None
        fnr = cm.false_negative / cm.positive_actual if cm.positive_actual > 0 else None
        specificity = cm.true_negative / cm.negative_actual if cm.negative_actual > 0 else None
        accuracy = (cm.true_positive + cm.true_negative) / n if n > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else None
        )

        # Risk-specific metrics
        bad_rate_overall = cm.positive_actual / n if n > 0 else None
        bad_rate_in_hits = cm.true_positive / hit_count if hit_count > 0 else None
        bad_rate_in_passed = cm.false_negative / cm.negative_predicted if cm.negative_predicted > 0 else None

        lift = (
            bad_rate_in_hits / bad_rate_overall
            if bad_rate_in_hits and bad_rate_overall and bad_rate_overall > 0
            else None
        )

        # Statistical metrics
        ks = self._compute_ks(labels, scores) if scores is not None else None
        auc_val = self._compute_auc(labels, scores) if scores is not None else None

        return EvaluationMetrics(
            sample_size=n,
            hit_count=hit_count,
            hit_rate=round(hit_rate, 4),
            pass_rate=round(pass_rate, 4),
            recall=round(recall, 4) if recall is not None else None,
            precision=round(precision, 4) if precision is not None else None,
            false_positive_rate=round(fpr, 4) if fpr is not None else None,
            false_negative_rate=round(fnr, 4) if fnr is not None else None,
            specificity=round(specificity, 4) if specificity is not None else None,
            accuracy=round(accuracy, 4) if accuracy is not None else None,
            f1_score=round(f1, 4) if f1 is not None else None,
            bad_rate_overall=round(bad_rate_overall, 4) if bad_rate_overall is not None else None,
            bad_rate_in_hits=round(bad_rate_in_hits, 4) if bad_rate_in_hits is not None else None,
            bad_rate_in_passed=round(bad_rate_in_passed, 4) if bad_rate_in_passed is not None else None,
            lift=round(lift, 4) if lift is not None else None,
            ks_statistic=round(ks, 4) if ks is not None else None,
            auc_roc=round(auc_val, 4) if auc_val is not None else None,
        )

    def _compute_ks(self, labels: pd.Series, scores: pd.Series) -> float | None:
        """Compute Kolmogorov-Smirnov statistic."""
        try:
            pos_scores = scores[labels == True]  # noqa: E712
            neg_scores = scores[labels == False]  # noqa: E712

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                return None

            from scipy import stats
            ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
            return float(ks_stat)
        except Exception as exc:
            logger.warning("KS computation failed", error=str(exc))
            return None

    def _compute_auc(self, labels: pd.Series, scores: pd.Series) -> float | None:
        """Compute AUC-ROC."""
        try:
            if labels.nunique() < 2:
                return None
            return float(roc_auc_score(labels.astype(int), scores))
        except Exception as exc:
            logger.warning("AUC computation failed", error=str(exc))
            return None

    def _compute_differences(
        self, a: EvaluationMetrics, b: EvaluationMetrics
    ) -> dict[str, Any]:
        """Compute differences between two metric sets."""
        diffs: dict[str, Any] = {}
        fields = [
            "recall", "precision", "false_positive_rate", "false_negative_rate",
            "accuracy", "f1_score", "hit_rate", "pass_rate",
            "bad_rate_in_hits", "bad_rate_in_passed", "lift",
            "ks_statistic", "auc_roc",
        ]

        for field in fields:
            val_a = getattr(a, field)
            val_b = getattr(b, field)
            if val_a is not None and val_b is not None:
                diff = val_b - val_a
                pct_diff = (diff / val_a * 100) if val_a != 0 else None
                diffs[field] = {
                    "a": val_a,
                    "b": val_b,
                    "absolute_diff": round(diff, 4),
                    "relative_diff_pct": round(pct_diff, 2) if pct_diff is not None else None,
                }

        return diffs

    def _generate_recommendation(
        self,
        a: EvaluationMetrics,
        b: EvaluationMetrics,
        diffs: dict[str, Any],
        name_a: str,
        name_b: str,
    ) -> str:
        """Generate a recommendation based on comparison."""
        if not diffs:
            return "Insufficient data for recommendation."

        recall_diff = diffs.get("recall", {}).get("absolute_diff", 0)
        fpr_diff = diffs.get("false_positive_rate", {}).get("absolute_diff", 0)
        pass_rate_diff = diffs.get("pass_rate", {}).get("absolute_diff", 0)

        parts = []

        if recall_diff > 0.05 and fpr_diff < 0.05:
            parts.append(f"{name_b} shows significantly better recall with similar FPR.")
        elif recall_diff > 0.02:
            parts.append(f"{name_b} has better recall.")
        elif recall_diff < -0.02:
            parts.append(f"{name_a} has better recall.")

        if fpr_diff < -0.05:
            parts.append(f"{name_b} has lower false positive rate.")
        elif fpr_diff > 0.05:
            parts.append(f"{name_a} has lower false positive rate.")

        if pass_rate_diff > 0.05:
            parts.append(f"{name_b} allows more approvals (higher pass rate).")
        elif pass_rate_diff < -0.05:
            parts.append(f"{name_a} allows more approvals.")

        return " ".join(parts) if parts else "Both strategies perform similarly."
