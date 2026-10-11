"""Tests for strategy evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_eval.core.evaluator import ComparisonResult, ConfusionMatrix, Evaluator, EvaluationMetrics


class TestEvaluator:
    """Tests for Evaluator."""

    @pytest.fixture
    def labeled_df(self) -> pd.DataFrame:
        """Return a labeled DataFrame for testing."""
        return pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "is_bad": [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        })

    def test_evaluate_perfect_predictions(self, labeled_df: pd.DataFrame) -> None:
        """Test evaluation with perfect predictions."""
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series(["accept", "accept", "accept", "accept", "reject",
                                 "accept", "reject", "reject", "reject", "reject"])
        metrics = evaluator.evaluate(labeled_df, predictions)

        assert metrics.recall == 1.0
        assert metrics.precision == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.false_positive_rate == 0.0

    def test_evaluate_all_reject(self, labeled_df: pd.DataFrame) -> None:
        """Test evaluation when all predictions are reject."""
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series(["reject"] * len(labeled_df))
        metrics = evaluator.evaluate(labeled_df, predictions)

        assert metrics.recall == 1.0
        assert metrics.precision == 0.5  # 5 bad out of 10 total
        assert metrics.false_positive_rate == 1.0
        assert metrics.pass_rate == 0.0

    def test_evaluate_all_accept(self, labeled_df: pd.DataFrame) -> None:
        """Test evaluation when all predictions are accept."""
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series(["accept"] * len(labeled_df))
        metrics = evaluator.evaluate(labeled_df, predictions)

        assert metrics.recall == 0.0
        assert metrics.precision is None  # No positive predictions
        assert metrics.false_positive_rate == 0.0
        assert metrics.pass_rate == 1.0

    def test_evaluate_with_boolean_predictions(self, labeled_df: pd.DataFrame) -> None:
        """Test evaluation with boolean predictions."""
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series([False, False, False, False, True,
                                 False, True, True, True, True])
        metrics = evaluator.evaluate(labeled_df, predictions)

        assert metrics.recall == 1.0
        assert metrics.accuracy == 1.0

    def test_evaluate_missing_label_column_raises(self, labeled_df: pd.DataFrame) -> None:
        """Test that missing label column raises error."""
        evaluator = Evaluator(label_column="nonexistent")
        with pytest.raises(Exception) as exc_info:
            evaluator.evaluate(labeled_df, pd.Series(["accept"] * len(labeled_df)))
        assert "not found" in str(exc_info.value)

    def test_compare_strategies(self, labeled_df: pd.DataFrame) -> None:
        """Test comparing two strategies."""
        evaluator = Evaluator(label_column="is_bad")
        pred_a = pd.Series(["accept", "accept", "accept", "accept", "reject",
                            "accept", "reject", "reject", "reject", "reject"])
        pred_b = pd.Series(["reject", "reject", "reject", "reject", "reject",
                            "reject", "reject", "reject", "reject", "reject"])

        comparison = evaluator.compare(labeled_df, pred_a, pred_b, name_a="A", name_b="B")

        assert isinstance(comparison, ComparisonResult)
        assert comparison.strategy_a_name == "A"
        assert comparison.strategy_b_name == "B"
        assert len(comparison.differences) > 0
        assert comparison.recommendation != ""

    def test_confusion_matrix_properties(self) -> None:
        """Test ConfusionMatrix dataclass."""
        cm = ConfusionMatrix(
            true_positive=80,
            false_positive=20,
            true_negative=70,
            false_negative=30,
        )

        assert cm.total == 200
        assert cm.positive_actual == 110
        assert cm.negative_actual == 90
        assert cm.positive_predicted == 100
        assert cm.negative_predicted == 100

    def test_metrics_with_scores(self, labeled_df: pd.DataFrame) -> None:
        """Test evaluation with continuous scores."""
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series(["accept", "accept", "accept", "accept", "reject",
                                 "accept", "reject", "reject", "reject", "reject"])
        scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.8, 0.85, 0.95, 0.99])

        metrics = evaluator.evaluate(labeled_df, predictions, scores=scores)

        assert metrics.auc_roc is not None
        assert 0.0 <= metrics.auc_roc <= 1.0

    def test_evaluate_with_missing_labels(self) -> None:
        """Test evaluation handles missing labels gracefully."""
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "is_bad": [0, 0, np.nan, 1, 1],
        })
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series(["accept", "accept", "reject", "reject", "reject"])

        metrics = evaluator.evaluate(df, predictions)
        assert metrics.sample_size == 4  # 1 row with missing label dropped

    def test_lift_calculation(self, labeled_df: pd.DataFrame) -> None:
        """Test lift metric calculation."""
        evaluator = Evaluator(label_column="is_bad")
        # Reject only bad ones
        predictions = pd.Series(["accept", "accept", "accept", "accept", "reject",
                                 "accept", "reject", "reject", "reject", "reject"])
        metrics = evaluator.evaluate(labeled_df, predictions)

        assert metrics.lift is not None
        assert metrics.lift > 0
        # Perfect precision means lift should be high
        assert metrics.lift > 1.0

    def test_f1_score_calculation(self, labeled_df: pd.DataFrame) -> None:
        """Test F1 score calculation."""
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series(["accept", "accept", "accept", "accept", "reject",
                                 "accept", "reject", "reject", "reject", "reject"])
        metrics = evaluator.evaluate(labeled_df, predictions)

        assert metrics.f1_score == 1.0  # Perfect predictions

    def test_specificity_calculation(self, labeled_df: pd.DataFrame) -> None:
        """Test specificity (true negative rate) calculation."""
        evaluator = Evaluator(label_column="is_bad")
        predictions = pd.Series(["accept", "accept", "accept", "accept", "reject",
                                 "accept", "reject", "reject", "reject", "reject"])
        metrics = evaluator.evaluate(labeled_df, predictions)

        assert metrics.specificity == 1.0  # No false positives
