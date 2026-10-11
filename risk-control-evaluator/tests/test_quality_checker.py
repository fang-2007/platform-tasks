"""Tests for data quality checker."""

from __future__ import annotations

import pandas as pd
import pytest

from risk_eval.core.quality_checker import QualityChecker, QualityReport


class TestQualityChecker:
    """Tests for QualityChecker."""

    def test_check_basic_dataset(self, sample_dataframe: pd.DataFrame) -> None:
        """Test quality check on clean dataset."""
        checker = QualityChecker()
        report = checker.check(sample_dataframe, dataset_name="test")

        assert isinstance(report, QualityReport)
        assert report.total_rows == len(sample_dataframe)
        assert report.total_columns == len(sample_dataframe.columns)
        assert report.dataset_name == "test"

    def test_check_with_label(self, sample_dataframe: pd.DataFrame) -> None:
        """Test quality check with label column specified."""
        checker = QualityChecker()
        report = checker.check(sample_dataframe, label_column="is_bad")

        assert report.label_summary is not None
        assert report.label_summary.column == "is_bad"
        assert report.label_summary.total_rows == len(sample_dataframe)

    def test_check_with_issues(self, sample_dataframe_with_issues: pd.DataFrame) -> None:
        """Test quality check detects issues."""
        checker = QualityChecker(null_threshold=0.3, imbalance_ratio_threshold=3.0)
        report = checker.check(
            sample_dataframe_with_issues,
            label_column="label",
            primary_key="id",
        )

        assert report.has_critical_issues(null_threshold=0.3)
        assert report.duplicates is not None
        assert report.duplicates.duplicate_rows == 1  # "A" is duplicated
        assert report.label_summary is not None
        assert report.label_summary.is_imbalanced

    def test_missing_label_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test check with missing label column gracefully handles."""
        checker = QualityChecker()
        report = checker.check(sample_dataframe, label_column="nonexistent")

        assert report.label_summary is None

    def test_consistency_checks(self, sample_dataframe: pd.DataFrame) -> None:
        """Test consistency check functionality."""
        checker = QualityChecker()
        report = checker.check(
            sample_dataframe,
            consistency_checks=[
                {
                    "name": "income_positive",
                    "type": "range",
                    "column": "income",
                    "min": 0,
                }
            ],
        )

        assert len(report.consistency_issues) == 0  # All incomes are positive

    def test_temporal_order_check(self) -> None:
        """Test temporal order consistency check."""
        df = pd.DataFrame({
            "apply_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
            "approve_date": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-06"]),
        })
        checker = QualityChecker()
        report = checker.check(
            df,
            consistency_checks=[
                {
                    "name": "approve_after_apply",
                    "type": "temporal_order",
                    "earlier_column": "apply_date",
                    "later_column": "approve_date",
                }
            ],
        )

        assert len(report.consistency_issues) == 1
        assert report.consistency_issues[0].affected_rows == 1

    def test_column_quality_metrics(self, sample_dataframe: pd.DataFrame) -> None:
        """Test column quality metrics are computed correctly."""
        checker = QualityChecker()
        report = checker.check(sample_dataframe)

        age_col = next(c for c in report.columns if c.column == "age")
        assert age_col.null_count == 0
        assert age_col.null_rate == 0.0
        assert age_col.is_const is False

    def test_duplicate_detection_full_row(self, sample_dataframe: pd.DataFrame) -> None:
        """Test duplicate detection without primary key."""
        df_with_dup = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)
        checker = QualityChecker()
        report = checker.check(df_with_dup)

        assert report.duplicates is not None
        assert report.duplicates.duplicate_rows == 1

    def test_warning_generation(self, sample_dataframe_with_issues: pd.DataFrame) -> None:
        """Test that warnings are generated for issues."""
        checker = QualityChecker(null_threshold=0.1, imbalance_ratio_threshold=3.0)
        report = checker.check(
            sample_dataframe_with_issues,
            label_column="label",
            primary_key="id",
        )

        assert len(report.warnings) > 0
        assert any("null" in w.lower() for w in report.warnings)
        assert any("duplicate" in w.lower() for w in report.warnings)
        assert any("imbalanced" in w.lower() for w in report.warnings)
