"""Data quality checking with comprehensive validation rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from risk_eval.utils.exceptions import DataValidationError
from risk_eval.utils.logger import get_logger

logger = get_logger(__name__)


class ColumnQuality(BaseModel):
    """Quality metrics for a single column."""

    column: str
    dtype: str
    total_rows: int
    null_count: int
    null_rate: float = Field(..., ge=0.0, le=1.0)
    unique_count: int
    unique_rate: float = Field(..., ge=0.0, le=1.0)
    is_const: bool
    outlier_count: int | None = None
    outlier_rate: float | None = Field(None, ge=0.0, le=1.0)
    low_freq_categories: list[dict[str, Any]] | None = None


class DuplicateSummary(BaseModel):
    """Summary of duplicate records."""

    total_rows: int
    duplicate_rows: int
    duplicate_rate: float = Field(..., ge=0.0, le=1.0)
    checked_columns: list[str]


class LabelSummary(BaseModel):
    """Summary of label/target column analysis."""

    column: str
    total_rows: int
    null_count: int
    class_distribution: dict[Any, int]
    class_rates: dict[str, float]
    is_imbalanced: bool
    imbalance_ratio: float | None = None


class ConsistencyIssue(BaseModel):
    """A single consistency issue found."""

    check_name: str
    affected_rows: int
    description: str
    sample_indices: list[int]


class QualityReport(BaseModel):
    """Complete data quality report."""

    dataset_name: str = ""
    total_rows: int
    total_columns: int
    columns: list[ColumnQuality]
    duplicates: DuplicateSummary | None = None
    label_summary: LabelSummary | None = None
    consistency_issues: list[ConsistencyIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def has_critical_issues(self, null_threshold: float = 0.5) -> bool:
        """Check if report contains critical quality issues.

        Args:
            null_threshold: Null rate threshold to consider critical.

        Returns:
            True if critical issues exist.
        """
        for col in self.columns:
            if col.null_rate > null_threshold:
                return True
        if self.duplicates and self.duplicates.duplicate_rate > 0.1:
            return True
        if self.label_summary and self.label_summary.null_count > 0:
            return True
        return bool(self.consistency_issues)


class QualityChecker:
    """Performs comprehensive data quality checks."""

    def __init__(
        self,
        null_threshold: float = 0.5,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        imbalance_ratio_threshold: float = 10.0,
        low_freq_threshold: int = 5,
    ) -> None:
        """Initialize quality checker with thresholds.

        Args:
            null_threshold: Null rate above which a column is flagged critical.
            outlier_method: Method for outlier detection ('iqr' or 'zscore').
            outlier_threshold: Threshold for outlier detection.
            imbalance_ratio_threshold: Ratio threshold for class imbalance flag.
            low_freq_threshold: Minimum frequency for a category to not be flagged.
        """
        self.null_threshold = null_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.low_freq_threshold = low_freq_threshold

    def check(
        self,
        df: pd.DataFrame,
        label_column: str | None = None,
        primary_key: str | list[str] | None = None,
        consistency_checks: list[dict[str, Any]] | None = None,
        dataset_name: str = "",
    ) -> QualityReport:
        """Run all quality checks on a DataFrame.

        Args:
            df: Input data.
            label_column: Name of the target/label column.
            primary_key: Column(s) to use for duplicate detection.
            consistency_checks: List of consistency check definitions.
            dataset_name: Optional name for the dataset.

        Returns:
            Complete quality report.
        """
        logger.info(
            "Starting quality check",
            rows=len(df),
            columns=len(df.columns),
            dataset=dataset_name,
        )

        columns = self._check_columns(df)
        duplicates = self._check_duplicates(df, primary_key)
        label_summary = self._check_label(df, label_column) if label_column else None
        consistency_issues = self._check_consistency(df, consistency_checks or [])

        warnings = self._generate_warnings(columns, duplicates, label_summary)

        report = QualityReport(
            dataset_name=dataset_name,
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=columns,
            duplicates=duplicates,
            label_summary=label_summary,
            consistency_issues=consistency_issues,
            warnings=warnings,
        )

        logger.info(
            "Quality check complete",
            warnings=len(warnings),
            critical=report.has_critical_issues(),
        )
        return report

    def _check_columns(self, df: pd.DataFrame) -> list[ColumnQuality]:
        """Check each column for quality issues."""
        results: list[ColumnQuality] = []
        total = len(df)

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_rate = null_count / total if total > 0 else 0.0
            unique_count = int(series.nunique(dropna=False))
            unique_rate = unique_count / total if total > 0 else 0.0

            outlier_count: int | None = None
            outlier_rate: float | None = None
            low_freq_categories: list[dict[str, Any]] | None = None

            # Numeric outlier detection
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                outlier_count, outlier_rate = self._detect_outliers(series)

            # Categorical low-frequency detection
            if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
                low_freq_categories = self._detect_low_freq_categories(series)

            results.append(
                ColumnQuality(
                    column=str(col),
                    dtype=str(series.dtype),
                    total_rows=total,
                    null_count=null_count,
                    null_rate=null_rate,
                    unique_count=unique_count,
                    unique_rate=unique_rate,
                    is_const=unique_count <= 1,
                    outlier_count=outlier_count,
                    outlier_rate=outlier_rate,
                    low_freq_categories=low_freq_categories,
                )
            )

        return results

    def _detect_outliers(self, series: pd.Series) -> tuple[int, float]:
        """Detect outliers in a numeric series."""
        clean = series.dropna()
        if len(clean) == 0:
            return 0, 0.0

        if self.outlier_method == "iqr":
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.outlier_threshold * iqr
            upper = q3 + self.outlier_threshold * iqr
            mask = (series < lower) | (series > upper)
        elif self.outlier_method == "zscore":
            mean = clean.mean()
            std = clean.std()
            if std == 0:
                return 0, 0.0
            z_scores = np.abs((series - mean) / std)
            mask = z_scores > self.outlier_threshold
        else:
            return 0, 0.0

        count = int(mask.sum())
        rate = count / len(series)
        return count, rate

    def _detect_low_freq_categories(self, series: pd.Series) -> list[dict[str, Any]] | None:
        """Detect categories with very low frequency."""
        value_counts = series.value_counts(dropna=False)
        low_freq = value_counts[value_counts < self.low_freq_threshold]
        if low_freq.empty:
            return None
        return [
            {"value": str(val), "count": int(cnt)} for val, cnt in low_freq.items()
        ]

    def _check_duplicates(
        self,
        df: pd.DataFrame,
        primary_key: str | list[str] | None,
    ) -> DuplicateSummary | None:
        """Check for duplicate records."""
        if primary_key is None:
            # Check full row duplicates
            dup_mask = df.duplicated()
            checked = list(df.columns)
        else:
            keys = [primary_key] if isinstance(primary_key, str) else primary_key
            missing = [k for k in keys if k not in df.columns]
            if missing:
                logger.warning("Primary key columns not found", missing=missing)
                return None
            dup_mask = df.duplicated(subset=keys)
            checked = keys

        dup_count = int(dup_mask.sum())
        total = len(df)
        rate = dup_count / total if total > 0 else 0.0

        return DuplicateSummary(
            total_rows=total,
            duplicate_rows=dup_count,
            duplicate_rate=rate,
            checked_columns=checked,
        )

    def _check_label(self, df: pd.DataFrame, label_column: str) -> LabelSummary | None:
        """Analyze label column for issues."""
        if label_column not in df.columns:
            logger.warning("Label column not found", column=label_column)
            return None

        series = df[label_column]
        total = len(series)
        null_count = int(series.isna().sum())
        value_counts = series.value_counts(dropna=False)

        class_distribution = {}
        for val, cnt in value_counts.items():
            key = "NULL" if pd.isna(val) else str(val)
            class_distribution[key] = int(cnt)

        class_rates = {
            str(k): round(v / total, 4) for k, v in class_distribution.items()
        }

        # Check imbalance
        is_imbalanced = False
        imbalance_ratio = None
        if len(value_counts) >= 2:
            major = value_counts.iloc[0]
            minor = value_counts.iloc[-1]
            imbalance_ratio = float(major / minor) if minor > 0 else float("inf")
            is_imbalanced = imbalance_ratio > self.imbalance_ratio_threshold

        return LabelSummary(
            column=label_column,
            total_rows=total,
            null_count=null_count,
            class_distribution=class_distribution,
            class_rates=class_rates,
            is_imbalanced=is_imbalanced,
            imbalance_ratio=imbalance_ratio,
        )

    def _check_consistency(
        self,
        df: pd.DataFrame,
        checks: list[dict[str, Any]],
    ) -> list[ConsistencyIssue]:
        """Run consistency checks defined by the user."""
        issues: list[ConsistencyIssue] = []

        for check_def in checks:
            name = check_def.get("name", "unnamed_check")
            check_type = check_def.get("type")

            try:
                if check_type == "temporal_order":
                    issue = self._check_temporal_order(df, check_def)
                elif check_type == "range":
                    issue = self._check_range(df, check_def)
                elif check_type == "dependency":
                    issue = self._check_dependency(df, check_def)
                else:
                    logger.warning("Unknown consistency check type", type=check_type)
                    continue

                if issue and issue.affected_rows > 0:
                    issues.append(issue)
            except Exception as exc:
                logger.warning(f"Consistency check '{name}' failed", error=str(exc))

        return issues

    def _check_temporal_order(
        self, df: pd.DataFrame, check_def: dict[str, Any]
    ) -> ConsistencyIssue | None:
        """Check that one date column is before another."""
        earlier_col = check_def["earlier_column"]
        later_col = check_def["later_column"]

        if earlier_col not in df.columns or later_col not in df.columns:
            return None

        earlier = pd.to_datetime(df[earlier_col], errors="coerce")
        later = pd.to_datetime(df[later_col], errors="coerce")

        invalid = (earlier > later) & earlier.notna() & later.notna()
        affected = int(invalid.sum())
        sample = df.index[invalid].tolist()[:5]

        return ConsistencyIssue(
            check_name=check_def.get("name", f"{earlier_col} <= {later_col}"),
            affected_rows=affected,
            description=f"Found {affected} rows where '{earlier_col}' > '{later_col}'",
            sample_indices=sample,
        )

    def _check_range(
        self, df: pd.DataFrame, check_def: dict[str, Any]
    ) -> ConsistencyIssue | None:
        """Check that values fall within an expected range."""
        col = check_def["column"]
        if col not in df.columns:
            return None

        min_val = check_def.get("min")
        max_val = check_def.get("max")
        series = df[col]

        invalid = pd.Series(False, index=df.index)
        if min_val is not None:
            invalid = invalid | (series < min_val)
        if max_val is not None:
            invalid = invalid | (series > max_val)

        affected = int(invalid.sum())
        sample = df.index[invalid].tolist()[:5]

        return ConsistencyIssue(
            check_name=check_def.get("name", f"{col}_range_check"),
            affected_rows=affected,
            description=f"Found {affected} rows outside expected range for '{col}'",
            sample_indices=sample,
        )

    def _check_dependency(
        self, df: pd.DataFrame, check_def: dict[str, Any]
    ) -> ConsistencyIssue | None:
        """Check conditional dependency between columns."""
        if_col = check_def["if_column"]
        if_value = check_def["if_value"]
        then_col = check_def["then_column"]
        then_expected = check_def.get("then_not_null", True)

        if if_col not in df.columns or then_col not in df.columns:
            return None

        condition = df[if_col] == if_value
        if then_expected:
            invalid = condition & df[then_col].isna()
        else:
            invalid = condition & df[then_col].notna()

        affected = int(invalid.sum())
        sample = df.index[invalid].tolist()[:5]

        return ConsistencyIssue(
            check_name=check_def.get("name", f"{if_col}_implies_{then_col}"),
            affected_rows=affected,
            description=f"Found {affected} rows violating dependency '{if_col}={if_value} -> {then_col}'",
            sample_indices=sample,
        )

    def _generate_warnings(
        self,
        columns: list[ColumnQuality],
        duplicates: DuplicateSummary | None,
        label_summary: LabelSummary | None,
    ) -> list[str]:
        """Generate human-readable warning messages."""
        warnings: list[str] = []

        for col in columns:
            if col.null_rate > self.null_threshold:
                warnings.append(
                    f"Column '{col.column}' has high null rate: {col.null_rate:.1%}"
                )
            if col.is_const:
                warnings.append(f"Column '{col.column}' is constant (no variance)")
            if col.outlier_rate and col.outlier_rate > 0.05:
                warnings.append(
                    f"Column '{col.column}' has {col.outlier_rate:.1%} outliers"
                )

        if duplicates and duplicates.duplicate_rate > 0:
            warnings.append(
                f"Found {duplicates.duplicate_rows} duplicate rows "
                f"({duplicates.duplicate_rate:.1%})"
            )

        if label_summary:
            if label_summary.null_count > 0:
                warnings.append(
                    f"Label column '{label_summary.column}' has "
                    f"{label_summary.null_count} null values"
                )
            if label_summary.is_imbalanced:
                warnings.append(
                    f"Label is imbalanced (ratio {label_summary.imbalance_ratio:.1f}:1)"
                )

        return warnings
