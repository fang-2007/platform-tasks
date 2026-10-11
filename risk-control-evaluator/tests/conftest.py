"""Pytest fixtures and utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from risk_eval.utils.config import ComparisonOperator, Condition, LogicalOperator, RuleConfig, RuleSetConfig


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample DataFrame for testing."""
    return pd.DataFrame({
        "user_id": ["U001", "U002", "U003", "U004", "U005", "U006", "U007", "U008", "U009", "U010"],
        "age": [25, 35, 45, 55, 28, 42, 33, 29, 51, 38],
        "income": [5000, 8000, 12000, 15000, 6000, 10000, 7500, 5500, 11000, 9000],
        "credit_score": [600, 700, 750, 800, 650, 720, 680, 620, 780, 710],
        "has_default": [1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        "loan_amount": [10000, 20000, 30000, 40000, 15000, 25000, 18000, 12000, 35000, 22000],
        "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "is_bad": [1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    })


@pytest.fixture
def sample_dataframe_with_issues() -> pd.DataFrame:
    """Return a DataFrame with quality issues for testing."""
    return pd.DataFrame({
        "id": ["A", "A", "B", "C", "D"],
        "value": [1.0, 2.0, None, 4.0, 5.0],
        "label": [1, 1, 1, 1, 0],
        "constant": ["X", "X", "X", "X", "X"],
    })


@pytest.fixture
def temp_csv_file(sample_dataframe: pd.DataFrame) -> Path:
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = Path(f.name)
    sample_dataframe.to_csv(path, index=False)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_excel_file(sample_dataframe: pd.DataFrame) -> Path:
    """Create a temporary Excel file."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = Path(f.name)
    sample_dataframe.to_excel(path, index=False)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def simple_rule_config() -> RuleSetConfig:
    """Return a simple rule set configuration."""
    return RuleSetConfig(
        name="Simple Strategy",
        description="Test strategy",
        rules=[
            RuleConfig(
                name="High Risk Age",
                description="Age > 50",
                conditions=[
                    Condition(field="age", operator=ComparisonOperator.GT, value=50),
                ],
                logic=LogicalOperator.AND,
                action="reject",
            ),
            RuleConfig(
                name="Low Credit Score",
                description="Credit score < 650",
                conditions=[
                    Condition(field="credit_score", operator=ComparisonOperator.LT, value=650),
                ],
                logic=LogicalOperator.AND,
                action="reject",
            ),
        ],
        combine_logic="any",
    )


@pytest.fixture
def yaml_rule_file(simple_rule_config: RuleSetConfig) -> Path:
    """Create a temporary YAML rule file."""
    import yaml

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        path = Path(f.name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(simple_rule_config.model_dump(mode="json"), f)
    yield path
    path.unlink(missing_ok=True)
