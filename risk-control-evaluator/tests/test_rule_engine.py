"""Tests for rule engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from risk_eval.core.rule_engine import ConditionExecutor, RuleEngine
from risk_eval.utils.config import ComparisonOperator, Condition, LogicalOperator, RuleConfig, RuleSetConfig
from risk_eval.utils.exceptions import RuleConfigError, RuleExecutionError


class TestConditionExecutor:
    """Tests for ConditionExecutor."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "age": [25, 35, 45, 55],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [100, 200, 300, None],
        })

    def test_numeric_gt(self, sample_df: pd.DataFrame) -> None:
        """Test greater-than condition."""
        executor = ConditionExecutor()
        condition = Condition(field="age", operator=ComparisonOperator.GT, value=30)
        result = executor.execute(sample_df, condition)
        assert result.tolist() == [False, True, True, True]

    def test_numeric_eq(self, sample_df: pd.DataFrame) -> None:
        """Test equals condition."""
        executor = ConditionExecutor()
        condition = Condition(field="age", operator=ComparisonOperator.EQ, value=35)
        result = executor.execute(sample_df, condition)
        assert result.tolist() == [False, True, False, False]

    def test_string_contains(self, sample_df: pd.DataFrame) -> None:
        """Test string contains condition."""
        executor = ConditionExecutor()
        condition = Condition(field="name", operator=ComparisonOperator.CONTAINS, value="li")
        result = executor.execute(sample_df, condition)
        assert result.tolist() == [True, False, True, False]

    def test_in_operator(self, sample_df: pd.DataFrame) -> None:
        """Test in-list condition."""
        executor = ConditionExecutor()
        condition = Condition(field="age", operator=ComparisonOperator.IN, value=[25, 55])
        result = executor.execute(sample_df, condition)
        assert result.tolist() == [True, False, False, True]

    def test_is_null(self, sample_df: pd.DataFrame) -> None:
        """Test is-null condition."""
        executor = ConditionExecutor()
        condition = Condition(field="score", operator=ComparisonOperator.IS_NULL)
        result = executor.execute(sample_df, condition)
        assert result.tolist() == [False, False, False, True]

    def test_field_not_found_raises(self, sample_df: pd.DataFrame) -> None:
        """Test that missing field raises RuleExecutionError."""
        executor = ConditionExecutor()
        condition = Condition(field="nonexistent", operator=ComparisonOperator.EQ, value=1)
        with pytest.raises(RuleExecutionError) as exc_info:
            executor.execute(sample_df, condition)
        assert "not found" in str(exc_info.value)

    def test_regex_operator(self, sample_df: pd.DataFrame) -> None:
        """Test regex condition."""
        executor = ConditionExecutor()
        condition = Condition(field="name", operator=ComparisonOperator.REGEX, value="^A.*")
        result = executor.execute(sample_df, condition)
        assert result.tolist() == [True, False, False, False]


class TestRuleEngine:
    """Tests for RuleEngine."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "age": [25, 35, 45, 55, 28, 60],
            "credit_score": [600, 700, 750, 800, 620, 580],
            "category": ["A", "B", "A", "C", "B", "A"],
        })

    def test_execute_simple_rules(self, sample_df: pd.DataFrame) -> None:
        """Test executing simple rules."""
        config = RuleSetConfig(
            name="Test Strategy",
            rules=[
                RuleConfig(
                    name="Old Age",
                    conditions=[Condition(field="age", operator=ComparisonOperator.GT, value=50)],
                    logic=LogicalOperator.AND,
                    action="reject",
                ),
            ],
            combine_logic="any",
        )
        engine = RuleEngine(config)
        result = engine.execute(sample_df)

        assert result.total_rows == len(sample_df)
        assert result.decision_distribution["reject"] == 2  # ages 55, 60
        assert result.decision_distribution["accept"] == 4

    def test_execute_and_logic(self, sample_df: pd.DataFrame) -> None:
        """Test AND logic between conditions."""
        config = RuleSetConfig(
            name="Test Strategy",
            rules=[
                RuleConfig(
                    name="High Risk",
                    conditions=[
                        Condition(field="age", operator=ComparisonOperator.GT, value=30),
                        Condition(field="credit_score", operator=ComparisonOperator.LT, value=700),
                    ],
                    logic=LogicalOperator.AND,
                    action="reject",
                ),
            ],
            combine_logic="any",
        )
        engine = RuleEngine(config)
        result = engine.execute(sample_df)

        # Only row with age>30 AND credit_score<700: age=60, score=580
        assert result.rule_summaries["High Risk"]["hit_count"] == 1

    def test_execute_or_logic(self, sample_df: pd.DataFrame) -> None:
        """Test OR logic between conditions."""
        config = RuleSetConfig(
            name="Test Strategy",
            rules=[
                RuleConfig(
                    name="Risky",
                    conditions=[
                        Condition(field="age", operator=ComparisonOperator.GT, value=50),
                        Condition(field="credit_score", operator=ComparisonOperator.LT, value=600),
                    ],
                    logic=LogicalOperator.OR,
                    action="reject",
                ),
            ],
            combine_logic="any",
        )
        engine = RuleEngine(config)
        result = engine.execute(sample_df)

        # age>50: 55, 60 | credit_score<600: 580 -> hits: 55, 60 (3 rows, 60 counted once)
        assert result.rule_summaries["Risky"]["hit_count"] == 2

    def test_disabled_rules_skipped(self, sample_df: pd.DataFrame) -> None:
        """Test that disabled rules are not executed."""
        config = RuleSetConfig(
            name="Test Strategy",
            rules=[
                RuleConfig(
                    name="Enabled",
                    conditions=[Condition(field="age", operator=ComparisonOperator.GT, value=50)],
                    logic=LogicalOperator.AND,
                    action="reject",
                    enabled=True,
                ),
                RuleConfig(
                    name="Disabled",
                    conditions=[Condition(field="age", operator=ComparisonOperator.GT, value=0)],
                    logic=LogicalOperator.AND,
                    action="reject",
                    enabled=False,
                ),
            ],
            combine_logic="any",
        )
        engine = RuleEngine(config)
        result = engine.execute(sample_df)

        assert "Enabled" in result.rule_summaries
        assert "Disabled" not in result.rule_summaries

    def test_no_enabled_rules_raises(self, sample_df: pd.DataFrame) -> None:
        """Test that no enabled rules raises error."""
        config = RuleSetConfig(
            name="Test Strategy",
            rules=[
                RuleConfig(
                    name="Disabled",
                    conditions=[Condition(field="age", operator=ComparisonOperator.GT, value=50)],
                    logic=LogicalOperator.AND,
                    action="reject",
                    enabled=False,
                ),
            ],
            combine_logic="any",
        )
        with pytest.raises(RuleConfigError) as exc_info:
            RuleEngine(config)
        assert "No enabled rules" in str(exc_info.value)

    def test_from_yaml(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        """Test loading rule engine from YAML file."""
        config = RuleSetConfig(
            name="YAML Strategy",
            rules=[
                RuleConfig(
                    name="Age Check",
                    conditions=[Condition(field="age", operator=ComparisonOperator.GT, value=50)],
                    logic=LogicalOperator.AND,
                    action="reject",
                ),
            ],
            combine_logic="any",
        )
        yaml_path = tmp_path / "rules.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(mode="json"), f)

        engine = RuleEngine.from_yaml(str(yaml_path))
        assert engine.config.name == "YAML Strategy"

    def test_from_yaml_invalid_raises(self, tmp_path: Path) -> None:
        """Test loading invalid YAML raises error."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("not: valid: yaml: [", encoding="utf-8")

        with pytest.raises(RuleConfigError) as exc_info:
            RuleEngine.from_yaml(str(yaml_path))
        assert "Invalid YAML" in str(exc_info.value)

    def test_explain(self, sample_df: pd.DataFrame) -> None:
        """Test rule explanation for a single row."""
        config = RuleSetConfig(
            name="Test Strategy",
            rules=[
                RuleConfig(
                    name="Age Check",
                    conditions=[Condition(field="age", operator=ComparisonOperator.GT, value=50)],
                    logic=LogicalOperator.AND,
                    action="reject",
                ),
            ],
            combine_logic="any",
        )
        engine = RuleEngine(config)
        explanations = engine.explain(sample_df, row_index=3)  # age=55

        assert len(explanations) == 1
        assert explanations[0]["rule_name"] == "Age Check"
        assert explanations[0]["rule_hit"] is True

    def test_score_based_strategy(self, sample_df: pd.DataFrame) -> None:
        """Test score-threshold combine logic."""
        config = RuleSetConfig(
            name="Scoring Strategy",
            rules=[
                RuleConfig(
                    name="High Age",
                    conditions=[Condition(field="age", operator=ComparisonOperator.GT, value=40)],
                    logic=LogicalOperator.AND,
                    action="reject",
                    score=30,
                ),
                RuleConfig(
                    name="Low Score",
                    conditions=[Condition(field="credit_score", operator=ComparisonOperator.LT, value=650)],
                    logic=LogicalOperator.AND,
                    action="reject",
                    score=40,
                ),
            ],
            combine_logic="score_threshold",
            score_threshold=50,
        )
        engine = RuleEngine(config)
        result = engine.execute(sample_df)

        # Rows with total score >= 50 should be rejected
        assert "reject" in result.decision_distribution
