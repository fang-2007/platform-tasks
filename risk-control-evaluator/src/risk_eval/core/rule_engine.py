"""Rule engine for executing risk control strategies."""

from __future__ import annotations

import re
from typing import Any, Protocol

import pandas as pd
import yaml
from pydantic import BaseModel

from risk_eval.utils.config import ComparisonOperator, Condition, LogicalOperator, RuleSetConfig
from risk_eval.utils.exceptions import RuleConfigError, RuleExecutionError
from risk_eval.utils.logger import get_logger

logger = get_logger(__name__)


class RuleEngineProtocol(Protocol):
    """Protocol for rule engine implementations."""

    def execute(self, df: pd.DataFrame) -> RuleExecutionSummary:
        """Execute rules against data."""
        ...


class RuleExecutionSummary:
    """Summary of rule execution across dataset."""

    def __init__(
        self,
        total_rows: int,
        rule_summaries: dict[str, dict[str, Any]],
        decision_distribution: dict[str, int],
        final_decisions: pd.Series,
        rule_hit_matrix: pd.DataFrame,
    ) -> None:
        self.total_rows = total_rows
        self.rule_summaries = rule_summaries
        self.decision_distribution = decision_distribution
        self.final_decisions = final_decisions
        self.rule_hit_matrix = rule_hit_matrix


class ConditionExecutor:
    """Executes a single condition against a DataFrame column."""

    _operator_map: dict[ComparisonOperator, str] = {
        ComparisonOperator.EQ: "eq",
        ComparisonOperator.NE: "ne",
        ComparisonOperator.GT: "gt",
        ComparisonOperator.GTE: "ge",
        ComparisonOperator.LT: "lt",
        ComparisonOperator.LTE: "le",
    }

    def execute(self, df: pd.DataFrame, condition: Condition) -> pd.Series:
        """Evaluate a condition against the DataFrame.

        Args:
            df: Input data.
            condition: Condition to evaluate.

        Returns:
            Boolean Series indicating which rows satisfy the condition.

        Raises:
            RuleExecutionError: If field not found or evaluation fails.
        """
        field = condition.field
        if field not in df.columns:
            raise RuleExecutionError(
                f"Field '{field}' not found in data",
                context={"available_columns": list(df.columns)},
            )

        series = df[field]
        op = condition.operator
        value = condition.value

        try:
            return self._evaluate(series, op, value)
        except Exception as exc:
            raise RuleExecutionError(
                f"Failed to evaluate condition on '{field}': {exc}",
                context={"operator": op, "value": value},
            ) from exc

    def _evaluate(
        self, series: pd.Series, op: ComparisonOperator, value: Any
    ) -> pd.Series:
        """Internal evaluation dispatcher."""
        if op in self._operator_map:
            method = getattr(series, self._operator_map[op])
            return method(value)

        if op == ComparisonOperator.IN:
            return series.isin(value if isinstance(value, (list, tuple, set)) else [value])
        if op == ComparisonOperator.NOT_IN:
            return ~series.isin(value if isinstance(value, (list, tuple, set)) else [value])

        if op == ComparisonOperator.CONTAINS:
            return series.astype(str).str.contains(str(value), na=False)
        if op == ComparisonOperator.STARTS_WITH:
            return series.astype(str).str.startswith(str(value), na=False)
        if op == ComparisonOperator.ENDS_WITH:
            return series.astype(str).str.endswith(str(value), na=False)
        if op == ComparisonOperator.REGEX:
            pattern = re.compile(str(value))
            return series.astype(str).apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)

        if op == ComparisonOperator.IS_NULL:
            return series.isna()
        if op == ComparisonOperator.IS_NOT_NULL:
            return series.notna()

        raise RuleExecutionError(f"Unsupported operator: {op}")


class RuleEngine:
    """Engine for executing rule sets against data."""

    def __init__(self, config: RuleSetConfig) -> None:
        """Initialize with rule set configuration.

        Args:
            config: Parsed rule set configuration.
        """
        self.config = config
        self.condition_executor = ConditionExecutor()
        self._validate_config()

    @classmethod
    def from_yaml(cls, path: str) -> "RuleEngine":
        """Load rule engine from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Configured RuleEngine instance.

        Raises:
            RuleConfigError: If configuration is invalid.
        """
        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            config = RuleSetConfig.model_validate(raw)
            logger.info("Loaded rule config", path=path, rules=len(config.rules))
            return cls(config)
        except yaml.YAMLError as exc:
            raise RuleConfigError(f"Invalid YAML in {path}: {exc}") from exc
        except Exception as exc:
            raise RuleConfigError(f"Failed to load rule config from {path}: {exc}") from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleEngine":
        """Load rule engine from dictionary.

        Args:
            data: Dictionary containing rule set configuration.

        Returns:
            Configured RuleEngine instance.
        """
        config = RuleSetConfig.model_validate(data)
        return cls(config)

    def _validate_config(self) -> None:
        """Validate that rule configuration is executable."""
        enabled_rules = [r for r in self.config.rules if r.enabled]
        if not enabled_rules:
            raise RuleConfigError("No enabled rules in configuration")

        if self.config.combine_logic == "score_threshold":
            if self.config.score_threshold is None:
                raise RuleConfigError("score_threshold required for score-based strategy")
            has_scores = any(r.score != 0 for r in enabled_rules)
            if not has_scores:
                logger.warning("Score-based strategy but no rules have non-zero scores")

    def execute(self, df: pd.DataFrame) -> RuleExecutionSummary:
        """Execute all rules against the data.

        Args:
            df: Input data to evaluate.

        Returns:
            Summary of rule execution including hit matrix and decisions.

        Raises:
            RuleExecutionError: If execution fails.
        """
        logger.info("Executing rules", rows=len(df), strategy=self.config.name)

        enabled_rules = [r for r in self.config.rules if r.enabled]
        hit_matrix = pd.DataFrame(index=df.index)
        decisions = pd.Series("accept", index=df.index)
        scores = pd.Series(0, index=df.index, dtype=int)

        rule_summaries: dict[str, dict[str, Any]] = {}

        for rule in enabled_rules:
            hits = self._evaluate_rule(df, rule)
            hit_matrix[rule.name] = hits

            hit_count = int(hits.sum())
            hit_rate = hit_count / len(df) if len(df) > 0 else 0.0

            rule_summaries[rule.name] = {
                "description": rule.description,
                "hit_count": hit_count,
                "hit_rate": round(hit_rate, 4),
                "action": rule.action,
                "score": rule.score,
            }

            # Update decisions based on rule hits
            if rule.action == "reject":
                decisions = decisions.where(~hits, "reject")
            elif rule.action == "review":
                decisions = decisions.where(~hits | (decisions == "reject"), "review")

            # Accumulate scores
            if rule.score != 0:
                scores += hits.astype(int) * rule.score

        # Apply combine logic for final decisions
        if self.config.combine_logic == "all":
            # All rules must hit for rejection
            all_hits = hit_matrix.all(axis=1)
            decisions = pd.Series("accept", index=df.index)
            decisions[all_hits] = "reject"
        elif self.config.combine_logic == "score_threshold":
            threshold = self.config.score_threshold or 0
            decisions = pd.Series("accept", index=df.index)
            decisions[scores >= threshold] = "reject"

        decision_distribution = decisions.value_counts().to_dict()
        decision_distribution = {str(k): int(v) for k, v in decision_distribution.items()}

        logger.info(
            "Rule execution complete",
            decisions=decision_distribution,
            rules_evaluated=len(enabled_rules),
        )

        return RuleExecutionSummary(
            total_rows=len(df),
            rule_summaries=rule_summaries,
            decision_distribution=decision_distribution,
            final_decisions=decisions,
            rule_hit_matrix=hit_matrix,
        )

    def _evaluate_rule(self, df: pd.DataFrame, rule: Any) -> pd.Series:
        """Evaluate a single rule's conditions.

        Args:
            df: Input data.
            rule: Rule configuration.

        Returns:
            Boolean Series indicating rule hits.
        """
        if not rule.conditions:
            return pd.Series(False, index=df.index)

        results: list[pd.Series] = []
        for condition in rule.conditions:
            result = self.condition_executor.execute(df, condition)
            results.append(result)

        if rule.logic == LogicalOperator.AND:
            combined = results[0]
            for r in results[1:]:
                combined = combined & r
            return combined
        else:  # OR
            combined = results[0]
            for r in results[1:]:
                combined = combined | r
            return combined

    def explain(self, df: pd.DataFrame, row_index: int) -> list[dict[str, Any]]:
        """Explain which rules hit for a specific row.

        Args:
            df: Input data.
            row_index: Index of row to explain.

        Returns:
            List of rule explanations with condition results.
        """
        if row_index not in df.index:
            raise RuleExecutionError(f"Row index {row_index} not found")

        row = df.loc[[row_index]]
        explanations = []

        for rule in self.config.rules:
            if not rule.enabled:
                continue

            condition_results = []
            condition_hits: list[bool] = []

            for condition in rule.conditions:
                try:
                    hit = self.condition_executor.execute(row, condition).iloc[0]
                except Exception as exc:
                    hit = False
                condition_results.append({
                    "field": condition.field,
                    "operator": condition.operator,
                    "value": condition.value,
                    "hit": bool(hit),
                })
                condition_hits.append(bool(hit))

            rule_hit = bool(condition_hits) and (
                all(condition_hits) if rule.logic == LogicalOperator.AND else any(condition_hits)
            )

            explanations.append({
                "rule_name": rule.name,
                "rule_hit": rule_hit,
                "action": rule.action,
                "conditions": condition_results,
            })

        return explanations
