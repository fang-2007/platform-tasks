"""Configuration models using Pydantic."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComparisonOperator(str, Enum):
    """Supported comparison operators for rule conditions."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class LogicalOperator(str, Enum):
    """Logical operators for combining conditions."""

    AND = "and"
    OR = "or"


class Condition(BaseModel):
    """A single rule condition."""

    field: str = Field(..., description="Column name to evaluate")
    operator: ComparisonOperator = Field(..., description="Comparison operator")
    value: Any = Field(None, description="Value to compare against", validate_default=True)

    @field_validator("value")
    @classmethod
    def validate_value_for_operator(cls, v: Any, info: Any) -> Any:
        """Validate that value is provided for operators that require it."""
        operator = info.data.get("operator") if hasattr(info, "data") else None
        if operator is None:
            return v
        null_operators = {ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL}
        if operator not in null_operators and v is None:
            raise ValueError(f"Operator '{operator}' requires a value")
        return v


class RuleConfig(BaseModel):
    """Configuration for a single rule."""

    name: str = Field(..., description="Human-readable rule name")
    description: str = Field("", description="Rule description")
    conditions: list[Condition] = Field(..., description="List of conditions")
    logic: LogicalOperator = Field(LogicalOperator.AND, description="How to combine conditions")
    action: Literal["accept", "reject", "review"] = Field("reject", description="Action when rule hits")
    score: int = Field(0, description="Score contribution (for scoring strategies)")
    enabled: bool = Field(True, description="Whether rule is active")


class RuleSetConfig(BaseModel):
    """Configuration for a set of rules (a strategy)."""

    name: str = Field(..., description="Strategy name")
    description: str = Field("", description="Strategy description")
    version: str = Field("1.0.0", description="Semantic version")
    rules: list[RuleConfig] = Field(..., description="Rules in this strategy")
    combine_logic: Literal["any", "all", "score_threshold"] = Field(
        "any", description="How rules combine into final decision"
    )
    score_threshold: int | None = Field(
        None, description="Threshold for score-based strategy", validate_default=True
    )

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, v: int | None, info: Any) -> int | None:
        """Ensure score_threshold is set when combine_logic is score_threshold."""
        combine = info.data.get("combine_logic") if hasattr(info, "data") else None
        if combine == "score_threshold" and v is None:
            raise ValueError("score_threshold is required when combine_logic is 'score_threshold'")
        return v


class AppConfig(BaseModel):
    """Application-level configuration."""

    log_level: str = Field("INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    json_logs: bool = Field(False, description="Output logs in JSON format")
    max_rows_display: int = Field(10000, description="Max rows to display in reports")
    default_label_column: str = Field("is_bad", description="Default target column name")

    model_config = ConfigDict(env_prefix="RISK_EVAL_")
