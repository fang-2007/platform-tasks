"""Behavioral checks for configuration, explanations, and zero-valued metrics."""

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from risk_eval.cli.main import app
from risk_eval.core.evaluator import Evaluator
from risk_eval.core.rule_engine import RuleEngine
from risk_eval.report.html_report import HTMLReportGenerator
from risk_eval.utils.config import Condition, RuleSetConfig


@pytest.mark.parametrize("operator", ["gt", "eq", "in", "contains", "regex"])
def test_comparison_requires_value_when_omitted(operator: str) -> None:
    with pytest.raises(ValidationError):
        Condition(field="age", operator=operator)


@pytest.mark.parametrize("operator", ["is_null", "is_not_null"])
def test_null_conditions_allow_omitted_value(operator: str) -> None:
    engine = RuleEngine.from_dict({
        "name": "null checks",
        "rules": [{"name": "missing", "conditions": [{"field": "age", "operator": operator}]}],
    })
    decisions = engine.execute(pd.DataFrame({"age": [None, 25]})).final_decisions.tolist()
    expected = ["reject", "accept"] if operator == "is_null" else ["accept", "reject"]
    assert decisions == expected


def test_score_threshold_required_when_omitted() -> None:
    with pytest.raises(ValidationError):
        RuleSetConfig(name="missing threshold", rules=[], combine_logic="score_threshold")


@pytest.mark.parametrize("logic", ["or", "and"])
@pytest.mark.parametrize("age,income,or_hit,and_hit", [
    (25, 5000, False, False),
    (55, 5000, True, False),
    (25, 60000, True, False),
    (55, 60000, True, True),
])
def test_explanation_matches_conditions(logic: str, age: int, income: int, or_hit: bool, and_hit: bool) -> None:
    engine = RuleEngine.from_dict({
        "name": "explanation check",
        "rules": [{"name": "risk", "logic": logic, "conditions": [
            {"field": "age", "operator": "gt", "value": 50},
            {"field": "income", "operator": "gt", "value": 50000},
        ]}],
    })
    # A non-default index also checks that explain selects a record by its label.
    df = pd.DataFrame({"age": [age], "income": [income]}, index=[17])
    expected = or_hit if logic == "or" else and_hit
    explanation = engine.explain(df, 17)[0]
    assert explanation["rule_hit"] is expected
    assert len(explanation["conditions"]) == 2
    assert engine.execute(df).final_decisions.iloc[0] == ("reject" if expected else "accept")


def test_disjoint_predictions_have_zero_f1_and_lift() -> None:
    metrics = Evaluator().evaluate(pd.DataFrame({"is_bad": [1, 0]}), pd.Series(["accept", "reject"]))
    assert metrics.f1_score == 0.0
    assert metrics.lift == 0.0


def test_cli_displays_zero_metrics_as_numbers(tmp_path: Path) -> None:
    data = tmp_path / "labeled.csv"
    rules = tmp_path / "rules.yaml"
    pd.DataFrame({"age": [25, 60], "is_bad": [1, 0]}).to_csv(data, index=False)
    rules.write_text('name: test\nrules:\n  - name: age\n    conditions:\n      - field: age\n        operator: gt\n        value: 50\n', encoding="utf-8")
    result = CliRunner().invoke(app, ["evaluate", str(data), "--rules", str(rules)])
    assert result.exit_code == 0, result.output
    recall_line = next(line for line in result.output.splitlines() if "Recall" in line)
    f1_line = next(line for line in result.output.splitlines() if "F1 Score" in line)
    assert "0.0%" in recall_line
    assert "0.000" in f1_line


def test_html_report_displays_zero_metrics(tmp_path: Path) -> None:
    metrics = Evaluator().evaluate(pd.DataFrame({"is_bad": [1, 0]}), pd.Series(["accept", "reject"]))
    report = HTMLReportGenerator().generate_evaluation_report(tmp_path / "report.html", metrics)
    html = report.read_text(encoding="utf-8")
    assert "N/A%" not in html
    assert '>0.0%</div>' in html
    assert '>0.00</div>' in html  # A zero Lift remains a defined numeric metric.
