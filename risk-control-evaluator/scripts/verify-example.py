"""Run the documented CLI workflow and check hand-reviewed sample expectations."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from risk_eval.core.evaluator import Evaluator
from risk_eval.core.rule_engine import RuleEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation-output"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    dataset = root / "data/sample_credit_data.csv"
    rule_a = root / "configs/example_rules.yaml"
    rule_b = root / "configs/example_rules_b.yaml"
    commands = [
        ["quality-check", str(dataset), "--label", "is_bad", "--report", str(output / "quality.html")],
        ["evaluate", str(dataset), "--rules", str(rule_a), "--report", str(output / "strategy-a.html")],
        ["evaluate", str(dataset), "--rules", str(rule_b), "--report", str(output / "strategy-b.html")],
        ["compare", str(dataset), "--rules-a", str(rule_a), "--rules-b", str(rule_b), "--report", str(output / "comparison.html")],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run([sys.executable, "-m", "risk_eval.cli.main", *command], cwd=root,
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        (output / f"cli-{index}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"CLI failed ({result.returncode}): {command[0]}\n{result.stderr}")

    df = pd.read_csv(dataset)
    assert len(df) == 50
    engines = [RuleEngine.from_yaml(str(path)) for path in [rule_a, rule_b]]
    predictions = [engine.execute(df).final_decisions for engine in engines]
    result = Evaluator().compare(df, *predictions, name_a=engines[0].config.name, name_b=engines[1].config.name)
    expected = [
        {"true_positive": 20, "false_positive": 0, "true_negative": 30, "false_negative": 0},
        {"true_positive": 17, "false_positive": 0, "true_negative": 30, "false_negative": 3},
    ]
    for metrics, counts, recall, hit_rate in zip([result.metrics_a, result.metrics_b], expected, [1.0, 0.85], [0.40, 0.34]):
        assert metrics.confusion_matrix.model_dump() == counts
        assert metrics.recall == recall
        assert metrics.hit_rate == hit_rate
        assert metrics.false_positive_rate == 0.0
        assert metrics.auc_roc is None and metrics.ks_statistic is None
    for name in ["quality.html", "strategy-a.html", "strategy-b.html", "comparison.html"]:
        content = (output / name).read_text(encoding="utf-8")
        assert "<html" in content and len(content) > 500
    (output / "metrics.json").write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "sample_rows": 50, "reports": 4, "output": str(output)}))


if __name__ == "__main__":
    main()
