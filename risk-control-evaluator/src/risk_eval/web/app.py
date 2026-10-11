"""Streamlit web application for risk-eval."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from risk_eval import __version__
from risk_eval.core.data_loader import DataLoaderFactory
from risk_eval.core.evaluator import Evaluator
from risk_eval.core.quality_checker import QualityChecker
from risk_eval.core.rule_engine import RuleEngine
from risk_eval.report.chart_generator import ChartGenerator
from risk_eval.utils.config import RuleSetConfig
from risk_eval.utils.exceptions import RiskEvalError
from risk_eval.utils.logger import configure_logging

configure_logging(log_level="INFO")

st.set_page_config(
    page_title="Risk Control Evaluator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> tuple[pd.DataFrame | None, str, RuleEngine | None, RuleEngine | None]:
    """Render sidebar and return user inputs."""
    st.sidebar.title("🛡️ Risk Control Evaluator")
    st.sidebar.caption(f"v{__version__}")

    # Data upload
    st.sidebar.header("📁 Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset (CSV, Excel, JSON, Parquet)",
        type=["csv", "xlsx", "xls", "json", "parquet"],
    )

    label_column = st.sidebar.text_input("Label column", value="is_bad")

    df = None
    if uploaded_file is not None:
        try:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(uploaded_file)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(uploaded_file)
            elif suffix == ".json":
                df = pd.read_json(uploaded_file)
            elif suffix == ".parquet":
                df = pd.read_parquet(uploaded_file)
            st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
        except Exception as exc:
            st.sidebar.error(f"Failed to load data: {exc}")
            df = None

    # Rules upload
    st.sidebar.header("📋 Rules")
    rules_a_file = st.sidebar.file_uploader("Rules A (YAML)", type=["yaml", "yml"], key="rules_a")
    rules_b_file = st.sidebar.file_uploader("Rules B (YAML, optional)", type=["yaml", "yml"], key="rules_b")

    engine_a = None
    engine_b = None

    if rules_a_file is not None:
        try:
            content = rules_a_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            config = RuleSetConfig.model_validate_yaml(content)
            engine_a = RuleEngine(config)
            st.sidebar.success(f"Rules A: {engine_a.config.name}")
        except Exception as exc:
            st.sidebar.error(f"Rules A error: {exc}")

    if rules_b_file is not None:
        try:
            content = rules_b_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            config = RuleSetConfig.model_validate_yaml(content)
            engine_b = RuleEngine(config)
            st.sidebar.success(f"Rules B: {engine_b.config.name}")
        except Exception as exc:
            st.sidebar.error(f"Rules B error: {exc}")

    # Or use inline rule editor
    st.sidebar.markdown("---")
    st.sidebar.info("Upload YAML rule files, or use the editor below to create rules.")

    return df, label_column, engine_a, engine_b


def render_data_preview(df: pd.DataFrame) -> None:
    """Render data preview section."""
    st.subheader("Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isna().sum().sum())

    with st.expander("View raw data"):
        st.dataframe(df.head(100), use_container_width=True)


def render_quality_tab(df: pd.DataFrame, label_column: str) -> None:
    """Render data quality check tab."""
    st.subheader("🔍 Data Quality Check")

    checker = QualityChecker()
    report = checker.check(df, label_column=label_column)

    # Summary cards
    cols = st.columns(4)
    cols[0].metric("Total Rows", report.total_rows)
    cols[1].metric("Total Columns", report.total_columns)
    cols[2].metric("Warnings", len(report.warnings))
    cols[3].metric(
        "Critical Issues",
        "Yes" if report.has_critical_issues() else "No",
        delta="⚠️" if report.has_critical_issues() else "✅",
    )

    if report.warnings:
        st.warning("\n".join(f"• {w}" for w in report.warnings))

    # Column quality table
    st.markdown("#### Column Quality Details")
    quality_df = pd.DataFrame([
        {
            "Column": c.column,
            "Type": c.dtype,
            "Null Rate": f"{c.null_rate:.1%}",
            "Unique Rate": f"{c.unique_rate:.1%}",
            "Constant": "Yes" if c.is_const else "No",
            "Outliers": f"{c.outlier_rate:.1%}" if c.outlier_rate else "N/A",
        }
        for c in report.columns
    ])
    st.dataframe(quality_df, use_container_width=True)

    # Duplicates
    if report.duplicates:
        st.markdown("#### Duplicates")
        st.write(
            f"Duplicate rows: **{report.duplicates.duplicate_rows:,}** "
            f"({report.duplicates.duplicate_rate:.2%}) "
            f"- Checked columns: {', '.join(report.duplicates.checked_columns)}"
        )

    # Label summary
    if report.label_summary:
        st.markdown("#### Label Distribution")
        ls = report.label_summary
        label_df = pd.DataFrame([
            {"Class": k, "Count": v, "Rate": f"{ls.class_rates.get(k, 0):.1%}"}
            for k, v in ls.class_distribution.items()
        ])
        st.dataframe(label_df, use_container_width=True)
        if ls.is_imbalanced:
            st.warning(f"Label is imbalanced (ratio {ls.imbalance_ratio:.1f}:1)")


def render_evaluation_tab(
    df: pd.DataFrame,
    label_column: str,
    engine: RuleEngine,
    strategy_name: str,
) -> None:
    """Render single strategy evaluation tab."""
    st.subheader(f"📊 Strategy Evaluation: {strategy_name}")

    rule_result = engine.execute(df)
    evaluator = Evaluator(label_column=label_column)
    metrics = evaluator.evaluate(df, rule_result.final_decisions, strategy_name=strategy_name)

    # Metrics cards
    cols = st.columns(4)
    cols[0].metric("Recall", f"{metrics.recall:.1%}" if metrics.recall is not None else "N/A")
    cols[1].metric("Precision", f"{metrics.precision:.1%}" if metrics.precision is not None else "N/A")
    cols[2].metric("FPR (Misjudgment)", f"{metrics.false_positive_rate:.1%}" if metrics.false_positive_rate is not None else "N/A")
    cols[3].metric("Hit Rate", f"{metrics.hit_rate:.1%}")

    cols2 = st.columns(4)
    cols2[0].metric("Pass Rate", f"{metrics.pass_rate:.1%}")
    cols2[1].metric("Accuracy", f"{metrics.accuracy:.1%}" if metrics.accuracy is not None else "N/A")
    cols2[2].metric("KS Statistic", f"{metrics.ks_statistic:.3f}" if metrics.ks_statistic is not None else "N/A")
    cols2[3].metric("AUC-ROC", f"{metrics.auc_roc:.3f}" if metrics.auc_roc is not None else "N/A")

    # Charts
    chart_gen = ChartGenerator()
    c1, c2 = st.columns(2)
    with c1:
        if metrics.confusion_matrix:
            fig = chart_gen.create_confusion_matrix_heatmap(
                metrics.confusion_matrix, f"Confusion Matrix - {strategy_name}"
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = chart_gen.create_metrics_gauge(metrics, strategy_name)
        st.plotly_chart(fig, use_container_width=True)

    # Rule hits
    st.markdown("#### Rule Hit Distribution")
    fig = chart_gen.create_rule_hit_distribution(rule_result.rule_hit_matrix)
    st.plotly_chart(fig, use_container_width=True)

    # Rule summary table
    st.markdown("#### Rule Summary")
    rule_df = pd.DataFrame([
        {
            "Rule": name,
            "Hits": s["hit_count"],
            "Hit Rate": f"{s['hit_rate']:.1%}",
            "Action": s["action"],
        }
        for name, s in rule_result.rule_summaries.items()
    ])
    st.dataframe(rule_df, use_container_width=True)


def render_comparison_tab(
    df: pd.DataFrame,
    label_column: str,
    engine_a: RuleEngine,
    engine_b: RuleEngine,
) -> None:
    """Render A/B comparison tab."""
    st.subheader("⚖️ A/B Strategy Comparison")

    result_a = engine_a.execute(df)
    result_b = engine_b.execute(df)

    evaluator = Evaluator(label_column=label_column)
    comparison = evaluator.compare(
        df,
        result_a.final_decisions,
        result_b.final_decisions,
        name_a=engine_a.config.name,
        name_b=engine_b.config.name,
    )

    st.info(comparison.recommendation)

    # Comparison chart
    chart_gen = ChartGenerator()
    fig = chart_gen.create_comparison_bar(
        comparison.metrics_a,
        comparison.metrics_b,
        engine_a.config.name,
        engine_b.config.name,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    comparison_df = pd.DataFrame([
        {
            "Metric": key.replace("_", " ").title(),
            engine_a.config.name: f"{diff['a']:.1%}",
            engine_b.config.name: f"{diff['b']:.1%}",
            "Difference": f"{diff['absolute_diff']:+.1%}",
            "Relative": f"{diff['relative_diff_pct']:+.1f}%" if diff["relative_diff_pct"] else "N/A",
        }
        for key, diff in comparison.differences.items()
    ])
    st.dataframe(comparison_df, use_container_width=True)


def main() -> None:
    """Main Streamlit app entry point."""
    st.title("🛡️ Risk Control Evaluator")
    st.caption("Professional toolkit for risk data quality and strategy evaluation")

    df, label_column, engine_a, engine_b = render_sidebar()

    if df is None:
        st.info("👈 Please upload a dataset in the sidebar to get started.")
        st.markdown("""
        ### Quick Start
        1. **Upload your dataset** (CSV, Excel, JSON, or Parquet)
        2. **Specify the label column** (e.g., `is_bad`, `default`, `fraud`)
        3. **Upload rule configuration** (YAML file with your risk rules)
        4. **View quality checks, evaluation metrics, and comparisons**

        ### Supported Rule Conditions
        - Numeric comparisons: `gt`, `gte`, `lt`, `lte`, `eq`, `ne`
        - String operations: `contains`, `starts_with`, `ends_with`, `regex`
        - Set operations: `in`, `not_in`
        - Null checks: `is_null`, `is_not_null`
        - Logical combinations: `and`, `or`
        """)
        return

    tabs = st.tabs(["📁 Data Preview", "🔍 Quality Check", "📊 Evaluation"])

    with tabs[0]:
        render_data_preview(df)

    with tabs[1]:
        render_quality_tab(df, label_column)

    with tabs[2]:
        if engine_a is None:
            st.warning("👈 Please upload a rules file to see evaluation results.")
        else:
            eval_tabs = st.tabs([engine_a.config.name] + ([engine_b.config.name] if engine_b else []) + (["A/B Comparison"] if engine_b else []))

            with eval_tabs[0]:
                render_evaluation_tab(df, label_column, engine_a, engine_a.config.name)

            if engine_b:
                with eval_tabs[1]:
                    render_evaluation_tab(df, label_column, engine_b, engine_b.config.name)

                with eval_tabs[2]:
                    render_comparison_tab(df, label_column, engine_a, engine_b)


if __name__ == "__main__":
    main()
