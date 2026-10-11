# Risk Control Evaluator

风控数据质量检查与规则策略离线评估练习。包含多格式加载、字段与标签检查、YAML规则执行、指标计算、CLI、HTML报告和Streamlit入口。

这是公开的学习与能力说明作品，代码和本次修复借助AI工具完成。仓库提供实现和验证方式，不以演示数据或提交记录证明线上采用、开发年限、全栈经验或生产性能。

## 安装和使用

Python 3.12是本次验证环境。直接依赖版本见`requirements-validation.txt`；该文件没有锁定间接依赖。项目声明的Python 3.10+范围尚未逐版本验证。

```bash
git clone https://github.com/fang-2007/platform-tasks.git
cd platform-tasks/risk-control-evaluator
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-validation.txt
python -m pip install --no-deps -e .
```

CLI接口保持以下用法：

```bash
risk-eval quality-check data/sample_credit_data.csv --label is_bad --report quality.html
risk-eval evaluate data/sample_credit_data.csv --rules configs/example_rules.yaml --label is_bad --report strategy-a.html
risk-eval compare data/sample_credit_data.csv --rules-a configs/example_rules.yaml --rules-b configs/example_rules_b.yaml --label is_bad --report comparison.html
risk-eval --help
```

Web入口为可选依赖：

```bash
python -m pip install ".[web]"
risk-eval serve
```

## 规则和指标口径

YAML支持数值比较、字符串与集合条件、空值检查及AND/OR组合。`combine_logic`可选`any`、`all`、`score_threshold`；启用打分策略时必须提供阈值。数值等比较条件必须有`value`，空值条件可以省略它。

```yaml
name: age-check
combine_logic: any
rules:
  - name: older-applicant
    conditions:
      - field: age
        operator: gt
        value: 50
    logic: and
    action: reject
```

`is_bad=1`计为坏样本。`reject`和`review`均计为命中，`accept`计为放行。指标在标签非空的记录上计算。A/B命名表示同批样本的两套规则离线对比，不表示线上随机分流实验。

Python接口可以接收连续分数计算KS和AUC。当前CLI及Web入口不传连续分数，因此这两个指标显示`N/A`。零值指标显示0，只有未定义的指标显示缺失。

## 自动验证与真实示例结果

```bash
python -m pytest
python scripts/verify-example.py --output validation-output
```

2026年10月11日本地Python 3.12复核：72项测试通过。覆盖率统计只包含核心模块及工具模块，排除了CLI、Web和报告；见`docs/validation.json`。示例复核脚本运行CLI并生成四份报告，使用固定、人工可核对的预期结果检查指标。

使用仓库原50行演示CSV与两份YAML得到：

| 指标 | A | B |
| --- | --- | --- |
| TP / FP / TN / FN | 20 / 0 / 30 / 0 | 17 / 0 / 30 / 3 |
| 召回率 | 100% | 85% |
| 误判率 | 0% | 0% |
| 命中率 | 40% | 34% |
| 放行率 | 60% | 66% |
| F1 | 1.0000 | 0.9189 |
| Lift | 2.50 | 2.50 |
| KS / AUC | N/A / N/A | N/A / N/A |

在此示例中，A召回率更高，B放行率更高。样本很小，不能据此推荐线上策略。演示数据来源及用途限定为仓库内练习，未接入真实信贷数据。

[任务及验收示例](docs/task-example.md)给出输入输出与判定标准；[里程碑](docs/milestones.md)记录公开提交和本次修复，没有追补不存在的历史开发记录。

## Docker和持续集成

```bash
docker build -t risk-control-validation .
docker run --rm risk-control-validation
```

Docker默认执行同一示例复核入口。工作流文件为仓库根目录`.github/workflows/risk-control-quality.yml`，配置Python测试、示例验证和Docker构建运行。实际运行状态应查看[GitHub Actions](https://github.com/fang-2007/platform-tasks/actions/workflows/risk-control-quality.yml)。本地Docker环境不可用，不能把本地Python通过写成Docker通过；远端结果会在验证记录中单独记载。

## 目录

```text
risk-control-evaluator/
  configs/                    两套示例规则
  data/sample_credit_data.csv  50行演示数据
  src/risk_eval/
    core/                     加载、质检、规则执行与评估
    cli/                      Typer终端入口
    web/                      Streamlit入口
    report/                   Plotly图表与HTML报告
    utils/                    配置、异常与日志
  tests/                      原测试与边界回归验证
  scripts/verify-example.py   可执行示例验收
  docs/                      任务、里程碑与验证证据
  Dockerfile
  requirements-validation.txt
  pyproject.toml
```

## 当前验证范围

已验证核心测试及小样本CLI流程。未验证生产数据库、多版本兼容、线上效果或大规模性能。SQL及额外格式的支持范围应结合具体输入、依赖和测试判断。MIT许可见`LICENSE`。
