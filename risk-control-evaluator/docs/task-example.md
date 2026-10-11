# 离线风控策略复核任务

信贷策略上线前，分析人员需要用同一批已标注样本对比规则。这个练习使用仓库内50行演示数据，不接入企业数据或线上系统。

输入为`data/sample_credit_data.csv`、`configs/example_rules.yaml`、`configs/example_rules_b.yaml`。`is_bad=1`为坏样本，0为好样本；`reject`和`review`都计入命中，`accept`计入放行。缺失标签的记录不参与指标统计。比较是同批样本的离线回放，不属于线上随机分流的A/B试验。

执行要求：完成数据质检、两套策略回放、召回率和误判率复核，并生成HTML报告。调用方式见README；参考实现为`src/risk_eval`。`python scripts/verify-example.py --output validation-output`自动调用三个CLI并独立检查输出。

验收口径：A混淆矩阵为TP20、FP0、TN30、FN0；B为TP17、FP0、TN30、FN3。A召回率1.00、命中率0.40；B召回率0.85、命中率0.34；两者误判率均为0。指标可由CSV标签和逐行规则命中结果人工核对。复核脚本使用这些固定结果，不从被测实现生成预期值。

产物为`quality.html`、`strategy-a.html`、`strategy-b.html`、`comparison.html`和`metrics.json`。三个CLI类型均须返回0；复核脚本核对混淆矩阵、关键指标及报告存在后返回0，差异则非零退出。报告的浏览器视觉效果需要另行检查。

没有向CLI提供连续风险分数，因此KS和AUC应为缺失，不能写成测得的数值。Python接口支持传入连续分数，相关能力有单独测试。空标签、规则缺值、OR解释与执行一致性、指标零值由测试覆盖；线上效果、实际样本规模与模型有效性不在这个示例的验收范围内。

Dockerfile提供相同输入、依赖和验证入口。容器运行是否通过应以真实日志为准，不能由Dockerfile存在推定。
