# 公开提交与修复记录

2026年10月11日，仓库以`54064c6cd24ff134ec53feaef1987184ef6113af`初始化；同日以`88d74d079bb791712f1edc74e49c596af133cf57`公开风控工具源码。这两次提交是公开发布记录，不能据此推断原始代码开发时间或工作采用情况。

本次修复针对公开源码可复现的问题：比较条件缺值和评分阈值省略仍通过校验；OR规则解释与真实执行矛盾；F1/Lift的有效零值丢失；CLI、HTML和看板将零值显示为N/A。修复由Codex协作完成，保留原命令、输入格式与默认决策口径。新增19例回归检查，原53例仍通过。

同时补列Parquet依赖，给出固定版本的验证依赖、任务与验收说明、Dockerfile和CI工作流。README示例以原50行CSV和规则实际回放结果更新。删除“生产就绪”“全栈交付”和没有证据的生产规模主张，不创建或倒填历史开发提交。

本地验证记录见`validation.json`。远端CI的实际状态以[工作流运行](https://github.com/fang-2007/platform-tasks/actions/workflows/risk-control-quality.yml)和对应提交SHA为准。CI工作流配置与实际通过是两类证据。

2026年10月11日，修复提交`60998c52b09bf2cc9047b140d4ef4714b22cc3ab`的[CI运行38130179831](https://github.com/fang-2007/platform-tasks/actions/runs/38130179831)成功，job为succeeded。工作流执行Linux测试、示例复核及Docker构建运行；证据摘要见`ci-receipt.json`。后续文档同步不改变该固定代码快照。
