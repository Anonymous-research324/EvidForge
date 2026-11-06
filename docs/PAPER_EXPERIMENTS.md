# 论文实验与配置索引

保留主论文17张表的文本源与对应原始数值；成品图像和PDF不分发。所有零值、反例和N/A保留。

| 实验 | 论文位置 | 结果 | 配置/脚本 |
|---|---|---|---|
| 八表准确修复主实验 | 表1、表9 | [结果](../results/studies/repair_main/) | [配置](../experiments/paper_runs/repair_main/) |
| 六表下游预测 | 表2 | [结果](../results/studies/downstream_main/) | [配置](../experiments/paper_runs/downstream_main/) |
| 完整六表配对模块消融 | 表5、表7 | [结果](../results/studies/module_ablation/) | [配置](../experiments/paper_runs/module_ablation/) |
| Car/Concrete独立消融与Concrete鲁棒性 | 表8、表13的Concrete及Hospital1015案例依据 | [结果](../results/studies/advisor_supplement/) | [配置](../experiments/paper_runs/advisor_supplement/) |
| 70%数字替换有/无LLM | 图2 | [结果](../results/studies/llm_digit70/) | [配置](../experiments/paper_runs/llm_digit70/) |
| 五档置信门槛敏感性 | 表11 | [结果](../results/studies/threshold/) | [配置](../experiments/paper_runs/threshold/) |
| 学习与外校联合示例预算 | 表12 | [结果](../results/studies/example_budget/) | [配置](../experiments/paper_runs/example_budget/) |
| Retinopathy污染率与错误类型 | 图3、表13的Retinopathy | [结果](../results/studies/robustness_retinopathy/) | [配置](../experiments/paper_runs/robustness_retinopathy/) |
| 历史模型调用与流程成本 | 表6 | [结果](../results/studies/efficiency/) | [配置](../experiments/paper_runs/efficiency/) |
| Hospital三格机制案例 | 表15 | [结果](../results/studies/hospital_cases/) | [配置](../experiments/paper_runs/hospital_cases/) |
| 原系统三类新错误有/无LLM | 表10 | [结果](../results/studies/new_errors/) | [配置](../experiments/paper_runs/new_errors/) |
| 同9B七表TabClean受控对照 | 表4、表16 | [结果](../results/studies/tabclean_same9b/) | [配置](../experiments/paper_runs/tabclean_same9b/) |
| 封存技能处理三个新批次 | 图4 | [结果](../results/studies/sealed_reuse/) | [配置](../experiments/paper_runs/sealed_reuse/) |
| 八份真实错误表四方法 | 表3、表17 | [结果](../results/studies/real_dirty/) | [配置](../experiments/paper_runs/real_dirty/) |
| Hospital/Flights历史受控冲突检查 | 表14；补充材料历史来源S9 | [结果](../results/studies/semantic_historical/) | [配置](../experiments/paper_runs/semantic_historical/) |

配置JSON是从实际最终协议提取的论文范围视图，原协议SHA和服务器来源一并提供。真实错误严格四方法；TabClean使用simple协议，不收录旧计划。原脚本存档保持原字节；迁移时使用显式路径绑定并保存新旧SHA。部分受控输入已可从官方原始数据重建，其他输入可由原作者导出固定输入包。已验证范围及尚待完成的独立流程见[复现说明](REPRODUCE.md)，不把源码存档等同于全部端到端复现通过。

完整源码/配置/记录来源和筛选对应见`provenance/paper_experiments.json`。主程序在`engine/repair_platform.py`，124核心文件保持不变。
