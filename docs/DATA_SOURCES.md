# 论文实验的数据准备

仓库保留数据构造代码、固定种子、实体分区与输入指纹，不分发完整输入包和正式参考数据。获得原始公开数据后，由策展脚本生成授权教学输入与独立评分参考；学习和正式修复进程禁止读取参考。不得用旧预测、旧技能或结果表充当重新运行的输出。

| 论文实验 | 数据准备代码/配置 | 输入来源与口径 |
|---|---|---|
| 八表修复、六表下游、A–F消融 | `experiments/templates/repair/generate_benchmarks.py`、`experiments/templates/downstream/generate_benchmarks.py`及相邻catalog | 官方UCI原始ZIP；按原种子和实体分组构造，预测标签与留出数据受保护。A–F使用同seed下游输入。主入口`reproduce.py`负责下载和构造。 |
| 导师补充Car/Concrete、Hospital案例 | `advisor_supplement/execution_code.tar.gz`中的`curator.py`、`curator_v2.py`、`curator_v3.py` | 同seed原下游输入；比例/错误类型按该轮协议。自然Hospital/Flights使用固定TabClean作者数据源，不额外污染原自然错误。 |
| 70%数字替换、有无LLM | `scripts/rebuild_controlled_inputs.py --study llm_digit70` | Retinopathy/Airfoil官方ZIP；直接污染70%可污染格，五种子。公开角色配置和逐文件SHA在provenance中。 |
| 三类新错误、有无LLM | `scripts/rebuild_controlled_inputs.py --study new_errors` | SPECTF/AutoMPG/Airfoil官方ZIP；各目标字段30%合格格，三种子；不叠旧污染。 |
| 阈值扫描 | `scripts/reproduce_threshold.py` | 使用同seed新完成的主消融A技能、原输入和独立参考；不重新污染或拟合；零偏移必须与该A输出字节一致。 |
| 联合示例预算 | `example_budget/execution_code.tar.gz`中的`curator.py` | 从同seed下游输入采样100个固定实体组，三档共享60组测试；学习/外校分别5/10/20组。保留原脏值，不重新污染。 |
| Retinopathy鲁棒性 | `robustness_retinopathy/execution_code.tar.gz`中的`curate_retino.py` | 使用原下游生成器，只绑定该轮种子、错误类型与比例；单类名义比例与实际比例分别记录。 |
| 同9B七表TabClean | `tabclean_same9b/execution_code.tar.gz`中的`data.py`；`input_source_catalog.json`、`original_config/simple_protocol_v1.json` | 固定七个UCI数据集、2021–2023三种子、confirmation/mixed；只运行该最终范围。`data.py`的默认pilot入口不属于论文运行命令。 |
| 封存技能处理新批次 | `sealed_reuse/execution_code.tar.gz`中的`curate_reuse.py`、`source_generator.py` | 官方Retinopathy/Airfoil；独立data_seed、三个实体隔离批次，不能替换成相同输入重复执行。 |
| 真实错误八表四方法 | `real_dirty/execution_code.tar.gz`中的`curate_formal.py`、`curate_legacy.py`、`curate_expansion_v2.py` | 使用原作者dirty/clean配对，不再污染。`public_data_sources.json`列出原始URL、ZIP成员、SHA与三个固定小子集的行索引；没有数据内容。 |
| Hospital/Flights受控语义检查 | `semantic_historical/execution_code.tar.gz`中的`semantic_curator_v2.py`及`original_config/controlled_protocol.json` | 从指定作者配对源还原后制造受控合法值冲突；不称为原始自然脏表或新盲测。 |

TabClean七表可直接使用`rebuild_tabclean_inputs.py`构造；真实错误八表可使用`rebuild_real_inputs.py`，从公开作者原表和固定行分区配置重建。两组均要求逐文件符合原输入SHA，详见[基线运行命令](BASELINE_RUNS.md)。行分区配置只有行号、合成实体组、字段声明和指纹，不含特征或参考值。

上表的研究目录均位于`experiments/paper_runs/`。原协议SHA、最终案例ID、模型参数与论文对应关系见各组`config.json`和总索引`provenance/paper_experiments.json`。原脚本中旧路径属于来源绑定，新机器需绑定本机原始数据、解释器、模型及基线源码路径；不能执行其中的历史PID或直接恢复旧controller。

## 同9B七表的最终输入构造

`data.py`的源解析使用名为`configs/protocol_v1.json`的目录。将仅包含七表源目录的`input_source_catalog.json`放到该位置；最终方法、种子和采样仍以`simple_protocol_v1.json`、`run_config.json`为准。导入`data`后按最终七表和2021–2023循环调用`data.build(dataset, 'confirmation', seed, 'mixed')`，不要运行其默认的pilot入口，也不要构造`structured_repeat`或Dry Bean等未采用条件。作者算法固定源码提交和依赖锁已收录。

## 真实错误数据来源

Movies与Beers使用固定Raha源码提交`7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c`下的作者配对CSV；Movies是`datasets/movies_1/`。地址表来自[AddressTable作者发布](https://zenodo.org/records/20841898)。Library与ShoppingCenter使用归档中的effectiveness配对；Museum及三个小表的母表使用相应named配对，不能互换。School/Government/JobPosting的固定小子集索引保存在数据配置中，按dirty身份选择，clean按原ID对齐；不能按修复结果重抽子集。四份地址类展示结果属于同一来源家族。

数据仍受上游来源和许可约束，代码仓库不改变其权利归属。需要原作者冻结输入时，可由有权限的作者运行`reproduction_data.py export-server`导出，再通过`verify`与`prepare`校验；该步骤不拟合、不调用模型、不导入旧输出。固定输入包不随本仓库发布。
