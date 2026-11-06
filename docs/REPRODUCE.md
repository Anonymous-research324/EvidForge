# 复现说明

仓库不包含成品图片。代码、论文表格数值、运行参数、种子、来源指纹及实际脚本分别保留。

## 1. 核对已发布结果（无需模型）

```sh
python3 scripts/verify_paper_release.py
python3 -B -m unittest discover -s tests -v
```

这会检查124核心、17张表源、结果计数/均值/样本SD和文件指纹。**这些是结果核对，不是重新运行模型实验。**

## 2. 运行系统及主三组实验

Linux建议核心/推理Python3.12，基线独立Python3.9；需要Git、Node/npm和PostgreSQL。依赖锁在`environments/`，来源提交在`provenance/baseline_sources.json`。

```sh
python3.12 scripts/setup.py all --python312 /path/to/python3.12 --python39 /path/to/python3.9
npm install --prefix .tools @openai/codex@0.153.1
python3 scripts/download_model.py --output models/Qwen3.5-9B
cp machine.example.json machine.json
```

编辑machine.json，填写本机解释器、模型、Codex、基线环境与数据源路径，以及空闲GPU/端口。权重须通过`provenance/model-files-9B.sha256`；不能静默替换新版模型。模型约18GB，不放进Git。主系统入口为`engine/repair_platform.py`，`run`加载封存技能执行，不继续调用模型。

```sh
.venvs/core/bin/python reproduce.py doctor
.venvs/core/bin/python reproduce.py run --study repair --seed 1013 --config machine.json
.venvs/core/bin/python reproduce.py run --study downstream --seed 1013 --config machine.json
.venvs/core/bin/python reproduce.py run --study ablation --seed 1013 --config machine.json
```

各轮还须运行1014/1015。curator下载官方UCI原始数据并检查原始SHA；已有分数不传给worker。正式参考和预测标签不能进入修复进程，输出封存后才能由scorer读取。失败阶段与评分marker不可删除后重跑。主三组便携入口现列出A–F消融。F仅关闭项目Hook注册，保留原MCP与后端权限和独立外校；与关闭源内反馈、完全不用LLM分别处理。A/E/F各自重新学习，B/C/D执行同一A技能。

## 3. 补充实验的输入与运行入口

`experiments/paper_runs/<study>/original_config/`保存实际协议原字节，指纹绑定见`provenance/original_config_bindings.json`。相邻`config.json`是论文范围索引，不代替执行协议。`execution_code.tar.gz`保存原工作脚本；请使用下列绑定工具，避免执行存档controller或历史PID。

### 从官方原始数据重建受控污染

以下两组已实现从官方UCI原始ZIP和公开实体角色ID重建：`new_errors`（27例）与`llm_digit70`（10例）。每份生成的公开/私有输入必须与原输入SHA逐文件完全相同，才可进入运行；不会读取旧预测、技能或分数来生成数据。

```sh
python scripts/reproduction_data.py download-uci --dataset airfoil --out raw/airfoil.zip
# new_errors另下载spectf、auto_mpg；llm_digit70另下载retinopathy。
python scripts/rebuild_controlled_inputs.py --study new_errors --raw-dir raw --output prepared/new_errors
# 若只做一例接口验证：追加 --cases original_cross_field_addition_airfoil_s7013
```

`curator_only/`是仅策展角色可访问的临时构造来源，不在方法输入清单中。模型仅访问授权学习样例；评分参考不进入学习与正式修复。

### 将固定输入包带到新机器

其他补充组已有输入导出/验证入口，输入包只含固定输入，不含旧输出、拟合模型、服务PID或预算。原作者可在来源服务器导出，再将输入包带到新机器；输入包目前没有公开下载地址，不能称其他组已实现从官方原始数据自动重建。

```sh
python scripts/reproduction_data.py export-server --study example_budget --output /new/staging/example_budget
python scripts/reproduction_data.py prepare --bundle-dir /path/to/bundle --work prepared/example_budget
```

### 新机器运行

```sh
cp supplement.machine.example.json supplement.machine.json
# 填写本机解释器、Codex、模型、空闲GPU和端口。
python scripts/supplement_runtime.py prepare --study new_errors --inputs prepared/new_errors \
  --work runs/new_errors --machine supplement.machine.json
python scripts/supplement_runtime.py doctor --work runs/new_errors
python scripts/supplement_runtime.py run-batch --work runs/new_errors --gpu 0 --max-real-calls 270
```

单例验证用`smoke --work ... --case ... --gpu 0`，只选一例时先在构造输入阶段指定该例。2026-11-05已在新目录实际完成Airfoil跨字段相加7013的一例：重新拟合、10次真实9B调用、关闭服务后一次评分；有/无LLM的全部准确修复计数与原例一致。证据见`provenance/real_smoke_independent_review_20261105.json`。这不等于其余实验都已重新运行。新错误和70%数字替换每例最多10次真实模型调用；其他原始模块消融含独立重学分支，调用上限依原协议计算。`prepare`、`doctor`与输入验证均不调用模型。启动等待由机器配置控制，不改变尝试次数、候选或门槛。原绘图脚本若需要字体，在`display_fonts`填本地中文常规/粗体及西文字体；仅复制到该次运行目录，不随仓库分发，字体替换不影响评分。整批入口会在调用模型前检查这些展示依赖。

当前新工作目录绑定与整批生命周期覆盖：新错误、70%数字替换、导师补充、Retinopathy鲁棒性、历史受控语义检查、样例预算。每组仍须先检查`doctor`，只有实际跑完才能称该机执行通过。源码中的老绝对路径只按显式字符串绑定，新旧SHA与绑定清单留档；124核心不变。已有状态目录拒绝自动重播；未知失败阻止评分，评分marker不能删除或重新评估。

封存新批次复用另有单例`smoke`生命周期入口：学习并封存后关闭全部自有服务，再调用原CPU离线执行器和原评分器；仍待实际复用验证，不能据代码已加入就称通过。

阈值扫描已增加`python scripts/reproduce_threshold.py prepare --main-runs runs --work runs/threshold --machine machine.json`及随后`run --work runs/threshold`。必须先跑完新的三种子主消融A；该入口拒绝代入其他目录的历史技能，δ=0必须与新A输出逐字节一致，不新增模型调用。尚未在新完成的主消融上实际运行整批90输出。

TabClean与真实错误四方法的独立入口、依赖目录、公开源数据构造和一次评分命令已补充，见[基线运行说明](BASELINE_RUNS.md)。两组分别用原策展函数/固定行分区重建21/24例，要求生成文件与论文原输入SHA一致。该文件构造检查不调用模型、不拟合或评分；完整方法运行仍由复现者按说明执行。历史耗时仅供核对原运行记录；新机器必须另记实际耗时，不以旧耗时冒充重跑。

TabClean固定`Wangyibo321/TabClean`提交`ae292b6ff8118c8786cd7c97364e9a8a07f00b62`。`python scripts/setup.py tabclean`下载固定源码并将实际依赖版本安装到单独`.dependencies/tabclean/`，不覆盖核心环境。七表同9B对照的0.7温度与真实错误轮、主系统0温度分开，不能统一覆盖。

## 统计与边界

保留所有预定种子；样本标准差ddof=1，缺失保持N/A，不补零。不通过外校的研究prior与部署资格分开。不同数据/污染/硬件轮不混合均值；历史计时边界不同，不计算统一加速比。真实错误的基线适配限制见KNOWN_LIMITATIONS.md。
