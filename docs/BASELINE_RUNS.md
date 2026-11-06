# 最终两组基线的独立运行命令

本页补齐同9B七表对照与真实错误八表四方法的入口。使用原归档工作脚本、源内学习和评分函数；新入口只绑定新机器路径、创建目录链接及自有服务，不运行历史controller、历史PID或旧输出导入逻辑。以下为复现者执行的命令，本次整理没有执行模型或训练任务。

## 依赖与机器配置

先按主复现说明安装core/inference/基线环境，再安装固定TabClean源码及独立依赖：

```sh
python scripts/setup.py tabclean
cp supplement.machine.example.json supplement.machine.json
```

填写`run_python`、`infer_python`、`codex`、`model_dir`、空闲GPU/端口。`baseline_sources`指向固定源码的`.sources`，`tabclean_dependencies`指向`.dependencies/tabclean`。真实错误组还须设置`baseline_environments`为包含raha/holoclean/cdc的环境目录，以及`postgres_bin`。所有源码提交、包版本均校验；依赖缺失在模型启动前报错。

## 同9B七表对照

```sh
# 用原始UCI策展代码重建21例；每份输入必须与论文原SHA相同。
python scripts/rebuild_tabclean_inputs.py --output prepared/tabclean
python scripts/paper_baselines.py prepare --study tabclean_same9b \
  --inputs prepared/tabclean --work runs/tabclean --machine supplement.machine.json
python scripts/owned_model_service.py start --work runs/tabclean --gpu 0
python scripts/paper_baselines.py run-case --work runs/tabclean \
  --case confirmation_banknote_s2021_mixed --method paired --gpu 0
# 对runs/tabclean/configs/cases.json中全部21例分别调用上面的run-case。
python scripts/owned_model_service.py stop --work runs/tabclean --gpu 0
python scripts/paper_baselines.py verify --work runs/tabclean
python scripts/paper_baselines.py score --work runs/tabclean
```

`paired`使用原`paired_simple.py`，同时运行EvidForge与作者TabClean图。所有方法保持该组0.7温度、top-p 0.8、top-k 20、8192输出上限，不能套用其他组的0温度。EvidForge的prior/context各5次真实调用；TabClean保持原作者10轮、早停3、开发调试3。独立评分只在两方法输出封存、模型停止后执行，调用原`study_simple`函数。结果写入新运行目录的`reports/all_scores.json`、`aggregate.json`及配对/成本文件。

## 真实错误八表四方法

从`experiments/paper_runs/real_dirty/public_data_sources.json`指定的公开作者源获取CSV，按其中`path`放到自己的原表缓存目录。该配置只有源URL、ZIP成员、文件SHA及固定子集行号，没有数据值。原表未改变时可从公开源和行分区配置重建24例，逐文件对照原输入SHA。

```sh
python scripts/rebuild_real_inputs.py --source-root /path/to/author_pairs \
  --output prepared/real_dirty
python scripts/paper_baselines.py prepare --study real_dirty \
  --inputs prepared/real_dirty --work runs/real_dirty --machine supplement.machine.json
python scripts/owned_model_service.py start --work runs/real_dirty --gpu 0
python scripts/paper_baselines.py run-case --work runs/real_dirty \
  --case movies_s1013 --method full --gpu 0
python scripts/paper_baselines.py run-case --work runs/real_dirty \
  --case movies_s1013 --method tabclean --gpu 0
# 对configs/cases.json中的全部24例运行full和tabclean后停止自有服务。
python scripts/owned_model_service.py stop --work runs/real_dirty --gpu 0
# HoloClean使用自有PostgreSQL14集群，原协议端口55443，不能停其他集群。
cd runs/real_dirty
/path/to/core/bin/python code/postgres_expansion.py init
/path/to/core/bin/python code/postgres_expansion.py start
cd ../..
python scripts/paper_baselines.py run-case --work runs/real_dirty \
  --case movies_s1013 --method raha_holoclean
python scripts/paper_baselines.py run-case --work runs/real_dirty \
  --case movies_s1013 --method cdc
# 对全部24例运行两个CPU方法，然后按同一自有集群的stop命令关闭PostgreSQL。
python scripts/paper_baselines.py verify --work runs/real_dirty
python scripts/paper_baselines.py score --work runs/real_dirty
```

该入口只允许论文四方法full、raha_holoclean、cdc、tabclean，不运行GARF。三个最终CDCv4脚本和HoloClean原种子脚本已归档。温度保持该组0，所有方法共享固定输入及授权源。原研究有25条沿用封存结果；在独立复现目录中重新运行全部96个方法输出，不能把那25条旧分数传给学习过程。评分调用原`score_recovery_v3.metric/align/truth_align/aggregate_records`；只将方法目录限定为最终四种，不改数值公式或资格阈值。

## 失败与评分边界

每个方法的首次派发写marker；失败保留日志、已用预算及部分技能，入口拒绝整例重播。未知失败、缺失输出、未闭合native、种子/采样不符、自有模型仍活均阻止评分。评分写一次性marker并读取新参考，每例每个参考文件一次；不得删marker重评分。评分结果与已发表结果分开存放，所有有效0分和负收益保留；不能强求重跑获得LLM阳性。

`owned_model_service.py`核查实际host、GPU计算进程、PID/PGID/cwd/端口，只操作自己的服务。它只管理生命周期，原方法的请求采样仍由归档`model_service.py`控制。
