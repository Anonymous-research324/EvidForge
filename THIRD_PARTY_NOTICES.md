# 来源与许可

Raha（BigDaMa/raha）、HoloClean（HoloClean/holoclean）和CDC（se-jaeger/conformal-data-cleaning）固定提交见 provenance/baseline_sources.json。安装器下载原仓库并保留其许可/署名。

GARF使用CDC作者公开SQLite移植，非原Oracle论文环境完全复现。CDC源内分组校准、小源采样适配，每字段60秒CPU/10超参试验；conformal-inference0.1.6。

CAAFE：noahho/CAAFE，冻结提交bb65b4d517bd86056352ad2ee98be6655f2d5aec。engine/baselines/caafe_official_source保留Python源署名，原README声明Apache2.0；此处为原提示与AST执行器的同9B/XGBoost适配。

Qwen权重、Codex CLI、第三方环境二进制不重新分发，按原提供方许可安装。本次不代作者新增开源许可证。仓库为公开状态；访问权限不等于获得第三方额外授权。


补充实验的 TabClean 使用作者固定源码及同9B适配，来源提交与参数见各轮协议和执行清单。真实错误表来自发布方dirty/reference配对；这些真实错误作者配对表原件不随仓库重新分发。字体、权重、虚拟环境和拟合大转储不随本次发布分发。


本地复现使用的Concrete Compressive Strength、Airfoil Self-Noise、Diabetic Retinopathy Debrecen均来自UCI Machine Learning Repository，官方页面声明CC BY 4.0。原作者、DOI、来源及许可链接见provenance/public_input_licenses.json。复现者从官方获取数据，按公开脚本执行论文原有固定分区、污染和批次安排，在本机生成输入及评分参考。服务器保存的冻结输入和参考答案不随仓库分发。原数据许可不改变仓库代码、第三方软件或模型的许可。
