# NO₂ 机器学习项目 P0 重构交付报告

完成日期：2026-11-15。交付位置：当前工作目录下独立的 `NO2-ML-Prediction/`。

## 1. 实际完成范围

已按 A→B→C→D 实施独立 Python 包、输入/QC/特征版本、统一评价、五模型嵌套分组训练、制品保存加载、新样本推理、配置化 CLI、保存结果绘图及发布文档。所有开发写入新项目；没有直接移动或改写旧项目代码，没有向旧数据、历史结果、模型或论文图表路径写文件。

本次没有使用真实研究数据训练、调参、评价或重绘论文图表。所有数值测试与软件演练只使用 `tests/` 生成的人工夹具，测试产生的指标不能作为论文性能。未执行 Git init/reset/clean/commit/push，也未上传 GitHub。

已完整阅读审计报告并参考相关旧实现。保留合理算法结构，未复制旧项目、数据、演示预测编辑器或目标误差构造器。本次软件测试不构成真实数据验证、论文结果复现或 Wiley Checklist 全面合规认证。

## 2. 阶段完成及实际测试

| 阶段 | 实际开发 | 验证结果 |
| --- | --- | --- |
| A | src 包、只读仪器解析、标准单位/时间检查、QC/候选边界版本、审核门禁、共享响应及两套特征 | 初次 7 项通过；最终补充到 9 项通过 |
| B | 七指标、有效性与原因、模型比较对齐、独立评价 scope、CSV 独占写入 | 7 项通过 |
| C | 五模型注册、折内预处理及调参、嵌套 OOF/单独 refit、manifest、模型制品与推理 | 6 项通过 |
| D | 配置 CLI、旧绘图参数兼容、冻结结果渲染、依赖/安装元数据/文档 | 5 项通过；另完成 wheel 构建及隔离目标目录包验证 |

最终执行：`python -m unittest discover -s tests -t . -v`，**27 项测试全部通过，0 个未解决失败、0 个错误**。全套在当前参考环境约 11.5 s，不代表其他硬件上的耗时。

测试覆盖：解析多通道/时间列位置/编码/单位/空文件/坏数值；不规则采样与原值保留；基线、峰值、初始斜率、60 s 响应、AUC、量程与缺少窗口；候选边界不能自动放行；指标边界/负 R²/RMSLE/Adjusted R²；模型测试集合和顺序；外层分组隔离及 scaler 的实际拟合样本；两套特征的训练/推理一致性；保存前后预测一致；版本/通道/QC/列顺序拒绝；加载前哈希校验；完整 CLI、旧参数兼容、无导入写文件副作用与 CSV 防覆盖。

还执行了五模型 CLI 软件演练（人工夹具）、最终模型保存、独立预测、安装后包的绘图包装入口和 PNG/PDF/SVG 导出；查看了人工测试图的布局。演练输出均位于被忽略的 `tests/_artifacts/`。构建/目标安装位于 `tests/_build/`，没有安装或升级系统依赖。

### 已修正的开发期问题

1. 首次 A 阶段测试命令从新项目目录重复解析了 `NO2-ML-Prediction/src`，导致 import 失败；修正为当前目录下 `src` 后通过。这是命令路径问题，未改动环境。
2. D 阶段首先发现 CSV 往返后接近零的指标有约 `4×10⁻¹⁶` 差异；保存保持 17 位有效数字，读取改为 `float_precision='round_trip'`，测试采用明确的绝对/相对容差 `1e-12`。没有更改预测来匹配指标。
3. 增加 QC 配置指纹，避免同一个 QC 名称但不同阈值的数据被误用于推理。

测试输出中的“不可信模型拒绝”“没有保存预测时拒绝回退”等 error 文本是预期负向测试，不是未解决失败。pytest/build 工具未安装；使用既有 unittest 和 setuptools backend 完成验证，没有安装缺少的依赖。

阶段修改记录见 `docs/CHANGELOG.md`；当前演练汇总见 `tests/_artifacts/validation_summary.json`。该 JSON 仅为本次软件验证记录，不是论文研究结果。

## 3. 继承关系与与历史实现的差异

| 历史依据 | 新版模块 | 继承内容与明确差异 |
| --- | --- | --- |
| pipeline.py、plot_real_resistance_files.py 的仪器读取 | io/instrument.py、schema.py | 按表头识别时间和通道；统一 s/Ω；修复全编码失败/固定列问题；不再从文件名推定浓度 |
| pipeline.py、07_channel_quality.py 的 QC | preprocessing/quality.py | 原值、检测填补信号、异常/量程/有效掩码分开；conservative_v1 为新的按真实时间窗口 QC，不冒充旧版掩码 |
| 08_exploratory_models.py 的候选检测 | preprocessing/cycles.py | legacy08_1s_v1 限定原索引算法的 1 s 条件；time_grid_v1 显式选择检测专用重采样；所有候选未审核 |
| 08 的 max_response/resp_60s/initial_slope | features/ | 继承数值定义及列顺序，单一函数供训练/推理使用；外部审核边界取代未确认的自动资格 |
| 07 的 AUC 定义及时间缺口约束 | paper_peak_slope_auc@1.0.0 | 明确 S−1、0–120 s、单位 s、端点/缺口条件；是目标方法实现，没有继承历史模型性能 |
| 08 的五模型、Pipeline、LOGO/内层网格 | models/registry.py、training/、validation/ | 保留参数及 family 选择规则；regressor__ 统一参数命名，完整日志；可显式选外层 GroupKFold |
| 08、S16、Table 及绘图中的 metrics | evaluation/ | 统一公式、定义域/适用性、顺序检查和 scope；不保留旧版常数目标 R² 的有限替代值 |
| summarize/merge/plot 的保存结果逻辑 | visualization/render.py | 保留 parity、技术 SD、残差及指标核对；不写回预测；旧 schema 标注历史未核实版本 |
| 09_render 与旧绘图参数 | 新 scripts/09_render_exploratory.py 与 cli render | 新包装入口接受 --results-dir，并正确解析保存 CSV；同时支持 --input-csv。旧目录文件保持不变 |
| joblib 保存/特定加载片段 | prediction/artifacts.py、predict.py | 通用制品与信号推理；信任确认、环境、哈希、特征/QC/通道/顺序检查 |

审计发现与当前代码一致：旧批量脚本和旧 README 的参数确实不兼容。本次遵守“原目录保留”的约束，在新项目中修复并提供可替代入口，**没有宣称旧路径命令已被修改**。

新版没有复制早期插值特征链，也没有把 08 的“候选数符合预期/持续下降”自动资格当作人工批准。`legacy08_mask_v1` 只追溯原始有效点掩码和整通道量程排除，需显式输入存档掩码；它本身不等于精确复现整套旧候选筛选。新 legacy 特征名字表示旧特征定义，不表示新的运行结果与旧性能必然相同。

## 4. 主要接口

- `read_instrument(path, source_id=..., invalid_resistance=...)` → SignalRecord；读取不写入。
- `quality_control(t,r,QCConfig,... )` → 原值之外的 QCResult 掩码和检测信号。
- `detect_cycles(...)` → 未批准候选；不参与自动训练放行。
- `extract_features(record,channel,reviewed_boundary,version,qc_config=...)` → 特征、有序定义、Ra、响应、QC 与异常提示。
- `features_from_manifest(...)` → 显式来源、分组、审核边界及特征表；训练/预测共享。
- `evaluate_regression(...)` → 七个指标对象及模型/版本/scope/样本信息。
- `evaluate_predictions(...)` / `compare_models(...)` → 同集合模型比较和多层次评价。
- `nested_validate(...)` → OOF、逐样本划分、内层分组、全部搜索记录及折估计器。
- `refit_final(...)` → 单独全数据选择与拟合，不输出假测试性能。
- `save_artifact(...)` / `load_artifact(...,trusted=True)` → Pipeline 与严格制品契约。
- `predict_manifest(...)` / `predict_features(...)` → 同特征定义的新样本预测。
- `python -m no2_ml.cli {extract,train,predict,evaluate,render}` → 统一命令。

## 5. 特征版本与模型

| 版本 | 列顺序 | 定义 |
| --- | --- | --- |
| legacy_peak_r60_slope@1.0.0 | max_response, resp_60s, initial_slope | max(S)、60±2 s 中位响应、0–30 s OLS 斜率 |
| paper_peak_slope_auc@1.0.0 | max_response, initial_slope, auc_0_120 | max(S)、0–30 s OLS（含支持范围检查）、0–120 s 实测时间梯形积分 S−1 |

两版共用 Ra/R。Ra 为起点前最多45 s 审核基线内有效原始电阻中位数；最少20点，CV阈值0.10。斜率至少6点。AUC 的窗口/缺口和单位已在 docs 中定义。定义指纹和 QC 配置指纹同时进入制品与预测检查，列顺序不一致直接报错。论文版 AUC 的科学适用窗口仍须作者确认；更改定义应新增版本。

正式注册仅 Linear、Ridge、SVR、RandomForest、XGBoost。五类均有真实框架调用和统一 fit/predict；没有 MLP/CNN/GCN 伪实现。参数沿用经审计的旧设置并可通过 YAML 明确变更。所有测试结果只证明接口工作，不构成模型效果比较结论。

## 6. 指标、验证和推理限制

MSE、MAE、RMSE、RMSLE、R²、Adjusted R²、mean_bias_error 已实现。Bias=预测−参考；MSE 单位 ppm²，MAE/RMSE/Bias 为 ppm，其余无量纲。负 R² 保留；小样本/常数目标 R² 为 undefined；负预测的 RMSLE 为 not_applicable，无截断；Adjusted R² 仅适用于显式满足自由度与满秩条件的无惩罚 Linear 训练诊断，对普通外层评价不机械输出。

支持 outer LOGO 或显式 GroupKFold；inner GroupKFold + GridSearchCV。Imputer/Scaler 只在折内拟合；外层所有模型共享划分；内层 family 选择与最终 refit 严格分离。支持 single_test_set 标量/表评价、per_fold、fold_mean/SD、pooled_oof，不将折均值称作总体指标。独立测试**划分训练流程**、bootstrap/显著性检验未在 P0 实现。

原始记录预测使用同一 extractor 和 artifact 中的 QC。通道、单位、特征版本/指纹/顺序、QC 指纹不满足则报错；没有缺失传感器曲线的合成默认值。个别不可用特征保留提示，由训练时拟合的 Imputer 处理；全缺失行拒绝。

Peak 与审核周期依赖完整时程，制品固定标记 `offline_reviewed_cycle`，不宣称实时监测。来源哈希和分组约束不能证明物理独立性，也不能发现全部近似复制或人为错误标签。

## 7. 环境和包验证

当前实际参考环境：Python 3.9.24；NumPy 1.26.4、pandas 2.3.3、SciPy 1.13.1、scikit-learn 1.6.1、XGBoost 2.1.4、Matplotlib 3.9.4、joblib 1.5.2、PyYAML 6.0.3。既有 setuptools 80.9.0 用于构建 wheel。

通过现有构建后端生成 wheel，并用 `pip --no-index --no-deps --target tests/_build/installed_final` 仅安装本项目到隔离目标目录。确认 import 来自该目录后完成预测和兼容绘图。未改变系统 site-packages，未下载依赖。wheel 不包含测试数据、模型权重或研究结果。**这不是干净环境依赖安装验证**；其他 Python/OS/硬件兼容性未验证。

模型制品保存完整 Pipeline、模型名、特征规范与顺序、QC配置及指纹、训练配置、环境、完整有效估计器参数、哈希、UTC时间、软件版本及源码哈希。未声称存在可用 Git commit ID。加载必须来自可信来源；哈希不能认证恶意来源，joblib 仍是可执行序列化。

## 8. 可复制命令

### 安装（由使用者在自选环境执行；本次未安装这些依赖）

```powershell
cd .\NO2-ML-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m no2_ml.cli --help
```

### 测试与软件夹具演练（首次、未使用的输出目录）

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m unittest discover -s tests -t . -v
python -m tests.make_fixture
python -m no2_ml.cli train --config tests/_artifacts/experiment.yaml
python -m no2_ml.cli predict --manifest tests/_artifacts/fixture/manifest.csv --artifact tests/_artifacts/software_run/artifact --trusted-model --output tests/_artifacts/software_predictions.csv
python -m no2_ml.cli render --results-dir tests/_artifacts/software_run --output-dir tests/_artifacts/software_figures --dpi 150
```

当前工作副本已保留演练目录，重复使用同名目录会被拒绝，这是防覆盖机制。可直接运行测试；如再次演练训练，复制配置并选择**尚不存在**的 tests 下输出目录，不能把测试输出写入研究结果目录。最终验证演练使用 `tests/_artifacts/experiment_verified.yaml`、`software_run_verified/`、`software_predictions_verified.csv` 和 `software_figures_verified/`。

### 后续研究输入（本次未执行）

```powershell
python -m no2_ml.cli train --config configs/experiment.yaml
python -m no2_ml.cli predict --manifest data/new_reviewed_cycles.csv --artifact runs/first_reviewed_run/artifact --trusted-model --feature-version legacy_peak_r60_slope --output runs/new_predictions.csv
python -m no2_ml.cli evaluate --input-csv runs/first_reviewed_run/out_of_fold_predictions.csv --output runs/rechecked_metrics.csv
python -m no2_ml.cli render --input-csv runs/first_reviewed_run/out_of_fold_predictions.csv --output-dir runs/figures_new --dpi 600
```

必须先准备真实、获准使用、边界与分组经过核实的 manifest，并确认配置。模板不含数据，不会自动查找旧目录来训练。

## 9. GitHub 准备程度与尚未完成项

P0 软件候选架构已可用，当前环境单元/集成测试及包验证通过。README、五份方法文档、配置模板、.gitignore、阶段记录均已提供。默认排除原始数据、权重、演练产物、缓存、密钥及敏感配置；公开项目没有导入历史目标误差生成器或伪装成研究数据的预测记录。

仍未完成：许可证由负责人选择；作者/DOI/引用信息待确认；干净环境及跨平台 CI；真实实验接入和实际统计验证；阀门/器件/分组/参考浓度审核；独立保留测试流程、bootstrap、不确定性与实时机器人接口；Excel 导出不是 P0 必需项，目前以公共依赖 CSV 为标准。没有自动上线或发布认证。

接入真实实验前优先确认：(1) 实际使用哪套特征、AUC窗口和QC版本；(2) 器件/通道身份及更严格来源分组；(3) 审核边界、参考标签与授权；(4) 既有模型/结果的版本 provenance；(5) 冻结验证方案后再开展新版本训练，不能把旧性能赋给新方法。

## 10. 新增/修改文件清单

下列均为新项目内新增文件；不存在对旧项目源代码的修改。构建副本、egg-info、缓存和 `tests/_artifacts`/`tests/_build` 自动产物不计作维护源文件，已由 .gitignore 排除。完整源文件清单附后。

```text
.gitignore
README.md
REFACTORING_REPORT.md
configs/experiment.yaml
configs/features.yaml
configs/models.yaml
configs/preprocessing.yaml
configs/validation.yaml
docs/CHANGELOG.md
docs/LICENSE_PENDING.md
docs/cross_validation.md
docs/evaluation_metrics.md
docs/feature_definitions.md
docs/model_training.md
docs/reproducibility.md
examples/README.md
pyproject.toml
requirements.txt
scripts/09_render_exploratory.py
scripts/plot_figure5b.py
src/no2_ml/__init__.py
src/no2_ml/cli.py
src/no2_ml/config.py
src/no2_ml/evaluation/__init__.py
src/no2_ml/evaluation/aggregation.py
src/no2_ml/evaluation/export.py
src/no2_ml/evaluation/metrics.py
src/no2_ml/features/__init__.py
src/no2_ml/features/extract.py
src/no2_ml/features/response.py
src/no2_ml/features/versions.py
src/no2_ml/io/__init__.py
src/no2_ml/io/instrument.py
src/no2_ml/io/manifest.py
src/no2_ml/io/schema.py
src/no2_ml/models/__init__.py
src/no2_ml/models/registry.py
src/no2_ml/prediction/__init__.py
src/no2_ml/prediction/artifacts.py
src/no2_ml/prediction/predict.py
src/no2_ml/preprocessing/__init__.py
src/no2_ml/preprocessing/cycles.py
src/no2_ml/preprocessing/quality.py
src/no2_ml/training/__init__.py
src/no2_ml/training/nested.py
src/no2_ml/training/run.py
src/no2_ml/training/search.py
src/no2_ml/validation/__init__.py
src/no2_ml/validation/splits.py
src/no2_ml/visualization/__init__.py
src/no2_ml/visualization/render.py
tests/__init__.py
tests/fixtures.py
tests/make_fixture.py
tests/test_stage_a.py
tests/test_stage_b.py
tests/test_stage_c.py
tests/test_stage_d.py
```
