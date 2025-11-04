# Hull Tactical Market Prediction - 完整部署指南

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- Python 3.8 或更高版本
- 至少 8GB RAM
- （可选）NVIDIA GPU 用于加速训练

#### 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n hull-prediction python=3.9
conda activate hull-prediction

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
# 克隆或创建项目目录
mkdir hull-tactical-prediction
cd hull-tactical-prediction

# 安装所有依赖
pip install -r requirements.txt

# 或手动安装核心依赖
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn pyyaml joblib
```

**requirements.txt 内容：**
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
catboost>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
joblib>=1.1.0
scipy>=1.7.0
tqdm>=4.62.0
```

### 3. 项目初始化

```bash
# 创建目录结构
python setup.py init

# 或手动创建
mkdir -p data/raw data/processed src outputs/{submissions,models,figures,logs} config notebooks scripts tests
```

### 4. 数据准备

```bash
# 将 Kaggle 数据集下载到 data/raw/
# 方法1: 使用 Kaggle API
kaggle competitions download -c hull-tactical-market-prediction -p data/raw/
unzip data/raw/hull-tactical-market-prediction.zip -d data/raw/

# 方法2: 手动下载
# 访问 https://www.kaggle.com/competitions/hull-tactical-market-prediction/data
# 下载 train.csv 和 test.csv 到 data/raw/
```

### 5. 配置文件设置

创建 `config/config.yaml`:

```yaml
# 数据路径
data:
  raw_path: './data/raw/'
  processed_path: './data/processed/'
  train_file: 'train.csv'
  test_file: 'test.csv'

# 特征工程配置
feature_engineering:
  missing_strategy: 'median'  # median, mean, forward_fill
  create_lag_features: true
  lag_periods: [1, 2, 3, 5, 10, 20]
  create_rolling_features: true
  rolling_windows: [5, 10, 20, 30, 60]
  create_momentum_features: true
  momentum_periods: [5, 10, 20]
  
# 模型配置
model:
  cv_splits: 5
  random_state: 42
  models_to_train: ['ridge', 'lasso', 'lgbm', 'xgb', 'catboost']
  
# 集成配置
ensemble:
  strategy: 'weighted_average'  # simple_average, weighted_average, median, rank_average
  
# 输出配置
output:
  save_models: true
  save_predictions: true
  create_figures: true
  
# 日志配置
logging:
  level: 'INFO'
  file: './outputs/logs/training.log'
```

## 📊 使用流程

### 方式 1: 使用主脚本（推荐）

```bash
# 完整流程：数据加载 -> 特征工程 -> 训练 -> 预测 -> 提交
python run_pipeline.py --config config/config.yaml

# 指定特定步骤
python run_pipeline.py --steps data,feature,train
python run_pipeline.py --steps predict,submit

# 快速测试（使用默认参数）
python run_pipeline.py --quick
```

### 方式 2: 分步执行

```bash
# 1. 数据探索
python scripts/eda.py

# 2. 训练模型
python scripts/train.py --config config/config.yaml

# 3. 生成预测
python scripts/predict.py --model-path outputs/models/best_model.pkl

# 4. 创建提交文件
python scripts/submit.py --predictions outputs/predictions.csv
```

### 方式 3: 使用 Jupyter Notebook

```bash
jupyter notebook

# 依次运行：
# notebooks/01_eda.ipynb
# notebooks/02_feature_engineering.ipynb
# notebooks/03_model_training.ipynb
```

### 方式 4: Python 交互式使用

```python
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineering
from src.model_builder import ModelBuilder
from src.ensemble import EnsembleModel

# 加载数据
loader = DataLoader(data_path='./data/raw/')
train_df, test_df = loader.load_data()
features, target = loader.prepare_features()

# 特征工程
fe = FeatureEngineering(train_df, test_df, features)
train_df, test_df = fe.handle_missing_values()

# 训练模型
builder = ModelBuilder(train_df, features, target)
X, y = builder.prepare_data()
models = builder.train_final_models(X, y)

# 预测和集成
predictions = builder.predict(test_df)
ensemble = EnsembleModel(predictions)
final_pred = ensemble.weighted_average()
```

## 🎯 调优指南

### 第一阶段：基线建立（1-2小时）

**目标：**建立可工作的基线模型

```bash
# 1. 运行默认配置
python run_pipeline.py --quick

# 2. 查看交叉验证结果
cat outputs/logs/cv_scores.txt

# 3. 提交基线结果
# 选择表现最好的模型提交到 Kaggle
```

**预期结果：**
- 获得初始 leaderboard 分数
- 了解数据特性
- 识别问题和改进方向

---

### 第二阶段：特征工程（2-4小时）

**优先级：⭐⭐⭐⭐⭐**

#### 2.1 滞后特征优化

修改 `config/config.yaml`:

```yaml
feature_engineering:
  create_lag_features: true
  lag_periods: [1, 2, 3, 5, 7, 10, 14, 20, 30]  # 增加更多滞后期
```

运行：
```bash
python run_pipeline.py --steps feature,train --config config/config.yaml
```

**调参建议：**
- 短期滞后 (1-5)：捕捉近期趋势
- 中期滞后 (7-20)：捕捉周期性
- 长期滞后 (30+)：捕捉长期趋势

#### 2.2 滚动窗口特征

```yaml
feature_engineering:
  create_rolling_features: true
  rolling_windows: [3, 5, 7, 10, 15, 20, 30, 60, 90]
  rolling_stats: ['mean', 'std', 'min', 'max']  # 增加更多统计量
```

**调参建议：**
- 尝试不同窗口大小
- 添加更多统计量（中位数、分位数、偏度）
- 计算多个特征的滚动统计

#### 2.3 自定义特征创建

编辑 `src/feature_engineering.py`，添加：

```python
def create_custom_features(self):
    """创建自定义金融特征"""
    # 波动率特征
    for window in [5, 10, 20]:
        self.train_df[f'volatility_{window}'] = (
            self.train_df['close'].rolling(window).std()
        )
    
    # 价格变化率
    for period in [1, 5, 10]:
        self.train_df[f'return_{period}'] = (
            self.train_df['close'].pct_change(period)
        )
    
    # 技术指标
    # RSI, MACD, Bollinger Bands 等
    ...
```

**预期提升：**0.5-2% 性能改进

---

### 第三阶段：模型调优（3-6小时）

**优先级：⭐⭐⭐⭐**

#### 3.1 LightGBM 调优（推荐重点）

创建 `config/lgbm_params.yaml`:

```yaml
# 第一轮：粗调
lgbm_round1:
  n_estimators: [100, 200, 300, 500]
  learning_rate: [0.01, 0.05, 0.1]
  max_depth: [3, 5, 7]
  num_leaves: [31, 63, 127]

# 第二轮：精调（基于第一轮最佳参数）
lgbm_round2:
  n_estimators: [400, 500, 600]      # 围绕最佳值
  learning_rate: [0.08, 0.1, 0.12]   # 围绕最佳值
  max_depth: [6, 7, 8]               # 围绕最佳值
  num_leaves: [50, 63, 80]           # 围绕最佳值
  min_child_samples: [10, 20, 30]
  subsample: [0.7, 0.8, 0.9]
  colsample_bytree: [0.7, 0.8, 0.9]
  reg_alpha: [0, 0.1, 0.5]
  reg_lambda: [0, 0.5, 1.0]
```

运行调优：

```bash
# 粗调（快速）
python scripts/tune_model.py --model lgbm --params config/lgbm_params.yaml --round 1 --n-iter 20

# 精调（基于粗调结果）
python scripts/tune_model.py --model lgbm --params config/lgbm_params.yaml --round 2 --n-iter 50
```

**时间成本：**
- 粗调：30-60分钟
- 精调：2-3小时

**预期提升：**1-3% 性能改进

#### 3.2 XGBoost 调优

```bash
python scripts/tune_model.py --model xgb --n-iter 30
```

#### 3.3 多模型对比

```python
# 在 scripts/train.py 中
python scripts/train.py --models lgbm,xgb,catboost,rf --compare
```

查看对比结果：
```bash
cat outputs/logs/model_comparison.txt
```

---

### 第四阶段：集成优化（1-2小时）

**优先级：⭐⭐⭐⭐**

#### 4.1 基于CV分数的加权集成

编辑 `src/ensemble.py`:

```python
def calculate_optimal_weights(self, cv_scores):
    """基于CV分数计算权重"""
    # 使用 RMSE 的倒数作为权重
    scores = np.array([cv_scores[model]['mean_rmse'] for model in self.predictions.keys()])
    weights = 1.0 / scores
    weights = weights / weights.sum()
    return weights
```

运行：
```bash
python scripts/ensemble.py --strategy weighted --use-cv-scores
```

#### 4.2 Stacking 集成

```python
# 使用 scripts/stacking.py
python scripts/stacking.py --base-models lgbm,xgb,catboost --meta-model ridge
```

#### 4.3 多层集成

```bash
# Level 1: 多个模型
# Level 2: 集成 Level 1 的预测
python scripts/multi_level_ensemble.py
```

**预期提升：**0.5-1.5% 性能改进

---

### 第五阶段：高级优化（2-4小时）

**优先级：⭐⭐⭐**

#### 5.1 特征选择

```bash
# 基于重要性的特征选择
python scripts/feature_selection.py --method importance --top-k 100

# 递归特征消除
python scripts/feature_selection.py --method rfe --n-features 50

# 相关性过滤
python scripts/feature_selection.py --method correlation --threshold 0.95
```

#### 5.2 交叉验证策略优化

```yaml
# config/config.yaml
model:
  cv_strategy: 'time_series'  # time_series, kfold, purged
  cv_splits: 10  # 增加折数以获得更稳定的验证
  purge_gap: 5   # 对于 purged CV
```

#### 5.3 数据增强

```python
# 在 src/feature_engineering.py 中添加
def augment_data(self):
    """数据增强"""
    # 添加噪声
    # 时间序列bootstrap
    # SMOTE（如果适用）
    ...
```

---

### 第六阶段：最终优化（1-2小时）

**优先级：⭐⭐**

#### 6.1 超参数微调

基于前面的最佳模型，进行最后的微调：

```bash
python scripts/final_tune.py --model best_lgbm --fine-tune --n-iter 100
```

#### 6.2 模型融合

```bash
# 融合多个最佳模型的预测
python scripts/blend_models.py --models model1.pkl,model2.pkl,model3.pkl --weights 0.4,0.4,0.2
```

#### 6.3 后处理优化

```python
# 在 scripts/postprocess.py 中
def postprocess_predictions(predictions):
    """预测后处理"""
    # 剪裁异常值
    predictions = np.clip(predictions, lower_bound, upper_bound)
    
    # 平滑处理
    predictions = smooth_predictions(predictions, window=3)
    
    return predictions
```

---

## 📈 调优监控

### 跟踪实验结果

创建实验日志：

```python
# experiments_log.csv
experiment_id,date,features,model,params,cv_score,lb_score,notes
exp_001,2025-11-04,baseline,lgbm,default,0.0156,0.0162,baseline
exp_002,2025-11-04,+lag_features,lgbm,default,0.0149,0.0155,added lag 1-10
exp_003,2025-11-04,+lag_features,lgbm,tuned,0.0142,0.0148,tuned lgbm
...
```

### 使用实验追踪工具

```bash
# 安装 MLflow
pip install mlflow

# 启动 MLflow UI
mlflow ui

# 在训练脚本中记录实验
python scripts/train.py --use-mlflow
```

### 可视化改进

```python
import matplotlib.pyplot as plt
import pandas as pd

# 读取实验日志
df = pd.read_csv('experiments_log.csv')

# 绘制进度曲线
plt.figure(figsize=(12, 6))
plt.plot(df['experiment_id'], df['cv_score'], marker='o', label='CV Score')
plt.plot(df['experiment_id'], df['lb_score'], marker='s', label='LB Score')
plt.xlabel('Experiment')
plt.ylabel('Score (RMSE)')
plt.title('Model Performance Over Experiments')
plt.legend()
plt.grid(True)
plt.savefig('outputs/figures/progress.png')
```

---

## 🔍 调试技巧

### 常见问题排查

#### 1. 本地CV与线上LB不一致

**可能原因：**
- 数据泄漏（特征工程使用了未来信息）
- CV划分方式不当
- 过拟合

**解决方案：**
```python
# 检查数据泄漏
python scripts/check_leakage.py

# 使用更严格的CV
cv = TimeSeriesSplit(n_splits=10)

# 增加正则化
lgbm_params['reg_alpha'] = 1.0
lgbm_params['reg_lambda'] = 1.0
```

#### 2. 训练速度太慢

**优化方案：**
```python
# 1. 使用GPU
lgbm_params['device'] = 'gpu'
xgb_params['tree_method'] = 'gpu_hist'

# 2. 减少特征数量
python scripts/feature_selection.py --top-k 50

# 3. 使用更少的CV折数
cv_splits = 3

# 4. 并行训练
n_jobs = -1
```

#### 3. 内存不足

**解决方案：**
```python
# 使用数据类型优化
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                # ... 其他类型
    return df

# 分批处理
chunk_size = 10000
for chunk in pd.read_csv('train.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

---

## 🎓 最佳实践

### 1. 版本控制

```bash
git init
git add .
git commit -m "Initial commit"

# 为每次重要改进创建分支
git checkout -b feature/lag-features
git checkout -b experiment/lgbm-tuning
```

### 2. 代码复用

将成功的特征工程和模型配置保存为模板：

```python
# templates/successful_features.py
BEST_LAG_CONFIG = {
    'periods': [1, 2, 3, 5, 10, 20],
    'features': ['feature_1', 'feature_2', ...]
}

BEST_ROLLING_CONFIG = {
    'windows': [5, 10, 20, 30],
    'stats': ['mean', 'std', 'min', 'max']
}
```

### 3. 自动化测试

```python
# tests/test_features.py
def test_no_future_leakage():
    """确保特征不包含未来信息"""
    fe = FeatureEngineering(train_df, test_df, features)
    fe.create_lag_features(columns, lags=[1, 2, 3])
    
    # 检查每个滞后特征
    for col in fe.train_df.columns:
        if 'lag' in col:
            assert not has_future_info(fe.train_df[col])

def test_no_data_leakage():
    """确保train和test没有信息泄漏"""
    assert len(set(train_df.index) & set(test_df.index)) == 0
```

运行测试：
```bash
pytest tests/
```

### 4. 文档记录

为每个实验记录详细信息：

```markdown
## Experiment 015 - 2025-11-04

### 改动
- 添加了波动率特征 (5, 10, 20日窗口)
- LightGBM: learning_rate=0.08, n_estimators=500

### 结果
- CV RMSE: 0.0142 (↓ 0.0007)
- LB RMSE: 0.0148 (↓ 0.0005)

### 分析
- 波动率特征贡献度较高 (feature importance top 5)
- 验证集和测试集表现一致，未过拟合

### 下一步
- 尝试添加更多技术指标
- 进一步调优 num_leaves 参数
```

---

## 📊 性能指标

### 预期改进路径

| 阶段 | 操作 | CV提升 | LB提升 | 时间投入 |
|------|------|--------|--------|----------|
| 基线 | 默认配置 | 0.0200 | 0.0205 | 1h |
| 特征工程 | 滞后+滚动特征 | -0.0030 | -0.0025 | 3h |
| 模型调优 | LightGBM调参 | -0.0020 | -0.0018 | 4h |
| 集成优化 | 加权集成 | -0.0015 | -0.0012 | 2h |
| 高级优化 | 特征选择+Stacking | -0.0010 | -0.0008 | 3h |
| **总计** | | **0.0125** | **0.0142** | **13h** |

---

## 🛠️ 故障排除

### 环境问题

```bash
# 问题：LightGBM安装失败
# 解决：
conda install -c conda-forge lightgbm

# 问题：XGBoost GPU不可用
# 解决：
pip install xgboost-gpu

# 问题：依赖冲突
# 解决：
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### 数据问题

```bash
# 检查数据完整性
python scripts/validate_data.py

# 修复损坏的CSV
python scripts/fix_csv.py
```

---

## 📞 获取帮助

### 资源链接

- **Kaggle Competition**: https://www.kaggle.com/competitions/hull-tactical-market-prediction
- **Discussion Forum**: https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion
- **LightGBM文档**: https://lightgbm.readthedocs.io/
- **XGBoost文档**: https://xgboost.readthedocs.io/

### 社区支持

- Kaggle Discussion 发帖提问
- GitHub Issues（如果代码有问题）

---

## ✅ 检查清单

部署前检查：
- [ ] 安装所有依赖
- [ ] 数据文件在正确位置
- [ ] 配置文件已创建
- [ ] 目录结构完整
- [ ] 可以运行 `python run_pipeline.py --quick`

提交前检查：
- [ ] 提交文件格式正确（id, target两列）
- [ ] 没有NaN或Inf值
- [ ] ID与test.csv完全匹配
- [ ] 预测值范围合理
- [ ] 已在本地验证

---

## 🎯 成功路线图

**第1天（4小时）**
- ✅ 环境搭建和数据加载
- ✅ 基线模型建立
- ✅ 首次提交

**第2-3天（8小时）**
- 🎨 特征工程迭代
- 🔧 LightGBM调优
- 📊 交叉验证优化

**第4-5天（8小时）**
- 🤝 集成学习
- 🎯 特征选择
- 🚀 最终优化

**第6-7天（4小时）**
- 📈 实验分析
- 🏆 提交最佳模型
- 📝 总结文档

**总时间投入：24小时**
**预期排名：Top 10-20%**

---

祝你在比赛中取得优异成绩！🏆

如果遇到任何问题，请参考故障排除部分或在 Discussion 中提问。
