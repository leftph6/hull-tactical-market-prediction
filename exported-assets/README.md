# Hull Tactical Market Prediction - 市场预测项目

基于机器学习的市场预测项目，使用时间序列特征工程和集成学习模型进行市场趋势预测。

## 📋 目录

- [项目简介](#项目简介)
- [文件架构](#文件架构)
- [快速开始](#快速开始)
- [部署指南](#部署指南)
- [使用方法](#使用方法)
- [模块说明](#模块说明)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 📖 项目简介

本项目是一个完整的机器学习预测框架，包含：

- **数据处理**: 自动化的数据加载、清洗和探索性分析
- **特征工程**: 滞后特征、滚动窗口特征、动量特征等
- **模型训练**: 支持多种梯度提升模型（LightGBM、XGBoost、CatBoost）
- **集成学习**: 多种集成策略（加权平均、简单平均、中位数等）
- **自动化流程**: 一键运行完整的数据处理到预测流程

---

## 📁 文件架构

### 当前项目结构

```
exported-assets/
├── data/                          # 数据目录
│   └── raw/                       # 原始数据
│       ├── train.csv              # 训练数据（如有）
│       └── test.csv               # 测试数据
│
├── input/                         # 输入数据目录
│   └── test.csv                   # 原始测试数据
│
├── ouput/                         # 输出目录（预留）
│
├── config.yaml                    # 主配置文件
├── requirements.txt               # Python依赖列表
│
├── data_loader.py                 # 数据加载模块
├── feature_engineering.py         # 特征工程模块
│
├── run_test.py                    # 基础测试脚本
├── run_full_test.py               # 完整测试脚本
│
├── deployment-guide.md            # 详细部署指南
├── quick-start.md                 # 快速开始指南
├── project-structure.md           # 项目结构说明
└── hull-market-prediction.md      # 项目文档
```

### 理想项目结构（完整版）

```
hull-tactical-prediction/
├── data/
│   ├── raw/                       # 原始数据
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/                 # 处理后的数据
│
├── src/                           # 源代码目录（可选）
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_builder.py
│   ├── ensemble.py
│   └── evaluator.py
│
├── outputs/                       # 输出目录
│   ├── submissions/               # 提交文件
│   ├── models/                    # 保存的模型
│   ├── figures/                   # 图表
│   └── logs/                      # 日志文件
│
├── config/
│   └── config.yaml                # 配置文件
│
├── notebooks/                     # Jupyter notebooks（可选）
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering.ipynb
│
├── scripts/                        # 脚本目录（可选）
│   ├── train.py
│   └── predict.py
│
├── requirements.txt               # 依赖列表
├── README.md                      # 项目说明（本文件）
└── run_pipeline.py                # 主运行脚本（待实现）
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.8 或更高版本
- **内存**: 建议 8GB 以上
- **操作系统**: Windows / Linux / macOS

### 2. 安装依赖

```bash
# 方法1: 使用 requirements.txt（推荐）
pip install -r requirements.txt

# 方法2: 手动安装核心依赖
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn pyyaml joblib
```

### 3. 准备数据

将数据文件放在 `input/` 目录下：

```bash
# 确保数据文件存在
input/
  └── test.csv
```

### 4. 运行测试

```bash
# 运行完整测试（推荐）
python run_full_test.py

# 或运行基础测试
python run_test.py
```

---

## 📦 部署指南

### Windows 部署

```powershell
# 1. 创建虚拟环境（可选但推荐）
python -m venv venv
venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据目录
mkdir -p data\raw
# 将数据文件复制到 data\raw\ 或 input\ 目录

# 4. 运行测试
python run_full_test.py
```

### Linux / macOS 部署

```bash
# 1. 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据目录
mkdir -p data/raw
# 将数据文件复制到 data/raw/ 或 input/ 目录

# 4. 运行测试
python run_full_test.py
```

### 使用 Conda 部署

```bash
# 1. 创建conda环境
conda create -n hull-prediction python=3.9
conda activate hull-prediction

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行测试
python run_full_test.py
```

---

## 💻 使用方法

### 方法1: 使用测试脚本（推荐新手）

```bash
# 完整流程测试
python run_full_test.py
```

这个脚本会：
1. 自动准备数据目录
2. 加载数据
3. 进行数据探索
4. 执行特征工程
5. 检查数据质量

### 方法2: 在Python代码中使用

```python
from data_loader import DataLoader
from feature_engineering import FeatureEngineering

# 1. 加载数据
loader = DataLoader(data_path='./data/raw/')
train_df, test_df = loader.load_data()

# 2. 数据探索
loader.basic_eda()
features, target = loader.prepare_features()

# 3. 特征工程
fe = FeatureEngineering(train_df, test_df, features)
train_df, test_df = fe.handle_missing_values(strategy='median')

# 4. 创建特征
fe.create_lag_features(features[:5], lags=[1, 2, 3])
fe.create_rolling_features(features[:5], windows=[5, 10])
all_features = fe.get_all_features()
```

### 方法3: 使用配置文件

编辑 `config.yaml` 来定制特征工程和模型参数：

```yaml
feature_engineering:
  missing_strategy: 'median'
  create_lag_features: true
  lag_periods: [1, 2, 3, 5, 10]
```

---

## 🔧 模块说明

### 1. data_loader.py

**功能**: 数据加载和预处理

**主要方法**:
- `load_data()`: 加载训练和测试数据
- `basic_eda()`: 基础数据探索分析
- `prepare_features()`: 准备特征和目标变量
- `check_data_quality()`: 数据质量检查
- `get_feature_types()`: 识别特征类型

**使用示例**:
```python
loader = DataLoader(data_path='./data/raw/')
train_df, test_df = loader.load_data()
loader.basic_eda()
features, target = loader.prepare_features()
```

### 2. feature_engineering.py

**功能**: 特征工程和特征创建

**主要方法**:
- `handle_missing_values()`: 处理缺失值
- `create_lag_features()`: 创建滞后特征
- `create_rolling_features()`: 创建滚动窗口特征
- `create_momentum_features()`: 创建动量特征
- `create_volatility_features()`: 创建波动率特征
- `get_all_features()`: 获取所有特征

**使用示例**:
```python
fe = FeatureEngineering(train_df, test_df, features)
fe.handle_missing_values(strategy='median')
fe.create_lag_features(features[:5], lags=[1, 2, 3])
all_features = fe.get_all_features()
```

---

## ⚙️ 配置说明

### config.yaml 配置文件

主要配置项：

```yaml
# 数据配置
data:
  raw_path: './data/raw/'
  processed_path: './data/processed/'
  train_file: 'train.csv'
  test_file: 'test.csv'

# 特征工程配置
feature_engineering:
  missing_strategy: 'median'  # median, mean, forward_fill, zero
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
```

详细配置说明请参考 `config.yaml` 文件中的注释。

---

## ❓ 常见问题

### Q1: 找不到数据文件

**问题**: `FileNotFoundError: 找不到数据文件`

**解决**:
1. 确保数据文件在 `input/` 或 `data/raw/` 目录下
2. 检查文件名是否正确（`test.csv`, `train.csv`）
3. 修改 `config.yaml` 中的路径配置

### Q2: 模块导入错误

**问题**: `ModuleNotFoundError: No module named 'data_loader'`

**解决**:
```bash
# 确保在项目根目录运行
cd exported-assets
python run_full_test.py

# 或添加当前目录到Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Q3: 依赖安装失败

**问题**: LightGBM 或 XGBoost 安装失败

**解决**:
```bash
# Windows: 使用conda安装
conda install -c conda-forge lightgbm xgboost

# Linux: 安装编译工具
sudo apt-get install build-essential
pip install lightgbm xgboost

# 或使用预编译版本
pip install --upgrade pip
pip install lightgbm xgboost --no-build-isolation
```

### Q4: 特征工程报错

**问题**: `KeyError: 'column_name'`

**解决**:
- 确保特征列在训练集和测试集中都存在
- 检查列名是否正确（注意大小写）
- 使用 `run_full_test.py` 会自动处理共同特征

### Q5: 内存不足

**问题**: 处理大数据时内存不足

**解决**:
1. 减少特征数量
2. 使用分批处理
3. 增加系统内存
4. 使用 `dtype` 优化（如 `float32` 代替 `float64`）

---

## 📚 相关文档

- **快速开始**: 查看 `quick-start.md`
- **详细部署**: 查看 `deployment-guide.md`
- **项目结构**: 查看 `project-structure.md`
- **完整文档**: 查看 `hull-market-prediction.md`

---

## 🔄 开发计划

### 已完成 ✅
- [x] 数据加载模块 (`data_loader.py`)
- [x] 特征工程模块 (`feature_engineering.py`)
- [x] 基础测试脚本 (`run_test.py`)
- [x] 完整测试脚本 (`run_full_test.py`)
- [x] 配置文件 (`config.yaml`)

### 待实现 🚧
- [ ] 模型构建模块 (`model_builder.py`)
- [ ] 集成学习模块 (`ensemble.py`)
- [ ] 评估分析模块 (`evaluator.py`)
- [ ] 主运行脚本 (`run_pipeline.py`)
- [ ] 超参数调优模块 (`tuner.py`)

---

## 📝 许可证

本项目仅供学习和研究使用。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 查看项目文档

---

**最后更新**: 2024年

**版本**: 1.0.0

