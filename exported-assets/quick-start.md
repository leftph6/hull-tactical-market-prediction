# 🚀 快速开始指南

## 10分钟快速部署

### 步骤 1: 创建项目结构 (1分钟)

```bash
# 创建项目目录
mkdir hull-tactical-prediction
cd hull-tactical-prediction

# 创建子目录
mkdir -p data/raw data/processed src outputs/{submissions,models,figures,logs} config scripts
```

### 步骤 2: 安装依赖 (2分钟)

```bash
# 创建并激活虚拟环境
conda create -n hull python=3.9 -y
conda activate hull

# 安装依赖
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn pyyaml joblib
```

### 步骤 3: 下载数据 (2分钟)

**方法 1: 使用 Kaggle API**
```bash
pip install kaggle
kaggle competitions download -c hull-tactical-market-prediction -p data/raw/
unzip data/raw/hull-tactical-market-prediction.zip -d data/raw/
```

**方法 2: 手动下载**
1. 访问: https://www.kaggle.com/competitions/hull-tactical-market-prediction/data
2. 下载 `train.csv` 和 `test.csv`
3. 放到 `data/raw/` 目录

### 步骤 4: 复制代码文件 (2分钟)

将以下文件保存到对应位置：

```
src/
  ├── data_loader.py          # 已提供
  ├── feature_engineering.py  # 已提供
  └── __init__.py            # 空文件

config/
  └── config.yaml            # 已提供

requirements.txt             # 已提供
run_pipeline.py             # 已提供
```

创建 `src/__init__.py`:
```bash
touch src/__init__.py
```

### 步骤 5: 运行基线模型 (3分钟)

```bash
# 快速测试（仅数据加载和特征工程）
python run_pipeline.py --quick

# 完整流程（需要等待实现完整的模型模块）
python run_pipeline.py --steps data,feature
```

---

## 📂 文件清单

确保你已经创建/下载了以下文件：

### ✅ 必需文件

- [ ] `src/data_loader.py` - 数据加载模块
- [ ] `src/feature_engineering.py` - 特征工程模块
- [ ] `config/config.yaml` - 配置文件
- [ ] `requirements.txt` - 依赖列表
- [ ] `run_pipeline.py` - 主运行脚本
- [ ] `data/raw/train.csv` - 训练数据
- [ ] `data/raw/test.csv` - 测试数据

### 📋 待实现文件

这些文件在原始框架文档中有完整代码，需要你复制到项目中：

- [ ] `src/model_builder.py` - 模型构建模块
- [ ] `src/ensemble.py` - 集成学习模块
- [ ] `src/evaluator.py` - 评估分析模块
- [ ] `src/tuner.py` - 超参数调优模块

---

## 🎯 第一次运行检查

### 测试数据加载

```python
# 在项目根目录运行 Python
python

>>> from src.data_loader import DataLoader
>>> loader = DataLoader(data_path='./data/raw/')
>>> train_df, test_df = loader.load_data()
>>> print(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")
```

**预期输出:**
```
✓ 数据加载成功
  训练集形状: (XXXX, YY)
  测试集形状: (ZZZZ, YY)
```

### 测试特征工程

```python
>>> from src.feature_engineering import FeatureEngineering
>>> features, target = loader.prepare_features()
>>> fe = FeatureEngineering(train_df, test_df, features)
>>> train_df, test_df = fe.handle_missing_values()
```

**预期输出:**
```
✓ 特征准备完成
  特征数量: XX
  
🔧 处理缺失值 (策略: median)
  ✓ 缺失值处理完成
```

---

## ⚡ 常见问题快速解决

### 问题 1: ModuleNotFoundError

```bash
# 错误: No module named 'src'
# 解决:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# 或在运行脚本时
python -m run_pipeline --quick
```

### 问题 2: FileNotFoundError

```bash
# 错误: 找不到 train.csv
# 解决: 确认文件位置
ls data/raw/

# 如果文件在其他位置，修改 config.yaml
vim config/config.yaml
# 修改 data.raw_path 为正确路径
```

### 问题 3: 依赖安装失败

```bash
# LightGBM 安装失败
conda install -c conda-forge lightgbm

# XGBoost GPU 版本
pip install xgboost-gpu

# 依赖冲突
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 📊 验证安装成功

运行以下命令，应该都能成功执行：

```bash
# 1. 检查 Python 版本
python --version
# 期望: Python 3.8+

# 2. 检查依赖
python -c "import pandas, numpy, sklearn, lightgbm, xgboost; print('✓ 所有依赖已安装')"

# 3. 检查数据文件
ls -lh data/raw/
# 应该看到 train.csv 和 test.csv

# 4. 测试运行
python run_pipeline.py --quick
# 应该能成功加载数据并运行特征工程
```

---

## 🎓 下一步

完成快速部署后，按以下顺序进行：

1. **运行完整EDA** (10分钟)
   ```python
   from src.data_loader import DataLoader
   loader = DataLoader()
   train_df, test_df = loader.load_data()
   loader.basic_eda()
   loader.check_data_quality()
   ```

2. **实现基线模型** (30分钟)
   - 复制 `model_builder.py` 到 `src/`
   - 运行基线训练
   - 生成第一次提交

3. **特征工程迭代** (2-4小时)
   - 创建滞后特征
   - 创建滚动统计特征
   - 分析特征重要性

4. **模型调优** (3-6小时)
   - LightGBM 参数调优
   - 交叉验证优化
   - 集成学习

5. **持续改进**
   - 跟踪实验结果
   - 分析错误案例
   - 迭代优化

---

## 💡 实用技巧

### 快速创建所有目录

```bash
# 一键创建完整目录结构
mkdir -p hull-tactical-prediction/{data/{raw,processed},src,outputs/{submissions,models,figures,logs},config,notebooks,scripts,tests}
```

### 使用别名简化命令

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias hull-train="python run_pipeline.py --steps train"
alias hull-predict="python run_pipeline.py --steps predict"
alias hull-full="python run_pipeline.py --steps all"
```

### Jupyter Notebook 快速启动

```bash
# 启动 Jupyter
jupyter notebook

# 在 notebook 中
import sys
sys.path.append('../src')
from data_loader import DataLoader
# ... 开始分析
```

---

## ✅ 部署完成检查清单

- [ ] Python 3.8+ 已安装
- [ ] 虚拟环境已创建并激活
- [ ] 所有依赖包已安装
- [ ] 项目目录结构已创建
- [ ] 数据文件在 `data/raw/` 目录
- [ ] 代码文件在 `src/` 目录
- [ ] 配置文件在 `config/` 目录
- [ ] 能成功运行 `python run_pipeline.py --quick`
- [ ] 能成功导入 `src.data_loader` 和 `src.feature_engineering`

**全部勾选完成？恭喜你，可以开始比赛了！🎉**

---

## 📞 获取帮助

遇到问题？

1. 查看完整部署指南: `deployment-guide.md`
2. 查看原始代码框架: `hull-market-prediction.md`
3. 在 Kaggle Discussion 发帖提问
4. 检查项目结构: `project-structure.md`

祝你在比赛中取得好成绩！🏆
