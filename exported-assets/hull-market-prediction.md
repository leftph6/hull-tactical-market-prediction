# Hull Tactical Market Prediction - 完整代码框架

## 1. 数据加载与预处理模块

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, data_path='./'):
        self.data_path = data_path
        self.train_df = None
        self.test_df = None
        self.features = None
        self.target = None
        
    def load_data(self):
        """加载训练和测试数据"""
        self.train_df = pd.read_csv(f'{self.data_path}/train.csv')
        self.test_df = pd.read_csv(f'{self.data_path}/test.csv')
        
        print(f"训练集形状: {self.train_df.shape}")
        print(f"测试集形状: {self.test_df.shape}")
        print(f"\n训练集列名:\n{self.train_df.columns.tolist()}")
        
        return self.train_df, self.test_df
    
    def basic_eda(self):
        """基础数据探索"""
        print("\n" + "="*50)
        print("数据基本信息")
        print("="*50)
        print(self.train_df.info())
        
        print("\n" + "="*50)
        print("数据统计描述")
        print("="*50)
        print(self.train_df.describe())
        
        print("\n" + "="*50)
        print("缺失值统计")
        print("="*50)
        missing = self.train_df.isnull().sum()
        print(missing[missing > 0])
        
    def prepare_features(self, target_col='target'):
        """准备特征和目标变量"""
        # 识别特征列（排除ID和目标列）
        exclude_cols = ['id', 'date', target_col]
        self.features = [col for col in self.train_df.columns 
                        if col not in exclude_cols]
        self.target = target_col
        
        print(f"\n特征数量: {len(self.features)}")
        print(f"特征列表: {self.features[:10]}...")  # 显示前10个
        
        return self.features, self.target

# 初始化数据加载器
loader = DataLoader()
train_df, test_df = loader.load_data()
loader.basic_eda()
features, target = loader.prepare_features()
```

## 2. 特征工程模块

```python
class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self, train_df, test_df, features):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.features = features
        
    def handle_missing_values(self, strategy='median'):
        """处理缺失值"""
        if strategy == 'median':
            for col in self.features:
                median_val = self.train_df[col].median()
                self.train_df[col].fillna(median_val, inplace=True)
                self.test_df[col].fillna(median_val, inplace=True)
        elif strategy == 'forward_fill':
            self.train_df[self.features] = self.train_df[self.features].fillna(method='ffill')
            self.test_df[self.features] = self.test_df[self.features].fillna(method='ffill')
        
        print(f"缺失值处理完成 (策略: {strategy})")
        return self.train_df, self.test_df
    
    def create_lag_features(self, columns, lags=[1, 2, 3, 5, 10]):
        """创建滞后特征"""
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                for lag in lags:
                    new_col = f'{col}_lag_{lag}'
                    self.train_df[new_col] = self.train_df[col].shift(lag)
                    self.test_df[new_col] = self.test_df[col].shift(lag)
                    new_features.append(new_col)
        
        print(f"创建了 {len(new_features)} 个滞后特征")
        return new_features
    
    def create_rolling_features(self, columns, windows=[5, 10, 20, 30]):
        """创建滚动窗口特征"""
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                for window in windows:
                    # 滚动均值
                    new_col_mean = f'{col}_roll_mean_{window}'
                    self.train_df[new_col_mean] = self.train_df[col].rolling(window).mean()
                    self.test_df[new_col_mean] = self.test_df[col].rolling(window).mean()
                    new_features.append(new_col_mean)
                    
                    # 滚动标准差
                    new_col_std = f'{col}_roll_std_{window}'
                    self.train_df[new_col_std] = self.train_df[col].rolling(window).std()
                    self.test_df[new_col_std] = self.test_df[col].rolling(window).std()
                    new_features.append(new_col_std)
        
        print(f"创建了 {len(new_features)} 个滚动窗口特征")
        return new_features
    
    def create_momentum_features(self, columns, periods=[5, 10, 20]):
        """创建动量特征"""
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                for period in periods:
                    new_col = f'{col}_momentum_{period}'
                    self.train_df[new_col] = self.train_df[col] - self.train_df[col].shift(period)
                    self.test_df[new_col] = self.test_df[col] - self.test_df[col].shift(period)
                    new_features.append(new_col)
        
        print(f"创建了 {len(new_features)} 个动量特征")
        return new_features
    
    def create_interaction_features(self, feature_pairs):
        """创建交互特征"""
        new_features = []
        
        for feat1, feat2 in feature_pairs:
            if feat1 in self.train_df.columns and feat2 in self.train_df.columns:
                new_col = f'{feat1}_x_{feat2}'
                self.train_df[new_col] = self.train_df[feat1] * self.train_df[feat2]
                self.test_df[new_col] = self.test_df[feat1] * self.test_df[feat2]
                new_features.append(new_col)
        
        print(f"创建了 {len(new_features)} 个交互特征")
        return new_features

# 使用特征工程
fe = FeatureEngineering(train_df, test_df, features)
train_df, test_df = fe.handle_missing_values(strategy='median')

# 可选：创建额外特征
# lag_features = fe.create_lag_features(features[:5], lags=[1, 2, 3, 5])
# rolling_features = fe.create_rolling_features(features[:5], windows=[5, 10, 20])
# momentum_features = fe.create_momentum_features(features[:5], periods=[5, 10, 20])
```

## 3. 模型构建模块

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

class ModelBuilder:
    """模型构建和训练类"""
    
    def __init__(self, train_df, features, target):
        self.train_df = train_df
        self.features = features
        self.target = target
        self.models = {}
        self.predictions = {}
        
    def prepare_data(self):
        """准备训练数据"""
        # 移除包含NaN的行
        self.train_df = self.train_df.dropna(subset=self.features + [self.target])
        
        X = self.train_df[self.features].values
        y = self.train_df[self.target].values
        
        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        return X, y
    
    def get_baseline_models(self):
        """获取基础模型集合"""
        models = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.01, random_state=42),
            'elasticnet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'lgbm': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ),
            'xgb': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                tree_method='hist'
            ),
            'catboost': CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=5,
                random_state=42,
                verbose=False
            )
        }
        return models
    
    def time_series_cv(self, X, y, n_splits=5):
        """时间序列交叉验证"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        models = self.get_baseline_models()
        cv_scores = {}
        
        print("\n" + "="*50)
        print("时间序列交叉验证")
        print("="*50)
        
        for model_name, model in models.items():
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 标准化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # 训练模型
                model.fit(X_train_scaled, y_train)
                
                # 预测
                y_pred = model.predict(X_val_scaled)
                
                # 计算RMSE
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            cv_scores[model_name] = {
                'mean_rmse': np.mean(scores),
                'std_rmse': np.std(scores),
                'scores': scores
            }
            
            print(f"{model_name:15s} - RMSE: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
        
        self.cv_scores = cv_scores
        return cv_scores
    
    def train_final_models(self, X, y):
        """训练最终模型"""
        print("\n" + "="*50)
        print("训练最终模型")
        print("="*50)
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        models = self.get_baseline_models()
        
        for model_name, model in models.items():
            print(f"训练 {model_name}...")
            model.fit(X_scaled, y)
            self.models[model_name] = model
        
        print(f"训练完成！共训练了 {len(self.models)} 个模型")
        return self.models
    
    def predict(self, test_df):
        """使用训练好的模型进行预测"""
        # 准备测试数据
        test_df = test_df.dropna(subset=self.features)
        X_test = test_df[self.features].values
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X_test_scaled)
        
        self.predictions = predictions
        return predictions

# 使用模型构建器
builder = ModelBuilder(train_df, features, target)
X, y = builder.prepare_data()

# 交叉验证
cv_scores = builder.time_series_cv(X, y, n_splits=5)

# 训练最终模型
models = builder.train_final_models(X, y)
```

## 4. 集成学习模块

```python
class EnsembleModel:
    """模型集成类"""
    
    def __init__(self, predictions_dict):
        self.predictions = predictions_dict
        self.weights = None
        
    def simple_average(self):
        """简单平均"""
        pred_array = np.array(list(self.predictions.values()))
        return np.mean(pred_array, axis=0)
    
    def weighted_average(self, weights=None):
        """加权平均"""
        if weights is None:
            # 默认权重：基于交叉验证性能
            weights = self.calculate_optimal_weights()
        
        self.weights = weights
        pred_array = np.array(list(self.predictions.values()))
        return np.average(pred_array, axis=0, weights=weights)
    
    def calculate_optimal_weights(self):
        """基于交叉验证分数计算最优权重"""
        # 这里需要cv_scores，简化处理：均等权重
        n_models = len(self.predictions)
        return np.ones(n_models) / n_models
    
    def median_ensemble(self):
        """中位数集成"""
        pred_array = np.array(list(self.predictions.values()))
        return np.median(pred_array, axis=0)
    
    def rank_average(self):
        """排名平均"""
        pred_array = np.array(list(self.predictions.values()))
        
        # 将每个模型的预测转换为排名
        ranked_preds = np.zeros_like(pred_array)
        for i in range(pred_array.shape[0]):
            ranked_preds[i] = np.argsort(np.argsort(pred_array[i]))
        
        # 平均排名
        avg_rank = np.mean(ranked_preds, axis=0)
        
        # 将排名转换回预测值（使用简单平均的尺度）
        simple_avg = self.simple_average()
        sorted_indices = np.argsort(avg_rank)
        sorted_values = np.sort(simple_avg)
        result = np.zeros_like(simple_avg)
        result[sorted_indices] = sorted_values
        
        return result

# 使用集成模型
predictions = builder.predict(test_df)
ensemble = EnsembleModel(predictions)

# 不同的集成策略
pred_simple = ensemble.simple_average()
pred_weighted = ensemble.weighted_average()
pred_median = ensemble.median_ensemble()
pred_rank = ensemble.rank_average()

print("\n集成预测完成！")
print(f"简单平均预测范围: [{pred_simple.min():.6f}, {pred_simple.max():.6f}]")
print(f"加权平均预测范围: [{pred_weighted.min():.6f}, {pred_weighted.max():.6f}]")
```

## 5. 提交文件生成模块

```python
class SubmissionGenerator:
    """生成提交文件"""
    
    def __init__(self, test_df, predictions):
        self.test_df = test_df
        self.predictions = predictions
        
    def create_submission(self, pred_values, filename='submission.csv'):
        """创建提交文件"""
        submission = pd.DataFrame({
            'id': self.test_df['id'].values[:len(pred_values)],
            'target': pred_values
        })
        
        submission.to_csv(filename, index=False)
        print(f"\n提交文件已保存: {filename}")
        print(f"提交文件形状: {submission.shape}")
        print(f"\n前5行预览:")
        print(submission.head())
        
        return submission
    
    def create_multiple_submissions(self, predictions_dict):
        """创建多个提交文件"""
        for name, pred in predictions_dict.items():
            filename = f'submission_{name}.csv'
            self.create_submission(pred, filename)

# 生成提交文件
gen = SubmissionGenerator(test_df, predictions)

# 生成不同集成策略的提交
ensemble_predictions = {
    'simple_avg': pred_simple,
    'weighted_avg': pred_weighted,
    'median': pred_median,
    'rank_avg': pred_rank
}

gen.create_multiple_submissions(ensemble_predictions)

# 也可以为单个模型生成提交
# gen.create_submission(predictions['lgbm'], 'submission_lgbm.csv')
```

## 6. 评估和分析模块

```python
class ModelAnalysis:
    """模型分析类"""
    
    def __init__(self, train_df, features, target):
        self.train_df = train_df
        self.features = features
        self.target = target
        
    def feature_importance_analysis(self, model, model_name='model', top_n=20):
        """特征重要性分析"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 6))
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.bar(range(top_n), importances[indices])
            plt.xticks(range(top_n), [self.features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # 打印重要特征
            print(f"\n{model_name} - Top {top_n} 最重要特征:")
            for i in range(top_n):
                idx = indices[i]
                print(f"{i+1}. {self.features[idx]:30s} : {importances[idx]:.6f}")
        else:
            print(f"{model_name} 不支持特征重要性分析")
    
    def plot_predictions_distribution(self, predictions_dict):
        """绘制预测分布"""
        plt.figure(figsize=(15, 10))
        
        for i, (name, pred) in enumerate(predictions_dict.items(), 1):
            plt.subplot(3, 3, i)
            plt.hist(pred, bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'{name} - 预测分布')
            plt.xlabel('预测值')
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self):
        """相关性分析"""
        # 计算特征与目标的相关性
        correlations = self.train_df[self.features + [self.target]].corr()[self.target].drop(self.target)
        correlations = correlations.sort_values(ascending=False)
        
        print("\n" + "="*50)
        print("特征与目标的相关性 (Top 20)")
        print("="*50)
        print(correlations.head(20))
        
        # 可视化
        plt.figure(figsize=(12, 8))
        correlations.head(20).plot(kind='barh')
        plt.title('Top 20 特征与目标的相关性')
        plt.xlabel('相关系数')
        plt.tight_layout()
        plt.savefig('feature_correlation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return correlations

# 使用分析模块
analyzer = ModelAnalysis(train_df, features, target)

# 特征重要性分析（以LightGBM为例）
if 'lgbm' in models:
    analyzer.feature_importance_analysis(models['lgbm'], 'LightGBM', top_n=20)

# 预测分布
all_predictions = {**predictions, **ensemble_predictions}
analyzer.plot_predictions_distribution(all_predictions)

# 相关性分析
correlations = analyzer.correlation_analysis()
```

## 7. 超参数调优模块

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

class HyperparameterTuning:
    """超参数调优类"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_params = {}
        
    def tune_lgbm(self, n_iter=20):
        """调优LightGBM"""
        param_dist = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'num_leaves': randint(20, 100),
            'min_child_samples': randint(10, 50),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
        
        lgbm = LGBMRegressor(random_state=42, verbose=-1)
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        random_search = RandomizedSearchCV(
            lgbm, param_dist, n_iter=n_iter, 
            cv=tscv, scoring='neg_mean_squared_error',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        print("\n开始LightGBM超参数调优...")
        random_search.fit(self.X, self.y)
        
        self.best_params['lgbm'] = random_search.best_params_
        print(f"\nLightGBM最佳参数:")
        print(random_search.best_params_)
        print(f"最佳RMSE: {np.sqrt(-random_search.best_score_):.6f}")
        
        return random_search.best_estimator_
    
    def tune_xgb(self, n_iter=20):
        """调优XGBoost"""
        param_dist = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2)
        }
        
        xgb = XGBRegressor(random_state=42, tree_method='hist')
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        random_search = RandomizedSearchCV(
            xgb, param_dist, n_iter=n_iter,
            cv=tscv, scoring='neg_mean_squared_error',
            random_state=42, n_jobs=-1, verbose=1
        )
        
        print("\n开始XGBoost超参数调优...")
        random_search.fit(self.X, self.y)
        
        self.best_params['xgb'] = random_search.best_params_
        print(f"\nXGBoost最佳参数:")
        print(random_search.best_params_)
        print(f"最佳RMSE: {np.sqrt(-random_search.best_score_):.6f}")
        
        return random_search.best_estimator_

# 超参数调优示例（可选，耗时较长）
# tuner = HyperparameterTuning(X, y)
# best_lgbm = tuner.tune_lgbm(n_iter=20)
# best_xgb = tuner.tune_xgb(n_iter=20)
```

## 8. 完整工作流程

```python
def main_pipeline():
    """完整的工作流程"""
    
    print("="*70)
    print("Hull Tactical Market Prediction - 完整流程")
    print("="*70)
    
    # 1. 数据加载
    print("\n步骤 1: 数据加载")
    print("-"*70)
    loader = DataLoader()
    train_df, test_df = loader.load_data()
    loader.basic_eda()
    features, target = loader.prepare_features()
    
    # 2. 特征工程
    print("\n步骤 2: 特征工程")
    print("-"*70)
    fe = FeatureEngineering(train_df, test_df, features)
    train_df, test_df = fe.handle_missing_values(strategy='median')
    
    # 可选：创建额外特征
    # new_features = []
    # new_features += fe.create_lag_features(features[:5], lags=[1, 2, 3])
    # new_features += fe.create_rolling_features(features[:5], windows=[5, 10, 20])
    # features = features + new_features
    
    # 3. 模型训练
    print("\n步骤 3: 模型训练")
    print("-"*70)
    builder = ModelBuilder(train_df, features, target)
    X, y = builder.prepare_data()
    
    # 交叉验证
    cv_scores = builder.time_series_cv(X, y, n_splits=5)
    
    # 训练最终模型
    models = builder.train_final_models(X, y)
    
    # 4. 预测
    print("\n步骤 4: 生成预测")
    print("-"*70)
    predictions = builder.predict(test_df)
    
    # 5. 集成
    print("\n步骤 5: 模型集成")
    print("-"*70)
    ensemble = EnsembleModel(predictions)
    pred_simple = ensemble.simple_average()
    pred_weighted = ensemble.weighted_average()
    pred_median = ensemble.median_ensemble()
    
    # 6. 生成提交
    print("\n步骤 6: 生成提交文件")
    print("-"*70)
    gen = SubmissionGenerator(test_df, predictions)
    
    ensemble_predictions = {
        'simple_avg': pred_simple,
        'weighted_avg': pred_weighted,
        'median': pred_median
    }
    
    gen.create_multiple_submissions(ensemble_predictions)
    
    # 7. 分析
    print("\n步骤 7: 模型分析")
    print("-"*70)
    analyzer = ModelAnalysis(train_df, features, target)
    
    if 'lgbm' in models:
        analyzer.feature_importance_analysis(models['lgbm'], 'LightGBM', top_n=20)
    
    all_predictions = {**predictions, **ensemble_predictions}
    analyzer.plot_predictions_distribution(all_predictions)
    
    print("\n" + "="*70)
    print("流程完成！")
    print("="*70)

# 运行完整流程
if __name__ == '__main__':
    main_pipeline()
```

## 9. 调参建议

### 快速调参清单

#### 数据预处理
- [ ] 尝试不同的缺失值填充策略（median, mean, forward_fill）
- [ ] 尝试不同的特征标准化方法（StandardScaler, RobustScaler, MinMaxScaler）
- [ ] 处理异常值（IQR方法、Z-score）

#### 特征工程
- [ ] 调整滞后特征的滞后期数：`lags=[1,2,3,5,10,20,30]`
- [ ] 调整滚动窗口大小：`windows=[3,5,7,10,15,20,30,60]`
- [ ] 创建更多动量特征和波动率特征
- [ ] 尝试特征交互（多项式特征、比率特征）
- [ ] 特征选择（基于重要性、相关性、递归特征消除）

#### 模型超参数（LightGBM - 推荐重点调优）
```python
lgbm_params = {
    'n_estimators': [100, 200, 300, 500],      # 树的数量
    'learning_rate': [0.01, 0.05, 0.1, 0.2],   # 学习率
    'max_depth': [3, 5, 7, 10],                 # 树的深度
    'num_leaves': [15, 31, 63, 127],            # 叶子节点数
    'min_child_samples': [10, 20, 30, 50],      # 最小样本数
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],     # 样本采样比例
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # 特征采样比例
    'reg_alpha': [0, 0.1, 0.5, 1.0],            # L1正则化
    'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0]       # L2正则化
}
```

#### 模型超参数（XGBoost）
```python
xgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.5, 1.0, 2.0]
}
```

#### 集成策略
- [ ] 调整集成权重（基于CV分数）
- [ ] 尝试Stacking（使用线性模型作为元学习器）
- [ ] 尝试Blending（在验证集上训练元模型）
- [ ] 多层Stacking

#### 交叉验证
- [ ] 调整交叉验证折数：`n_splits=[3, 5, 10]`
- [ ] 尝试不同的验证策略（KFold, StratifiedKFold, PurgedKFold）
- [ ] 注意避免数据泄漏（时间序列特性）

### 改进优先级（推荐顺序）

1. **高优先级**（最可能提升性能）
   - 特征工程：滞后特征、滚动统计
   - LightGBM调参：learning_rate, n_estimators, max_depth
   - 集成策略：加权平均、Stacking

2. **中优先级**
   - 异常值处理
   - 特征选择
   - XGBoost调参
   - 不同的缺失值填充策略

3. **低优先级**（可能提升较小）
   - 复杂的特征交互
   - 深度学习模型
   - 极端的过采样/欠采样

## 10. 常见问题和注意事项

### 时间序列数据处理
- ⚠️ 避免未来信息泄漏（不要使用未来的数据预测过去）
- ✅ 使用TimeSeriesSplit进行交叉验证
- ✅ 按时间顺序创建训练/验证集

### 评估指标
- 本比赛使用的评估指标需要查看比赛官方说明
- 可能是RMSE、MAE或自定义指标
- 确保本地验证指标与线上一致

### 提交格式
- 确保ID列与test.csv完全匹配
- 检查预测值范围是否合理
- 避免NaN和Inf值

### 性能优化
- 使用`n_jobs=-1`并行计算
- 对于大数据集，考虑增量学习
- 使用GPU加速（XGBoost, LightGBM）

---

## 使用说明

1. **安装依赖**
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

2. **数据准备**
- 将train.csv和test.csv放在工作目录
- 或修改`DataLoader`中的路径

3. **运行流程**
```python
# 完整运行
main_pipeline()

# 或分步运行
# ... （使用上面的各个模块）
```

4. **调参建议**
- 先用默认参数跑通流程
- 分析特征重要性
- 重点调优最重要的几个模型
- 尝试不同的集成策略

祝比赛顺利！🚀
