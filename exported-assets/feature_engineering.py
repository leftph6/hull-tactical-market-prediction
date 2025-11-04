"""
src/feature_engineering.py
特征工程模块
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self, train_df, test_df, features):
        """
        初始化特征工程器
        
        Args:
            train_df: 训练数据框
            test_df: 测试数据框  
            features: 特征列表
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.features = features
        self.new_features = []
        
    def handle_missing_values(self, strategy='median'):
        """
        处理缺失值
        
        Args:
            strategy: 填充策略 ('median', 'mean', 'forward_fill', 'zero')
            
        Returns:
            处理后的训练和测试数据
        """
        print(f"\n🔧 处理缺失值 (策略: {strategy})")
        
        for col in self.features:
            train_missing = self.train_df[col].isnull().sum()
            test_missing = self.test_df[col].isnull().sum()
            
            if train_missing > 0 or test_missing > 0:
                if strategy == 'median':
                    fill_value = self.train_df[col].median()
                elif strategy == 'mean':
                    fill_value = self.train_df[col].mean()
                elif strategy == 'zero':
                    fill_value = 0
                elif strategy == 'forward_fill':
                    self.train_df[col] = self.train_df[col].fillna(method='ffill')
                    self.test_df[col] = self.test_df[col].fillna(method='ffill')
                    continue
                else:
                    raise ValueError(f"未知的填充策略: {strategy}")
                
                self.train_df[col].fillna(fill_value, inplace=True)
                self.test_df[col].fillna(fill_value, inplace=True)
        
        print(f"  ✓ 缺失值处理完成")
        return self.train_df, self.test_df
    
    def create_lag_features(self, columns: List[str], lags: List[int] = [1, 2, 3, 5, 10]):
        """
        创建滞后特征
        
        Args:
            columns: 要创建滞后特征的列
            lags: 滞后期数列表
            
        Returns:
            新特征名列表
        """
        print(f"\n🔧 创建滞后特征")
        print(f"  列数: {len(columns)}, 滞后期: {lags}")
        
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                for lag in lags:
                    new_col = f'{col}_lag_{lag}'
                    self.train_df[new_col] = self.train_df[col].shift(lag)
                    self.test_df[new_col] = self.test_df[col].shift(lag)
                    new_features.append(new_col)
        
        self.new_features.extend(new_features)
        print(f"  ✓ 创建了 {len(new_features)} 个滞后特征")
        return new_features
    
    def create_rolling_features(self, columns: List[str], 
                               windows: List[int] = [5, 10, 20, 30],
                               stats: List[str] = ['mean', 'std']):
        """
        创建滚动窗口特征
        
        Args:
            columns: 要创建滚动特征的列
            windows: 窗口大小列表
            stats: 统计量列表 ('mean', 'std', 'min', 'max', 'median')
            
        Returns:
            新特征名列表
        """
        print(f"\n🔧 创建滚动窗口特征")
        print(f"  列数: {len(columns)}, 窗口: {windows}, 统计: {stats}")
        
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                for window in windows:
                    for stat in stats:
                        new_col = f'{col}_roll_{stat}_{window}'
                        
                        if stat == 'mean':
                            self.train_df[new_col] = self.train_df[col].rolling(window).mean()
                            self.test_df[new_col] = self.test_df[col].rolling(window).mean()
                        elif stat == 'std':
                            self.train_df[new_col] = self.train_df[col].rolling(window).std()
                            self.test_df[new_col] = self.test_df[col].rolling(window).std()
                        elif stat == 'min':
                            self.train_df[new_col] = self.train_df[col].rolling(window).min()
                            self.test_df[new_col] = self.test_df[col].rolling(window).min()
                        elif stat == 'max':
                            self.train_df[new_col] = self.train_df[col].rolling(window).max()
                            self.test_df[new_col] = self.test_df[col].rolling(window).max()
                        elif stat == 'median':
                            self.train_df[new_col] = self.train_df[col].rolling(window).median()
                            self.test_df[new_col] = self.test_df[col].rolling(window).median()
                        
                        new_features.append(new_col)
        
        self.new_features.extend(new_features)
        print(f"  ✓ 创建了 {len(new_features)} 个滚动窗口特征")
        return new_features
    
    def create_momentum_features(self, columns: List[str], periods: List[int] = [5, 10, 20]):
        """
        创建动量特征（价格变化）
        
        Args:
            columns: 要创建动量特征的列
            periods: 时间周期列表
            
        Returns:
            新特征名列表
        """
        print(f"\n🔧 创建动量特征")
        print(f"  列数: {len(columns)}, 周期: {periods}")
        
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                for period in periods:
                    # 绝对动量
                    new_col = f'{col}_momentum_{period}'
                    self.train_df[new_col] = self.train_df[col] - self.train_df[col].shift(period)
                    self.test_df[new_col] = self.test_df[col] - self.test_df[col].shift(period)
                    new_features.append(new_col)
                    
                    # 相对动量（百分比变化）
                    new_col_pct = f'{col}_momentum_pct_{period}'
                    self.train_df[new_col_pct] = self.train_df[col].pct_change(period)
                    self.test_df[new_col_pct] = self.test_df[col].pct_change(period)
                    new_features.append(new_col_pct)
        
        self.new_features.extend(new_features)
        print(f"  ✓ 创建了 {len(new_features)} 个动量特征")
        return new_features
    
    def create_interaction_features(self, feature_pairs: List[tuple]):
        """
        创建交互特征
        
        Args:
            feature_pairs: 特征对列表 [(feat1, feat2), ...]
            
        Returns:
            新特征名列表
        """
        print(f"\n🔧 创建交互特征")
        print(f"  特征对数量: {len(feature_pairs)}")
        
        new_features = []
        
        for feat1, feat2 in feature_pairs:
            if feat1 in self.train_df.columns and feat2 in self.train_df.columns:
                # 乘积
                new_col_mult = f'{feat1}_x_{feat2}'
                self.train_df[new_col_mult] = self.train_df[feat1] * self.train_df[feat2]
                self.test_df[new_col_mult] = self.test_df[feat1] * self.test_df[feat2]
                new_features.append(new_col_mult)
                
                # 比率（避免除零）
                new_col_ratio = f'{feat1}_div_{feat2}'
                self.train_df[new_col_ratio] = self.train_df[feat1] / (self.train_df[feat2] + 1e-5)
                self.test_df[new_col_ratio] = self.test_df[feat1] / (self.test_df[feat2] + 1e-5)
                new_features.append(new_col_ratio)
        
        self.new_features.extend(new_features)
        print(f"  ✓ 创建了 {len(new_features)} 个交互特征")
        return new_features
    
    def create_diff_features(self, columns: List[str], periods: List[int] = [1, 2]):
        """
        创建差分特征
        
        Args:
            columns: 要差分的列
            periods: 差分阶数
            
        Returns:
            新特征名列表
        """
        print(f"\n🔧 创建差分特征")
        
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                for period in periods:
                    new_col = f'{col}_diff_{period}'
                    self.train_df[new_col] = self.train_df[col].diff(period)
                    self.test_df[new_col] = self.test_df[col].diff(period)
                    new_features.append(new_col)
        
        self.new_features.extend(new_features)
        print(f"  ✓ 创建了 {len(new_features)} 个差分特征")
        return new_features
    
    def create_volatility_features(self, columns: List[str], windows: List[int] = [5, 10, 20]):
        """
        创建波动率特征
        
        Args:
            columns: 要计算波动率的列
            windows: 窗口大小列表
            
        Returns:
            新特征名列表
        """
        print(f"\n🔧 创建波动率特征")
        
        new_features = []
        
        for col in columns:
            if col in self.train_df.columns:
                # 先计算收益率
                returns_col = f'{col}_returns'
                self.train_df[returns_col] = self.train_df[col].pct_change()
                self.test_df[returns_col] = self.test_df[col].pct_change()
                
                # 计算滚动波动率
                for window in windows:
                    new_col = f'{col}_volatility_{window}'
                    self.train_df[new_col] = self.train_df[returns_col].rolling(window).std()
                    self.test_df[new_col] = self.test_df[returns_col].rolling(window).std()
                    new_features.append(new_col)
        
        self.new_features.extend(new_features)
        print(f"  ✓ 创建了 {len(new_features)} 个波动率特征")
        return new_features
    
    def get_all_features(self):
        """
        获取所有特征（原始+新建）
        
        Returns:
            所有特征列表
        """
        all_features = self.features + self.new_features
        print(f"\n📊 特征统计:")
        print(f"  原始特征: {len(self.features)}")
        print(f"  新建特征: {len(self.new_features)}")
        print(f"  总特征数: {len(all_features)}")
        return all_features
    
    def remove_low_variance_features(self, threshold=0.01):
        """
        移除低方差特征
        
        Args:
            threshold: 方差阈值
            
        Returns:
            保留的特征列表
        """
        print(f"\n🔧 移除低方差特征 (阈值: {threshold})")
        
        all_features = self.get_all_features()
        variances = self.train_df[all_features].var()
        low_var_features = variances[variances < threshold].index.tolist()
        
        if low_var_features:
            print(f"  移除 {len(low_var_features)} 个低方差特征")
            self.train_df.drop(columns=low_var_features, inplace=True)
            self.test_df.drop(columns=low_var_features, inplace=True)
            
            # 更新特征列表
            self.features = [f for f in self.features if f not in low_var_features]
            self.new_features = [f for f in self.new_features if f not in low_var_features]
        else:
            print(f"  无低方差特征需要移除")
        
        return self.get_all_features()


# 使用示例
if __name__ == '__main__':
    from data_loader import DataLoader
    
    # 加载数据
    loader = DataLoader()
    train_df, test_df = loader.load_data()
    features, target = loader.prepare_features()
    
    # 特征工程
    fe = FeatureEngineering(train_df, test_df, features)
    
    # 处理缺失值
    train_df, test_df = fe.handle_missing_values(strategy='median')
    
    # 创建各种特征
    fe.create_lag_features(features[:5], lags=[1, 2, 3, 5, 10])
    fe.create_rolling_features(features[:5], windows=[5, 10, 20])
    fe.create_momentum_features(features[:5], periods=[5, 10, 20])
    
    # 获取所有特征
    all_features = fe.get_all_features()
    print(f"\n最终特征数量: {len(all_features)}")
