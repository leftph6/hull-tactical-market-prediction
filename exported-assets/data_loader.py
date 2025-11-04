"""
src/data_loader.py
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, data_path='./data/raw/'):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = Path(data_path)
        self.train_df = None
        self.test_df = None
        self.features = None
        self.target = None
        
    def load_data(self, train_file='train.csv', test_file='test.csv'):
        """
        加载训练和测试数据
        
        Args:
            train_file: 训练文件名
            test_file: 测试文件名
            
        Returns:
            train_df, test_df: 训练和测试数据框
        """
        try:
            self.train_df = pd.read_csv(self.data_path / train_file)
            self.test_df = pd.read_csv(self.data_path / test_file)
            
            print(f"✓ 数据加载成功")
            print(f"  训练集形状: {self.train_df.shape}")
            print(f"  测试集形状: {self.test_df.shape}")
            
            return self.train_df, self.test_df
            
        except FileNotFoundError as e:
            print(f"✗ 错误: 找不到数据文件")
            print(f"  请确保 {train_file} 和 {test_file} 在 {self.data_path} 目录下")
            raise e
    
    def basic_eda(self, verbose=True):
        """
        基础数据探索分析
        
        Args:
            verbose: 是否打印详细信息
        """
        if self.train_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        if verbose:
            print("\n" + "="*60)
            print("📊 数据基本信息")
            print("="*60)
            
            print("\n1. 数据维度")
            print(f"   训练集: {self.train_df.shape}")
            print(f"   测试集: {self.test_df.shape}")
            
            print("\n2. 数据类型")
            print(self.train_df.dtypes.value_counts())
            
            print("\n3. 缺失值统计")
            missing = self.train_df.isnull().sum()
            missing_pct = 100 * missing / len(self.train_df)
            missing_table = pd.DataFrame({
                '缺失数量': missing,
                '缺失百分比': missing_pct
            })
            missing_table = missing_table[missing_table['缺失数量'] > 0].sort_values(
                '缺失数量', ascending=False
            )
            
            if len(missing_table) > 0:
                print(missing_table.head(10))
            else:
                print("   无缺失值")
            
            print("\n4. 统计描述")
            print(self.train_df.describe().T)
        
        return {
            'missing': missing,
            'stats': self.train_df.describe()
        }
    
    def prepare_features(self, target_col='target', exclude_cols=None):
        """
        准备特征和目标变量
        
        Args:
            target_col: 目标变量列名
            exclude_cols: 要排除的列名列表
            
        Returns:
            features: 特征列表
            target: 目标变量名
        """
        if self.train_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        # 默认排除的列
        default_exclude = ['id', 'date', target_col]
        if exclude_cols:
            default_exclude.extend(exclude_cols)
        
        # 识别特征列
        self.features = [col for col in self.train_df.columns 
                        if col not in default_exclude]
        self.target = target_col
        
        print(f"\n✓ 特征准备完成")
        print(f"  特征数量: {len(self.features)}")
        print(f"  目标变量: {self.target}")
        
        if len(self.features) <= 20:
            print(f"  特征列表: {self.features}")
        else:
            print(f"  前10个特征: {self.features[:10]}")
            print(f"  后10个特征: {self.features[-10:]}")
        
        return self.features, self.target
    
    def get_feature_types(self):
        """
        识别特征类型
        
        Returns:
            dict: 包含数值型和类别型特征的字典
        """
        if self.features is None:
            raise ValueError("请先调用 prepare_features()")
        
        numeric_features = self.train_df[self.features].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        categorical_features = self.train_df[self.features].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        feature_types = {
            'numeric': numeric_features,
            'categorical': categorical_features
        }
        
        print(f"\n特征类型分析:")
        print(f"  数值型特征: {len(numeric_features)}")
        print(f"  类别型特征: {len(categorical_features)}")
        
        return feature_types
    
    def check_data_quality(self):
        """
        数据质量检查
        
        Returns:
            dict: 质量检查结果
        """
        if self.train_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        results = {}
        
        # 检查重复行
        duplicates = self.train_df.duplicated().sum()
        results['duplicates'] = duplicates
        
        # 检查无穷大值
        inf_counts = {}
        for col in self.train_df.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(self.train_df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        results['infinity_values'] = inf_counts
        
        # 检查常数特征
        constant_features = []
        for col in self.features if self.features else self.train_df.columns:
            if self.train_df[col].nunique() == 1:
                constant_features.append(col)
        results['constant_features'] = constant_features
        
        print("\n" + "="*60)
        print("🔍 数据质量检查")
        print("="*60)
        print(f"重复行数: {duplicates}")
        print(f"无穷大值: {len(inf_counts)} 列")
        print(f"常数特征: {len(constant_features)} 列")
        
        if constant_features:
            print(f"  常数特征列表: {constant_features}")
        
        return results
    
    def save_processed_data(self, output_path='./data/processed/'):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.train_df.to_csv(output_path / 'train_processed.csv', index=False)
        self.test_df.to_csv(output_path / 'test_processed.csv', index=False)
        
        print(f"\n✓ 数据已保存到 {output_path}")


# 使用示例
if __name__ == '__main__':
    # 初始化
    loader = DataLoader(data_path='./data/raw/')
    
    # 加载数据
    train_df, test_df = loader.load_data()
    
    # 数据探索
    loader.basic_eda()
    
    # 准备特征
    features, target = loader.prepare_features()
    
    # 质量检查
    loader.check_data_quality()
    
    # 特征类型分析
    feature_types = loader.get_feature_types()
