"""
测试运行脚本 - 使用input目录中的test.csv数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 导入数据加载和特征工程模块
from data_loader import DataLoader
from feature_engineering import FeatureEngineering

def main():
    """主测试函数"""
    print("="*70)
    print("测试运行 - Hull Tactical Market Prediction")
    print("="*70)
    
    # 1. 检查并准备数据目录
    print("\n步骤 1: 准备数据目录")
    print("-"*70)
    data_raw_dir = Path('./data/raw')
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果test.csv不在data/raw目录,从input目录复制
    input_test = Path('./input/test.csv')
    data_test = data_raw_dir / 'test.csv'
    
    if not data_test.exists() and input_test.exists():
        print(f"  从 {input_test} 复制到 {data_test}")
        import shutil
        shutil.copy(input_test, data_test)
        print(f"  ✓ 数据文件已准备")
    elif data_test.exists():
        print(f"  ✓ 数据文件已存在: {data_test}")
    else:
        print(f"  ✗ 错误: 找不到数据文件 {input_test}")
        return
    
    # 2. 加载数据
    print("\n步骤 2: 加载数据")
    print("-"*70)
    try:
        # 由于只有test.csv,我们将其作为测试数据加载
        loader = DataLoader(data_path='./data/raw/')
        
        # 检查是否有train.csv,如果没有就只用test.csv
        train_file = data_raw_dir / 'train.csv'
        if train_file.exists():
            train_df, test_df = loader.load_data()
            print(f"  训练集形状: {train_df.shape}")
            print(f"  测试集形状: {test_df.shape}")
        else:
            # 只有test.csv,将其作为测试数据
            print("  注意: 未找到train.csv,仅加载test.csv作为测试数据")
            test_df = pd.read_csv(data_test)
            print(f"  测试集形状: {test_df.shape}")
            print(f"  测试集列数: {len(test_df.columns)}")
            
            # 显示基本信息
            print("\n  测试集基本信息:")
            print(f"    列名: {list(test_df.columns[:10])}...")
            print(f"    缺失值: {test_df.isnull().sum().sum()}")
            print(f"    数据类型: {test_df.dtypes.value_counts().to_dict()}")
            
            # 检查是否有目标变量
            if 'lagged_forward_returns' in test_df.columns:
                print(f"    找到目标变量: lagged_forward_returns")
                print(f"    目标变量统计: 均值={test_df['lagged_forward_returns'].mean():.6f}, "
                      f"标准差={test_df['lagged_forward_returns'].std():.6f}")
            
            return test_df
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 数据探索
    print("\n步骤 3: 数据探索")
    print("-"*70)
    try:
        loader.basic_eda(verbose=True)
    except Exception as e:
        print(f"  ⚠ 数据探索遇到问题: {e}")
    
    # 4. 准备特征
    print("\n步骤 4: 准备特征")
    print("-"*70)
    try:
        # 尝试准备特征,如果test.csv没有target列,使用lagged_forward_returns作为目标
        if 'target' not in train_df.columns and 'lagged_forward_returns' in train_df.columns:
            features, target = loader.prepare_features(target_col='lagged_forward_returns')
        else:
            features, target = loader.prepare_features()
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 特征工程测试
    print("\n步骤 5: 特征工程测试")
    print("-"*70)
    try:
        fe = FeatureEngineering(train_df, test_df, features)
        
        # 处理缺失值
        train_df_processed, test_df_processed = fe.handle_missing_values(strategy='median')
        print(f"  ✓ 缺失值处理完成")
        
        # 显示特征统计
        all_features = fe.get_all_features()
        print(f"\n  特征统计:")
        print(f"    原始特征数: {len(features)}")
        print(f"    新建特征数: {len(fe.features) - len(features) if len(fe.features) > len(features) else 0}")
        print(f"    总特征数: {len(all_features)}")
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("✓ 测试运行完成!")
    print("="*70)


if __name__ == '__main__':
    main()


