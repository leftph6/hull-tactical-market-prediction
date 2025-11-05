"""
完整测试脚本 - 使用input目录中的test.csv数据测试完整流程
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
    print("完整测试 - Hull Tactical Market Prediction")
    print("="*70)
    
    # 1. 准备数据
    print("\n步骤 1: 准备数据")
    print("-"*70)
    data_raw_dir = Path('./data/raw')
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    input_test = Path('./input/test.csv')
    data_test = data_raw_dir / 'test.csv'
    
    if not data_test.exists() and input_test.exists():
        import shutil
        shutil.copy(input_test, data_test)
        print(f"  ✓ 已复制数据文件到 {data_test}")
    
    # 读取测试数据
    test_df = pd.read_csv(data_test)
    print(f"  原始测试数据形状: {test_df.shape}")
    print(f"  列数: {len(test_df.columns)}")
    
    # 2. 准备训练和测试数据
    print("\n步骤 2: 准备训练和测试数据集")
    print("-"*70)
    
    # 如果有目标变量,将数据分为训练集和测试集
    if 'lagged_forward_returns' in test_df.columns:
        # 使用前80%作为训练集,后20%作为测试集
        split_idx = int(len(test_df) * 0.8)
        train_df = test_df.iloc[:split_idx].copy()
        test_df_split = test_df.iloc[split_idx:].copy()
        
        # 重命名目标变量为target(如果不存在)
        if 'target' not in train_df.columns:
            train_df['target'] = train_df['lagged_forward_returns']
            # 测试集中删除目标变量(模拟真实预测场景)
            test_df_split = test_df_split.drop(columns=['lagged_forward_returns'], errors='ignore')
        
        print(f"  训练集形状: {train_df.shape}")
        print(f"  测试集形状: {test_df_split.shape}")
        print(f"  目标变量: target")
        
        # 保存临时训练数据
        train_file = data_raw_dir / 'train.csv'
        train_df.to_csv(train_file, index=False)
        test_df_split.to_csv(data_test, index=False)
        print(f"  ✓ 已保存训练和测试数据")
    else:
        print("  ⚠ 未找到目标变量,跳过训练测试")
        return
    
    # 3. 使用DataLoader加载数据
    print("\n步骤 3: 使用DataLoader加载数据")
    print("-"*70)
    try:
        loader = DataLoader(data_path='./data/raw/')
        train_df_loaded, test_df_loaded = loader.load_data()
        print(f"  ✓ 数据加载成功")
        print(f"    训练集: {train_df_loaded.shape}")
        print(f"    测试集: {test_df_loaded.shape}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 数据探索
    print("\n步骤 4: 数据探索分析")
    print("-"*70)
    try:
        loader.basic_eda(verbose=True)
    except Exception as e:
        print(f"  ⚠ 数据探索遇到问题: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 准备特征
    print("\n步骤 5: 准备特征")
    print("-"*70)
    try:
        features, target = loader.prepare_features(target_col='target')
        print(f"  ✓ 特征准备完成")
        print(f"    原始特征数量: {len(features)}")
        print(f"    目标变量: {target}")
        
        # 只保留训练集和测试集都存在的特征
        common_features = [f for f in features if f in train_df_loaded.columns and f in test_df_loaded.columns]
        print(f"    共同特征数量: {len(common_features)}")
        
        if len(common_features) < len(features):
            removed = set(features) - set(common_features)
            print(f"    移除的特征(仅在训练集存在): {list(removed)}")
        
        features = common_features
        
        # 显示部分特征名
        if len(features) <= 20:
            print(f"    特征列表: {features}")
        else:
            print(f"    前10个特征: {features[:10]}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 特征工程
    print("\n步骤 6: 特征工程")
    print("-"*70)
    try:
        fe = FeatureEngineering(train_df_loaded, test_df_loaded, features)
        
        # 处理缺失值
        print("  6.1 处理缺失值...")
        train_df_processed, test_df_processed = fe.handle_missing_values(strategy='median')
        print(f"    ✓ 缺失值处理完成")
        
        # 创建滞后特征(使用前5个数值特征)
        print("  6.2 创建滞后特征...")
        numeric_features = [f for f in features if train_df_processed[f].dtype in ['float64', 'int64']]
        if len(numeric_features) >= 5:
            fe.create_lag_features(numeric_features[:5], lags=[1, 2, 3])
            print(f"    ✓ 滞后特征创建完成")
        
        # 创建滚动窗口特征
        print("  6.3 创建滚动窗口特征...")
        if len(numeric_features) >= 5:
            fe.create_rolling_features(numeric_features[:5], windows=[5, 10], stats=['mean', 'std'])
            print(f"    ✓ 滚动窗口特征创建完成")
        
        # 创建动量特征
        print("  6.4 创建动量特征...")
        if len(numeric_features) >= 3:
            fe.create_momentum_features(numeric_features[:3], periods=[5, 10])
            print(f"    ✓ 动量特征创建完成")
        
        # 获取所有特征
        all_features = fe.get_all_features()
        print(f"\n  📊 最终特征统计:")
        print(f"    原始特征: {len(features)}")
        print(f"    新建特征: {len(all_features) - len(features)}")
        print(f"    总特征数: {len(all_features)}")
        
        # 显示处理后的数据形状
        print(f"\n  处理后数据形状:")
        print(f"    训练集: {train_df_processed.shape}")
        print(f"    测试集: {test_df_processed.shape}")
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 数据质量检查
    print("\n步骤 7: 数据质量检查")
    print("-"*70)
    try:
        loader.check_data_quality()
    except Exception as e:
        print(f"  ⚠ 数据质量检查遇到问题: {e}")
    
    # 8. 特征类型分析
    print("\n步骤 8: 特征类型分析")
    print("-"*70)
    try:
        feature_types = loader.get_feature_types()
        print(f"  ✓ 特征类型分析完成")
    except Exception as e:
        print(f"  ⚠ 特征类型分析遇到问题: {e}")
    
    print("\n" + "="*70)
    print("✓ 完整测试运行成功!")
    print("="*70)
    print("\n总结:")
    print(f"  - 成功加载数据: {train_df_loaded.shape[0]} 训练样本, {test_df_loaded.shape[0]} 测试样本")
    print(f"  - 特征工程完成: {len(all_features)} 个特征")
    print(f"  - 数据已准备好进行模型训练")


if __name__ == '__main__':
    main()

