import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import sys
import datetime
from scipy import stats

# 添加父目录到系统路径，以便导入模型模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.time_series_model import HybridTemporalModel, TimeSeriesDataset, train_model, evaluate_model, plot_results

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 处理异常值
def handle_outliers(df, column, method='zscore', threshold=3):
    """处理异常值"""
    sales_values = df[column].copy()
    
    if method == 'zscore':
        # 使用Z分数检测异常值
        z_scores = stats.zscore(sales_values)
        outliers = np.abs(z_scores) > threshold
        
        # 打印异常值信息
        outlier_indices = np.where(outliers)[0]
        print(f"检测到 {len(outlier_indices)} 个异常值 (Z分数法)")
        for i in outlier_indices:
            print(f"日期: {df.iloc[i, 0]}, 销售量: {df.iloc[i, 1]}, Z分数: {z_scores[i]:.2f}")
        
        # 使用中位数替换异常值
        median_value = np.median(sales_values)
        sales_values[outliers] = median_value
        
    elif method == 'iqr':
        # 使用IQR检测异常值
        Q1 = np.percentile(sales_values, 25)
        Q3 = np.percentile(sales_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (sales_values < lower_bound) | (sales_values > upper_bound)
        
        # 打印异常值信息
        outlier_indices = np.where(outliers)[0]
        print(f"检测到 {len(outlier_indices)} 个异常值 (IQR法)")
        for i in outlier_indices:
            print(f"日期: {df.iloc[i, 0]}, 销售量: {df.iloc[i, 1]}")
        
        # 异常值替换为边界值
        sales_values[sales_values < lower_bound] = lower_bound
        sales_values[sales_values > upper_bound] = upper_bound
    
    elif method == 'cap':
        # 使用百分位数上下限截断
        lower_percentile = np.percentile(sales_values, 5)
        upper_percentile = np.percentile(sales_values, 95)
        
        outliers = (sales_values < lower_percentile) | (sales_values > upper_percentile)
        
        # 打印异常值信息
        outlier_indices = np.where(outliers)[0]
        print(f"检测到 {len(outlier_indices)} 个异常值 (百分位截断法)")
        
        # 截断异常值
        sales_values[sales_values < lower_percentile] = lower_percentile
        sales_values[sales_values > upper_percentile] = upper_percentile
    
    elif method == 'log':
        # 对数变换
        print("应用对数变换以减小极端值的影响")
        # 添加一个小值以处理可能的零值
        sales_values = np.log1p(sales_values)
    
    return sales_values

# 特征工程
def feature_engineering(df):
    """添加时间特征和统计特征"""
    df_copy = df.copy()
    
    # 转换日期列为datetime类型
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    
    # 提取时间特征
    df_copy['year'] = df_copy['日期'].dt.year
    df_copy['month'] = df_copy['日期'].dt.month
    df_copy['day'] = df_copy['日期'].dt.day
    df_copy['dayofweek'] = df_copy['日期'].dt.dayofweek  # 0=周一, 6=周日
    df_copy['quarter'] = df_copy['日期'].dt.quarter
    df_copy['is_weekend'] = df_copy['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 是否为特殊月份（如双11、双12）
    df_copy['is_special_month'] = df_copy['month'].apply(lambda x: 1 if x in [11, 12, 1] else 0)
    
    # 添加滞后特征
    for lag in [1, 3, 7, 14]:
        df_copy[f'lag_{lag}'] = df_copy['成交商品件数'].shift(lag)
    
    # 添加滚动统计特征
    for window in [3, 7, 14]:
        df_copy[f'rolling_mean_{window}'] = df_copy['成交商品件数'].rolling(window=window).mean()
        df_copy[f'rolling_std_{window}'] = df_copy['成交商品件数'].rolling(window=window).std()
        df_copy[f'rolling_max_{window}'] = df_copy['成交商品件数'].rolling(window=window).max()
        df_copy[f'rolling_min_{window}'] = df_copy['成交商品件数'].rolling(window=window).min()
    
    # 计算环比变化率
    df_copy['pct_change'] = df_copy['成交商品件数'].pct_change()
    
    # 填充NaN值
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

# 创建时间序列数据的滑动窗口（多特征版本）
def create_multivariate_sequences(data, seq_length, target_col='成交商品件数'):
    """创建多变量时间序列的滑动窗口"""
    feature_cols = [col for col in data.columns if col != '日期']
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # 获取所有特征的序列窗口
        features_window = data.iloc[i:(i+seq_length)][feature_cols].values
        # 获取目标值（下一时间步的销售量）
        target = data.iloc[i+seq_length][target_col]
        
        X.append(features_window)
        y.append(target)
    
    return np.array(X), np.array(y).reshape(-1, 1)

def main():
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确保目录存在
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_save_path = os.path.join(parent_dir, 'best_model.pth')
    
    # 加载数据
    file_path = '../total.csv'
    df = load_data(file_path)
    
    # 数据预处理
    print(f"原始数据形状: {df.shape}")
    
    # 提取销售数据
    sales_column = '成交商品件数'
    
    # 可视化原始数据
    plt.figure(figsize=(12, 6))
    plt.plot(df[sales_column].values)
    plt.title("原始销售数据")
    plt.xlabel("时间")
    plt.ylabel("销售量")
    plt.tight_layout()
    plt.savefig('original_data.png')
    plt.close()
    
    # 处理异常值
    df[sales_column] = handle_outliers(df, sales_column, method='cap')
    
    # 可视化处理后的数据
    plt.figure(figsize=(12, 6))
    plt.plot(df[sales_column].values)
    plt.title("异常值处理后的销售数据")
    plt.xlabel("时间")
    plt.ylabel("销售量")
    plt.tight_layout()
    plt.savefig('processed_data.png')
    plt.close()
    
    # 特征工程
    df_with_features = feature_engineering(df)
    print(f"特征工程后的数据形状: {df_with_features.shape}")
    print(f"特征列表: {df_with_features.columns.tolist()}")
    
    # 使用RobustScaler以减少异常值的影响
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    # 选择特征和目标列
    feature_cols = [col for col in df_with_features.columns if col not in ['日期']]
    
    # 缩放特征
    scaled_features = scaler_X.fit_transform(df_with_features[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols)
    
    # 缩放目标变量
    sales_scaled = scaler_y.fit_transform(df_with_features[[sales_column]])
    df_scaled[sales_column] = sales_scaled
    
    # 创建时间序列数据
    seq_length = 14  # 使用过去14天的数据预测下一天
    X, y = create_multivariate_sequences(df_scaled, seq_length)
    print(f"输入序列形状: {X.shape}, 输出序列形状: {y.shape}")
    
    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型超参数
    input_size = X.shape[2]  # 特征数量
    hidden_size = 128  # 增加隐藏层大小
    num_layers = 3  # 增加LSTM层数
    output_size = 1  # 预测下一天的销售量
    
    # 创建模型
    model = HybridTemporalModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        seq_len=seq_length,
        dropout=0.3  # 增加dropout比例
    )
    
    # 打印模型结构
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练模型
    num_epochs = 100  # 增加训练轮数
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=20  # 增加耐心值
    )
    
    # 保存模型到上级目录
    torch.save(model.state_dict(), model_save_path)
    print(f"最佳模型已保存到: {model_save_path}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 在测试集上评估模型
    predictions, actuals, mse, rmse, r2 = evaluate_model(model, test_loader, scaler_y, device)
    
    # 打印评估指标
    print(f"测试集MSE: {mse:.4f}")
    print(f"测试集RMSE: {rmse:.4f}")
    print(f"测试集R²: {r2:.4f}")
    
    # 可视化结果
    plot_results(predictions, actuals, train_losses, val_losses)
    
    # 保存最终预测结果
    results_df = pd.DataFrame({
        'Actual': actuals.flatten(),
        'Predicted': predictions.flatten()
    })
    results_path = os.path.join(parent_dir, 'prediction_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"训练完成，预测结果已保存到 {results_path}")

if __name__ == "__main__":
    main() 