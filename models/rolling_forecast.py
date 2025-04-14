import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib
import os
from matplotlib import font_manager
import seaborn as sns
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

def create_features(df, target_col='成交商品件数'):
    """创建特征，包括更多的时间特征和统计特征"""
    df_copy = df.copy()
    
    # 基础时间特征
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    df_copy['dayofweek'] = df_copy['日期'].dt.dayofweek
    df_copy['is_weekend'] = df_copy['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df_copy['month'] = df_copy['日期'].dt.month
    df_copy['day'] = df_copy['日期'].dt.day
    
    # 周期性特征
    df_copy['sin_day'] = np.sin(2 * np.pi * df_copy['日期'].dt.day / 31)
    df_copy['cos_day'] = np.cos(2 * np.pi * df_copy['日期'].dt.day / 31)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy['日期'].dt.month / 12)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy['日期'].dt.month / 12)
    
    # 滞后特征
    for i in range(1, 15):
        df_copy[f'lag_{i}'] = df_copy[target_col].shift(i)
    
    # 移动平均特征
    for window in [7, 14, 30]:
        df_copy[f'rolling_mean_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).mean()
        df_copy[f'rolling_std_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).std()
        # 添加移动平均比率
        df_copy[f'ma_ratio_{window}'] = df_copy[target_col] / df_copy[f'rolling_mean_{window}']
    
    # 差分特征
    df_copy['diff_1'] = df_copy[target_col].diff()
    df_copy['diff_7'] = df_copy[target_col].diff(7)
    
    # 趋势特征
    df_copy['trend'] = df_copy[target_col].rolling(window=7, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # 删除不需要的列并处理缺失值
    df_copy = df_copy.drop(['dayofweek'], axis=1)
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

def prepare_data(df, target_col='成交商品件数', train_ratio=0.8):
    """准备训练数据"""
    # 创建特征
    df_processed = create_features(df, target_col)
    
    # 分离特征和目标
    feature_cols = [col for col in df_processed.columns if col not in ['日期', target_col]]
    X = df_processed[feature_cols].values
    y = df_processed[target_col].values.reshape(-1, 1)
    
    # 数据缩放
    feature_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # 数据分割
    train_size = int(len(X_scaled) * train_ratio)
    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_test = y_scaled[train_size:]
    
    return X_train, y_train, X_test, y_test, feature_scaler, y_scaler, feature_cols

def train_model(X_train, y_train, input_size, device='cpu', epochs=200, lr=0.001):
    """训练模型"""
    model = SimpleNN(input_size).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def rolling_forecast(model, X_test, y_scaler, feature_scaler, feature_cols, pred_days=30, device='cpu', stability_window=3):
    """执行滚动预测，增加稳定性控制"""
    model.eval()
    predictions = []
    current_features = X_test.copy()
    
    with torch.no_grad():
        for i in range(pred_days):
            # 获取多个预测值并取平均
            pred_window = []
            for _ in range(stability_window):
                X_current = torch.FloatTensor(current_features[i:i+1]).to(device)
                pred = model(X_current).cpu().numpy()
                pred_window.append(pred[0][0])
            
            # 使用平均预测值
            final_pred = np.mean(pred_window)
            predictions.append(final_pred)
            
            if i < pred_days - 1:
                new_features = current_features[i+1].copy()
                
                # 计算最近的趋势
                lag_indices = [feature_cols.index(f'lag_{j}') for j in range(1, 4)]
                recent_values = [current_features[i][idx] for idx in lag_indices]
                recent_trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
                
                # 使用趋势调整的预测值
                adjusted_pred = final_pred + recent_trend * 0.5
                
                # 更新lag特征
                for j in range(14, 0, -1):
                    lag_idx = feature_cols.index(f'lag_{j}')
                    if j == 1:
                        new_features[lag_idx] = adjusted_pred
                    else:
                        prev_lag_idx = feature_cols.index(f'lag_{j-1}')
                        new_features[lag_idx] = current_features[i][prev_lag_idx]
                
                # 更新移动平均特征
                for window in [7, 14, 30]:
                    # 使用指数加权移动平均
                    alpha = 2 / (window + 1)
                    mean_idx = feature_cols.index(f'rolling_mean_{window}')
                    new_features[mean_idx] = (alpha * adjusted_pred + 
                        (1 - alpha) * current_features[i][mean_idx])
                    
                    # 更新其他统计特征
                    std_idx = feature_cols.index(f'rolling_std_{window}')
                    ratio_idx = feature_cols.index(f'ma_ratio_{window}')
                    new_features[std_idx] = current_features[i][std_idx]  # 保持不变
                    new_features[ratio_idx] = adjusted_pred / new_features[mean_idx]
                
                current_features[i+1] = new_features
    
    predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def plot_results(actual, predicted, title="滚动预测结果"):
    """绘制预测结果和误差"""
    plt.figure(figsize=(15, 10))
    
    # 预测结果对比图
    plt.subplot(2, 1, 1)
    plt.plot(actual, '-o', label='实际值', color='#1f77b4')
    plt.plot(predicted, '-s', label='预测值', color='#ff7f0e')
    plt.fill_between(range(len(actual)), actual, predicted, alpha=0.2, color='#1f77b4')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.ylabel("销量")
    plt.xlabel("预测天数")
    
    # 预测误差图
    plt.subplot(2, 1, 2)
    errors = actual - predicted
    plt.bar(range(len(errors)), errors, color='#1f77b4', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("预测误差")
    plt.ylabel("误差")
    plt.xlabel("预测天数")
    
    plt.tight_layout()
    
    # 确保models目录存在
    os.makedirs('models', exist_ok=True)
    # 保存图片
    plt.savefig('models/rolling_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n预测结果图表已保存到 models/rolling_predictions.png")

def evaluate_predictions(actual, predicted, window_size=7):
    """评估预测结果，包括整体评估和分窗口评估"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 整体评估
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print("\n=== 整体评估指标 ===")
    print(f"R² 分数: {r2:.4f}  (1.0为完美预测，0为无预测能力)")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    
    # 按时间窗口评估
    n_windows = len(actual) // window_size
    print(f"\n=== 按时间窗口评估（每{window_size}天） ===")
    
    window_r2_scores = []
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_actual = actual[start_idx:end_idx]
        window_pred = predicted[start_idx:end_idx]
        
        window_mse = mean_squared_error(window_actual, window_pred)
        window_rmse = np.sqrt(window_mse)
        window_mae = mean_absolute_error(window_actual, window_pred)
        window_r2 = r2_score(window_actual, window_pred)
        window_r2_scores.append(window_r2)
        
        print(f"\n时间窗口 {i+1}（第{start_idx+1}-{end_idx}天）：")
        print(f"R² 分数: {window_r2:.4f}")
        print(f"RMSE: {window_rmse:.2f}")
        print(f"MAE: {window_mae:.2f}")
        print(f"平均预测误差: {np.mean(window_actual - window_pred):.2f}")
        print(f"预测误差标准差: {np.std(window_actual - window_pred):.2f}")
    
    # 输出R²分数统计
    print("\n=== R²分数统计 ===")
    print(f"平均R²: {np.mean(window_r2_scores):.4f}")
    print(f"最大R²: {np.max(window_r2_scores):.4f}")
    print(f"最小R²: {np.min(window_r2_scores):.4f}")
    print(f"R²标准差: {np.std(window_r2_scores):.4f}")
    
    # 如果R²分数过低，给出警告和建议
    if r2 < 0.5:
        print("\n⚠️ 警告：整体R²分数较低，模型预测能力可能不足")
        print("建议：")
        print("1. 检查特征工程是否充分")
        print("2. 考虑增加历史数据窗口大小")
        print("3. 尝试调整模型架构或超参数")
    elif r2 < 0.7:
        print("\n⚠️ 提示：R²分数一般，还有提升空间")
        print("建议考虑优化特征工程或模型参数")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否可用CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    df = pd.read_csv('total_cleaned.csv')  # 修改为正确的文件名
    print(f'数据加载成功: {df.shape}')
    
    # 准备数据
    X_train, y_train, X_test, y_test, feature_scaler, y_scaler, feature_cols = prepare_data(df)
    print(f'特征工程完成: {df.shape}')
    
    # 训练模型
    input_size = X_train.shape[1]
    print(f'\n测试集: {len(X_test)} 样本')
    model = train_model(X_train, y_train, input_size, device=device)
    
    # 执行滚动预测
    pred_days = 30
    predictions = rolling_forecast(model, X_test, y_scaler, feature_scaler, feature_cols, 
                                pred_days=pred_days, device=device)
    
    # 获取实际值
    actual_values = y_scaler.inverse_transform(y_test[:pred_days].reshape(-1, 1)).flatten()
    
    # 保存预测结果
    results_df = pd.DataFrame({
        '实际值': actual_values,
        '预测值': predictions,
        '预测误差': actual_values - predictions
    })
    results_df.to_csv('models/prediction_results.csv', index=False)
    print("\n预测结果已保存到 models/prediction_results.csv")
    
    # 绘制结果
    plot_results(actual_values, predictions, title="滚动预测结果（未来30天）")
    
    # 评估预测结果
    evaluate_predictions(actual_values, predictions)

if __name__ == '__main__':
    main() 