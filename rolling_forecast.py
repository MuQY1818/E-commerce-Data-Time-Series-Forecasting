import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from models.model_comparison import create_advanced_features, prepare_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_advanced_features(df, target_col='成交商品件数'):
    """创建高级特征"""
    df_copy = df.copy()
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    
    # 基础时间特征
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
        df_copy[f'ma_ratio_{window}'] = df_copy[target_col] / df_copy[f'rolling_mean_{window}']
    
    # 差分特征
    df_copy['diff_1'] = df_copy[target_col].diff()
    df_copy['diff_7'] = df_copy[target_col].diff(7)
    
    # 趋势特征
    df_copy['trend'] = df_copy[target_col].rolling(window=7, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # 处理缺失值
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

class EnhancedNN(torch.nn.Module):
    def __init__(self, input_size):
        super(EnhancedNN, self).__init__()
        
        # 特征提取层
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        # LSTM层
        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )
        
        # 输出层
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(64, 32),  # 64 from bidirectional LSTM (32*2)
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # 检查输入维度并调整
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列长度维度
        elif len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # 添加batch和序列长度维度
            
        batch_size, seq_len, features = x.size()
        
        # 特征提取
        x = x.reshape(-1, features)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步的输出
        final_hidden = lstm_out[:, -1, :]
        
        # 生成预测
        out = self.regressor(final_hidden)
        return out

def custom_loss(pred, target):
    """自定义损失函数，包含多个组件"""
    # MSE损失
    mse_loss = torch.nn.functional.mse_loss(pred, target)
    
    # Huber损失
    huber_loss = torch.nn.functional.huber_loss(pred, target, delta=1.0)
    
    # 方向一致性损失（带有梯度裁剪）
    pred_diff = pred[1:] - pred[:-1]
    target_diff = target[1:] - target[:-1]
    direction_loss = torch.mean(
        torch.clamp(
            -torch.sign(pred_diff) * torch.sign(target_diff),
            min=0.0
        )
    )
    
    # 总损失
    total_loss = 0.5 * mse_loss + 0.3 * huber_loss + 0.2 * direction_loss
    return total_loss

def load_models():
    """加载所有训练好的模型"""
    models = {}
    
    # 加载XGBoost模型
    models['xgb'] = joblib.load('models/xgb_model.joblib')
    
    # 加载LightGBM模型
    models['lgb'] = joblib.load('models/lgb_model.joblib')
    
    # 加载RandomForest模型
    models['rf'] = joblib.load('models/rf_model.joblib')
    
    # 加载数据缩放器
    models['X_scaler'] = joblib.load('models/X_scaler.joblib')
    models['y_scaler'] = joblib.load('models/y_scaler.joblib')
    
    # 获取特征数量
    df = pd.read_csv('total_cleaned.csv')
    features = create_advanced_features(df)
    input_size = len(features.columns) - 2  # 减去日期和成交商品件数两列
    
    # 加载神经网络模型（使用原始的SimpleNN结构）
    nn_model = SimpleNN(input_size)
    nn_model.load_state_dict(torch.load('models/nn_model.pth'))
    models['nn'] = nn_model
    
    return models

def make_single_prediction(models, X):
    """使用所有模型进行单步预测"""
    # 数据缩放
    X_scaled = models['X_scaler'].transform(X)
    
    # 获取各个模型的预测
    predictions = []
    
    # RandomForest预测
    rf_pred = models['rf'].predict(X_scaled).reshape(-1, 1)
    predictions.append(rf_pred)
    
    # 神经网络预测
    with torch.no_grad():
        models['nn'].eval()
        # 将输入转换为张量
        X_tensor = torch.FloatTensor(X_scaled)
        nn_pred = models['nn'](X_tensor).numpy()
    
    # 最终预测 = 神经网络预测
    final_pred_scaled = nn_pred
    
    # 反缩放预测结果
    final_pred = models['y_scaler'].inverse_transform(final_pred_scaled)
    
    return final_pred.flatten()

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, sequence_length=5):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # 返回一个序列和其对应的目标值
        return (
            self.features[idx:idx + self.sequence_length],
            self.targets[idx + self.sequence_length - 1]
        )

def train_model_with_rolling_window(df, window_size=180, prediction_size=30, validation_size=30):
    """使用滚动窗口方法训练模型"""
    print("开始滚动窗口训练...")
    
    # 创建特征
    features_df = create_advanced_features(df)
    feature_cols = [col for col in features_df.columns if col not in ['日期', '成交商品件数']]
    
    # 初始化模型
    input_size = len(feature_cols)
    model = EnhancedNN(input_size)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
    
    # 初始化数据缩放器
    X_scaler = joblib.load('models/X_scaler.joblib')
    y_scaler = joblib.load('models/y_scaler.joblib')
    
    # 滚动训练参数
    best_model = None
    best_val_loss = float('inf')
    patience = 8
    min_epochs = 5
    sequence_length = 7
    
    for start_idx in range(0, len(df) - window_size - prediction_size, prediction_size):
        window_end = start_idx + window_size
        
        # 获取数据
        train_features = features_df[feature_cols].iloc[start_idx:window_end]
        train_targets = features_df['成交商品件数'].iloc[start_idx:window_end]
        val_features = features_df[feature_cols].iloc[window_end:window_end+validation_size]
        val_targets = features_df['成交商品件数'].iloc[window_end:window_end+validation_size]
        
        # 数据缩放
        train_features_scaled = X_scaler.transform(train_features)
        train_targets_scaled = y_scaler.transform(train_targets.values.reshape(-1, 1))
        val_features_scaled = X_scaler.transform(val_features)
        val_targets_scaled = y_scaler.transform(val_targets.values.reshape(-1, 1))
        
        # 创建数据加载器
        train_dataset = TimeSeriesDataset(train_features_scaled, train_targets_scaled, sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = TimeSeriesDataset(val_features_scaled, val_targets_scaled, sequence_length)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 学习率调度
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=20,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e4
        )
        
        no_improve_count = 0
        window_best_loss = float('inf')
        
        # 训练循环
        for epoch in range(20):
            model.train()
            total_loss = 0
            
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(batch_features)
                
                # 计算损失
                loss = custom_loss(outputs, batch_targets)
                
                # 检查损失值是否为NaN
                if torch.isnan(loss):
                    print(f"警告：检测到NaN损失值，跳过此批次")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_features, val_targets in val_loader:
                    val_outputs = model(val_features)
                    val_loss += custom_loss(val_outputs, val_targets).item()
            
            val_loss /= len(val_loader)
            
            # 检查验证损失是否为NaN
            if torch.isnan(torch.tensor(val_loss)):
                print(f"警告：检测到NaN验证损失值，跳过此轮次")
                continue
            
            # 早停检查
            if epoch >= min_epochs and val_loss < window_best_loss:
                window_best_loss = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict().copy()
                no_improve_count = 0
            elif epoch >= min_epochs:
                no_improve_count += 1
            
            if epoch >= min_epochs and no_improve_count >= patience:
                print(f"Early stopping at window {start_idx}, epoch {epoch}")
                break
            
            if epoch % 2 == 0:
                print(f"Window {start_idx}-{window_end}, Epoch {epoch}, "
                      f"Train Loss: {total_loss/len(train_loader):.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def rolling_forecast(df, models, steps=60, window_size=180):
    """执行滚动预测"""
    # 创建预测结果DataFrame
    forecast_df = pd.DataFrame()
    
    # 获取最后的数据点
    last_date = pd.to_datetime(df['日期'].iloc[-1])
    
    # 准备初始数据
    current_data = df.copy()
    
    # 计算最近的趋势和波动性
    recent_window = min(window_size, len(df))
    recent_data = df.tail(recent_window)['成交商品件数'].values
    trend = np.polyfit(range(len(recent_data)), recent_data, deg=1)[0]
    volatility = np.std(recent_data) / np.mean(recent_data)
    
    # 存储预测结果
    predictions = []
    predictions_upper = []
    predictions_lower = []
    dates = []
    
    # 动态调整预测区间
    base_confidence = 0.9
    for i in range(steps):
        # 计算预测日期
        forecast_date = last_date + timedelta(days=i+1)
        dates.append(forecast_date)
        
        # 创建特征
        features = create_advanced_features(current_data)
        latest_features = features.iloc[-1:].drop(['日期', '成交商品件数'], axis=1)
        
        # 进行预测
        pred = make_single_prediction(models, latest_features)
        
        # 应用动态趋势调整
        trend_weight = np.exp(-i/steps)  # 趋势权重随时间衰减
        trend_adjustment = trend * i * trend_weight
        adjusted_pred = max(0, pred[0] + trend_adjustment)
        
        # 动态计算预测区间
        confidence_level = base_confidence * (1 - volatility * (i/steps))
        prediction_std = volatility * adjusted_pred * (1 + i/steps)
        z_score = 1.96  # 95% 置信区间
        
        upper_bound = adjusted_pred + z_score * prediction_std
        lower_bound = max(0, adjusted_pred - z_score * prediction_std)
        
        predictions.append(adjusted_pred)
        predictions_upper.append(upper_bound)
        predictions_lower.append(lower_bound)
        
        # 更新数据
        new_row = pd.DataFrame({
            '日期': [forecast_date],
            '成交商品件数': [adjusted_pred]
        })
        current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    # 创建预测结果DataFrame
    forecast_df = pd.DataFrame({
        '日期': dates,
        '预测值': predictions,
        '上界': predictions_upper,
        '下界': predictions_lower
    })
    
    return forecast_df

def evaluate_predictions(df, forecast_df):
    """评估预测结果与真实值的对比"""
    # 获取预测期间的真实值
    actual_data = df[df['日期'].isin(forecast_df['日期'])].copy()
    actual_data.set_index('日期', inplace=True)
    
    if len(actual_data) > 0:
        # 将预测数据也设置相同的索引
        forecast_data = forecast_df.set_index('日期')
        
        # 确保索引对齐
        common_dates = actual_data.index.intersection(forecast_data.index)
        actual_data = actual_data.loc[common_dates]
        forecast_data = forecast_data.loc[common_dates]
        
        # 计算评估指标
        mse = mean_squared_error(actual_data['成交商品件数'], forecast_data['预测值'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_data['成交商品件数'], forecast_data['预测值'])
        mape = np.mean(np.abs((actual_data['成交商品件数'] - forecast_data['预测值']) / actual_data['成交商品件数'])) * 100
        
        # 计算预测区间覆盖率
        coverage = np.mean((actual_data['成交商品件数'] >= forecast_data['下界']) & 
                         (actual_data['成交商品件数'] <= forecast_data['上界'])) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'coverage': coverage,
            'actual_data': actual_data.reset_index()
        }
    return None

def plot_forecast(df, forecast_df):
    """绘制预测结果"""
    plt.figure(figsize=(15, 10))
    
    # 创建两个子图
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # 只显示最近180天的历史数据
    recent_history = df.tail(180).copy()
    
    # 在第一个子图中绘制预测结果
    ax1.plot(recent_history['日期'], recent_history['成交商品件数'], 
            label='历史数据', color='blue', alpha=0.7)
    
    ax1.plot(forecast_df['日期'], forecast_df['预测值'], 
            label='预测值', color='red', linestyle='--')
    
    # 添加预测区间
    ax1.fill_between(forecast_df['日期'], 
                    forecast_df['下界'],
                    forecast_df['上界'],
                    color='red', alpha=0.1,
                    label='预测区间')
    
    # 添加趋势线
    z = np.polyfit(range(len(recent_history)), recent_history['成交商品件数'], 1)
    p = np.poly1d(z)
    trend_line = p(range(len(recent_history)))
    ax1.plot(recent_history['日期'], trend_line, 
            'g--', alpha=0.5, label='趋势线')
    
    # 如果有真实值，添加到图中
    evaluation_results = evaluate_predictions(df, forecast_df)
    if evaluation_results is not None:
        actual_data = evaluation_results['actual_data']
        ax1.plot(actual_data['日期'], actual_data['成交商品件数'],
                label='真实值', color='green', linewidth=2)
        
        # 在第二个子图中绘制预测误差
        forecast_data = forecast_df.set_index('日期').loc[actual_data['日期']]
        error = forecast_data['预测值'].values - actual_data['成交商品件数'].values
        ax2.bar(actual_data['日期'], error, color='gray', alpha=0.5, label='预测误差')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.set_title('预测误差分析')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('误差')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend()
    
    ax1.set_title('销量预测结果（含趋势和置信区间）', fontsize=14)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('销量', fontsize=12)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加预测信息
    avg_pred = np.mean(forecast_df['预测值'])
    trend_pct = (forecast_df['预测值'].iloc[-1] - forecast_df['预测值'].iloc[0]) / forecast_df['预测值'].iloc[0] * 100
    
    info_text = f'平均预测值: {avg_pred:.0f}\n'
    info_text += f'预测趋势: {"上升" if trend_pct > 0 else "下降"} ({abs(trend_pct):.1f}%)'
    
    if evaluation_results is not None:
        info_text += f'\nRMSE: {evaluation_results["rmse"]:.0f}'
        info_text += f'\nMAPE: {evaluation_results["mape"] :.1f}%'
        info_text += f'\n预测区间覆盖率: {evaluation_results["coverage"]:.1f}%'
    
    ax1.text(0.02, 0.98, info_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('rolling_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    print("加载数据...")
    df = pd.read_csv('total_cleaned.csv')
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 划分训练集和测试集
    test_size = 60  # 使用最后60天作为测试集
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    # 使用滚动窗口方法训练模型
    print("训练模型...")
    nn_model = train_model_with_rolling_window(train_df)
    
    # 保存训练好的模型
    torch.save(nn_model.state_dict(), 'models/rolling_nn_model.pth')
    
    # 加载其他模型
    print("加载其他模型...")
    models = load_models()
    models['nn'] = nn_model  # 使用新训练的模型替换原来的模型
    
    # 执行滚动预测
    print("执行滚动预测...")
    forecast_df = rolling_forecast(train_df, models, steps=60)
    
    # 合并训练集和测试集用于绘图
    full_df = pd.concat([train_df, test_df])
    
    # 保存预测结果
    forecast_df.to_csv('rolling_forecast.csv', index=False)
    
    # 绘制预测结果
    print("绘制预测结果...")
    plot_forecast(full_df, forecast_df)
    
    # 评估预测结果
    evaluation_results = evaluate_predictions(full_df, forecast_df)
    if evaluation_results is not None:
        print("\n预测评估结果：")
        print(f"RMSE: {evaluation_results['rmse']:.0f}")
        print(f"MAE: {evaluation_results['mae']:.0f}")
        print(f"MAPE: {evaluation_results['mape']:.1f}%")
        print(f"预测区间覆盖率: {evaluation_results['coverage']:.1f}%")
    
    print("\n预测完成！结果已保存到 rolling_forecast.csv 和 rolling_forecast.png")

if __name__ == "__main__":
    main() 