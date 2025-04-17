import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import lr_scheduler
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

def prepare_data(df, target_col='成交商品件数', train_ratio=0.8, val_ratio=0.1):
    """准备模型数据"""
    # 创建高级特征
    df_features = create_advanced_features(df, target_col)
    
    # 准备Prophet数据
    prophet_df = pd.DataFrame({
        'ds': df_features['日期'],
        'y': df_features[target_col]
    })
    
    # 准备其他模型的特征
    feature_cols = [col for col in df_features.columns 
                   if col not in ['日期', target_col]]
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    # 数据分割
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size+val_size]
    y_test = y[train_size+val_size:]
    
    prophet_train = prophet_df[:train_size]
    prophet_val = prophet_df[train_size:train_size+val_size]
    prophet_test = prophet_df[train_size+val_size:]
    
    # 数据缩放
    X_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    X_train_scaled = pd.DataFrame(
        X_scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        X_scaler.transform(X_val),
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        X_scaler.transform(X_test),
        columns=X_test.columns
    )
    
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
    
    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled), \
           (prophet_train, prophet_val, prophet_test), (X_scaler, y_scaler), feature_cols

def train_prophet_model(train_df):
    """训练Prophet模型"""
    model = Prophet(
        yearly_seasonality=20,
        weekly_seasonality=10,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=15,
        holidays_prior_scale=20,
        changepoint_range=0.9,
        seasonality_mode='additive'
    )
    
    model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=6)
    
    model.fit(train_df)
    return model

def train_ensemble_models(X_train, y_train):
    """训练集成模型"""
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # RandomForest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # 训练模型
    print("训练XGBoost...")
    xgb_model.fit(X_train, y_train.ravel())
    
    print("训练LightGBM...")
    lgb_model.fit(X_train, y_train.ravel())
    
    print("训练随机森林...")
    rf_model.fit(X_train, y_train.ravel())
    
    return xgb_model, lgb_model, rf_model

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_neural_network(X_train, y_train, X_val, y_val, epochs=500, batch_size=32, lr=0.001, patience=20):
    """训练神经网络模型"""
    # 准备数据
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = SimpleNN(X_train.shape[1])
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 早停设置
    best_val_loss = float('inf')
    best_model = None
    no_improve_count = 0
    
    # 训练模型
    model.train()
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor.view(-1, 1))
            
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    return model

class NeuralEnsemble(nn.Module):
    def __init__(self, n_models, n_features, hidden_size=32):
        super(NeuralEnsemble, self).__init__()
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 预测提取器
        self.prediction_extractor = nn.Sequential(
            nn.Linear(n_models, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 残差预测器
        self.residual_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x_features, x_predictions):
        # 确保输入维度正确
        if len(x_features.shape) == 1:
            x_features = x_features.unsqueeze(0)
        if len(x_predictions.shape) == 1:
            x_predictions = x_predictions.unsqueeze(0)
            
        # 提取特征
        features = self.feature_extractor(x_features)
        predictions = self.prediction_extractor(x_predictions)
        
        # 打印维度信息用于调试
        print(f"特征维度: {features.shape}")
        print(f"预测维度: {predictions.shape}")
        
        # 组合特征和预测
        combined = torch.cat([features, predictions], dim=1)
        
        # 预测残差
        residual = self.residual_predictor(combined)
        
        return residual

class DynamicLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(DynamicLoss, self).__init__()
        self.alpha = alpha  # 动态权重损失系数
        self.beta = beta    # 趋势一致性损失系数
        self.gamma = gamma  # 平滑性损失系数
        
    def forward(self, pred, target, prev_pred=None):
        # 1. 动态权重损失
        abs_error = torch.abs(pred - target)
        error_weights = torch.exp(-abs_error / torch.mean(abs_error))
        weighted_mse = torch.mean(error_weights * (pred - target) ** 2)
        
        # 2. 趋势一致性损失
        if prev_pred is not None:
            pred_trend = pred[1:] - pred[:-1]
            target_trend = target[1:] - target[:-1]
            trend_loss = torch.mean(torch.abs(torch.sign(pred_trend) - torch.sign(target_trend)))
        else:
            trend_loss = torch.tensor(0.0, device=pred.device)
        
        # 3. 相对误差损失
        relative_error = torch.abs(pred - target) / (torch.abs(target) + 1e-6)
        relative_loss = torch.mean(relative_error)
        
        # 4. 平滑性损失
        if prev_pred is not None:
            smoothness = torch.abs(pred[1:] - pred[:-1])
            smoothness_loss = torch.mean(smoothness)
        else:
            smoothness_loss = torch.tensor(0.0, device=pred.device)
        
        # 组合所有损失
        total_loss = (self.alpha * weighted_mse + 
                     self.beta * trend_loss + 
                     (1 - self.alpha - self.beta) * relative_loss +
                     self.gamma * smoothness_loss)
        
        return total_loss

def train_neural_ensemble(base_predictions, y_true, val_predictions, y_val, 
                         X_train_features, X_val_features, epochs=500, batch_size=32, lr=0.0005):
    """训练神经网络残差预测器"""
    print(f"输入特征数量: {X_train_features.shape[1]}")
    print(f"输入模型数量: {base_predictions.shape[1]}")
    
    # 计算残差（实际值与神经网络预测值之间的差异）
    nn_train_pred = base_predictions[:, -1].reshape(-1, 1)  # 最后一列是神经网络预测
    nn_val_pred = val_predictions[:, -1].reshape(-1, 1)
    
    train_residuals = y_true - nn_train_pred
    val_residuals = y_val - nn_val_pred
    
    # 打印残差统计信息
    print("\n残差统计信息:")
    print(f"训练集残差均值: {np.mean(train_residuals):.4f}")
    print(f"训练集残差标准差: {np.std(train_residuals):.4f}")
    print(f"验证集残差均值: {np.mean(val_residuals):.4f}")
    print(f"验证集残差标准差: {np.std(val_residuals):.4f}")
    
    # 准备数据
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_features),
        torch.FloatTensor(base_predictions),
        torch.FloatTensor(train_residuals)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_features),
        torch.FloatTensor(val_predictions),
        torch.FloatTensor(val_residuals)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = NeuralEnsemble(
        n_models=base_predictions.shape[1],
        n_features=X_train_features.shape[1]
    )
    
    # 使用新的动态损失函数
    criterion = DynamicLoss(alpha=0.4, beta=0.3, gamma=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # 计算总步数
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    
    # 设置余弦退火学习率调度器
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    # 早停设置
    best_val_loss = float('inf')
    best_model = None
    no_improve_count = 0
    patience = 30
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # 训练模型
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        prev_pred = None
        for batch_features, batch_predictions, batch_residual in train_loader:
            optimizer.zero_grad()
            pred_residual = model(batch_features, batch_predictions)
            loss = criterion(pred_residual, batch_residual, prev_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            prev_pred = pred_residual.detach()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 记录当前学习率
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        prev_pred = None
        with torch.no_grad():
            for val_features, val_predictions, val_residual in val_loader:
                pred_residual = model(val_features, val_predictions)
                val_loss += criterion(pred_residual, val_residual, prev_pred).item()
                prev_pred = pred_residual
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], LR: {current_lr:.6f}, '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    
    # 评估验证集上的性能
    model.eval()
    with torch.no_grad():
        # 分批处理验证集数据
        val_residual_pred = []
        for val_features, val_predictions, _ in val_loader:
            batch_pred = model(val_features, val_predictions)
            val_residual_pred.append(batch_pred.numpy())
        
        val_residual_pred = np.concatenate(val_residual_pred, axis=0)
        
        # 计算验证集上的改进
        nn_val_error = np.abs(val_residuals)
        ensemble_val_error = np.abs(val_residuals - val_residual_pred)
        val_improvement = (nn_val_error - ensemble_val_error) / nn_val_error * 100
        val_improvement_rate = np.mean(val_improvement)
        
        print(f"\n验证集上的改进率: {val_improvement_rate:.2f}%")
    
    return model

def plot_model_comparison(results, y_true, predictions):
    """绘制模型对比结果"""
    # 创建三个子图
    fig = plt.figure(figsize=(15, 18))
    gs = plt.GridSpec(3, 1, height_ratios=[1, 0.8, 1])
    
    # 第一个子图：所有模型预测对比
    ax1 = fig.add_subplot(gs[0])
    
    # 设置颜色和样式
    colors = {
        'XGBoost': '#1f77b4',      # 蓝色
        'LightGBM': '#ff7f0e',     # 橙色
        'RandomForest': '#2ca02c',  # 绿色
        'NeuralNetwork': '#9467bd', # 紫色
        'NeuralEnsemble': '#d62728' # 红色
    }
    
    markers = {
        'XGBoost': 's',        # 方形
        'LightGBM': '^',       # 三角形
        'RandomForest': 'D',   # 菱形
        'NeuralNetwork': 'v',  # 倒三角
        'NeuralEnsemble': 'p'  # 五角星
    }
    
    # 先画预测值
    for model_name, pred in predictions.items():
        ax1.plot(pred, label=model_name, 
                color=colors[model_name],
                linestyle='-',
                linewidth=1.5,
                alpha=0.8,
                marker=markers[model_name],
                markersize=8,
                markerfacecolor='white',
                markeredgecolor=colors[model_name],
                markeredgewidth=1.5)
    
    # 最后画实际值
    ax1.plot(y_true, label='实际值', color='black', 
             linewidth=2, linestyle='-',
             marker='o', markersize=8, 
             markerfacecolor='white',
             markeredgecolor='black', 
             markeredgewidth=2,
             zorder=10)
    
    ax1.set_title('各模型预测结果对比', fontsize=14, pad=20)
    ax1.set_xlabel('样本索引', fontsize=12)
    ax1.set_ylabel('销量', fontsize=12)
    ax1.legend(fontsize=10, loc='best', framealpha=0.8)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 第三个子图：残差修正效果对比
    ax3 = fig.add_subplot(gs[2])
    
    nn_pred = predictions['NeuralNetwork']
    ensemble_pred = predictions['NeuralEnsemble']
    residuals = ensemble_pred - nn_pred
    
    # 计算修正成功的点
    nn_error = np.abs(nn_pred - y_true)
    ensemble_error = np.abs(ensemble_pred - y_true)
    
    # 判断修正成功的条件：
    # 1. 集成模型的误差小于神经网络的误差
    # 2. 误差改善超过5%（避免微小的改善）
    improvement = (nn_error - ensemble_error) / nn_error * 100
    successful_correction = (improvement > 5) & (ensemble_error < nn_error)
    
    # 绘制神经网络预测
    ax3.plot(nn_pred, label='神经网络预测', 
             color=colors['NeuralNetwork'],
             linestyle='-',
             linewidth=1.5,
             marker=markers['NeuralNetwork'],
             markersize=8,
             markerfacecolor='white',
             markeredgecolor=colors['NeuralNetwork'],
             markeredgewidth=1.5)
    
    # 绘制修正后的预测
    ax3.plot(ensemble_pred, label='残差修正后', 
             color=colors['NeuralEnsemble'],
             linestyle='-',
             linewidth=1.5,
             marker=markers['NeuralEnsemble'],
             markersize=8,
             markerfacecolor='white',
             markeredgecolor=colors['NeuralEnsemble'],
             markeredgewidth=1.5)
    
    # 绘制实际值
    ax3.plot(y_true, label='实际值', color='black',
             linewidth=2, linestyle='-',
             marker='o', markersize=8, 
             markerfacecolor='white',
             markeredgecolor='black', 
             markeredgewidth=2,
             zorder=10)
    
    # 标记修正成功的点
    for i in range(len(y_true)):
        if successful_correction[i]:
            ax3.plot(i, ensemble_pred[i], 'g*', markersize=12, alpha=0.5, zorder=5)
    
    # 绘制残差（使用双轴）
    ax3_twin = ax3.twinx()
    ax3_twin.plot(residuals, label='残差', 
                  color='#e377c2',  # 粉色
                  linestyle='--', 
                  linewidth=1.5,
                  marker='.',
                  markersize=6,
                  alpha=0.6)
    ax3_twin.set_ylabel('残差值', color='#e377c2', fontsize=12)
    ax3_twin.tick_params(axis='y', labelcolor='#e377c2')
    
    # 合并两个轴的图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    success_patch = mpatches.Patch(color='green', alpha=0.5, label='修正成功')
    ax3.legend(lines1 + lines2 + [success_patch], 
              labels1 + labels2 + ['修正成功'],
              fontsize=10, loc='best', framealpha=0.8)
    
    # 添加修正成功率文本
    success_rate = np.mean(successful_correction) * 100
    ax3.text(0.02, 0.98, f'修正成功率: {success_rate:.2f}%',
             transform=ax3.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax3.set_title('残差修正效果对比', fontsize=14, pad=20)
    ax3.set_xlabel('样本索引', fontsize=12)
    ax3.set_ylabel('销量', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # 第二个子图：误差指标对比
    ax2 = fig.add_subplot(gs[1])
    metrics = ['rmse', 'mae', 'mape']
    x = np.arange(len(metrics))
    width = 0.15
    
    max_rmse = max(result['rmse'] for result in results.values())
    max_mae = max(result['mae'] for result in results.values())
    max_mape = max(result['mape'] for result in results.values())
    
    for i, (model_name, result) in enumerate(results.items()):
        values = [result[metric] for metric in metrics]
        bars = ax2.bar(x + i*width, values, width, 
                      label=model_name, 
                      color=colors[model_name],
                      alpha=0.8)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', 
                    rotation=45,
                    fontsize=8)
    
    ax2.set_title('各模型误差指标对比', fontsize=14, pad=20)
    ax2.set_xticks(x + width*2.5)
    ax2.set_xticklabels(['RMSE', 'MAE', 'MAPE'])
    ax2.set_ylabel('误差值', fontsize=12)
    ax2.set_ylim(0, max(max_rmse, max_mae, max_mape) * 1.2)
    ax2.legend(fontsize=10, loc='best', framealpha=0.8)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('models/model_comparison.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()

def evaluate_model(y_true, y_pred, model_name, nn_pred=None):
    """评估模型性能"""
    # 计算相对误差
    relative_error = np.abs(y_true - y_pred) / np.abs(y_true)
    mape = np.mean(relative_error) * 100
    
    # 计算其他指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 如果是集成模型，计算相对于神经网络的改进
    if model_name == 'NeuralEnsemble' and nn_pred is not None:
        nn_error = np.abs(y_true - nn_pred)
        ensemble_error = np.abs(y_true - y_pred)
        improvement = (nn_error - ensemble_error) / nn_error * 100
        improvement_rate = np.mean(improvement)
        print(f"相对于神经网络的改进率: {improvement_rate:.2f}%")
    
    print(f"\n{model_name} 模型评估指标：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    print(f"R² 分数: {r2:.4f}")
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

def make_ensemble_predictions(other_models, simple_nn, X_test, X_test_features, y_scaler, nn_ensemble=None):
    """使用残差预测进行集成预测"""
    # 其他模型预测（只使用RandomForest）
    other_preds = []
    for model in other_models:
        pred = model.predict(X_test).reshape(-1, 1)
        other_preds.append(pred)
    
    # SimpleNN预测
    with torch.no_grad():
        simple_nn.eval()
        simple_nn_pred = simple_nn(torch.FloatTensor(X_test.values)).numpy()
        simple_nn_pred = simple_nn_pred.reshape(-1, 1)
    
    # 组合所有预测结果（只包含RandomForest和SimpleNN）
    all_preds = np.concatenate(other_preds + [simple_nn_pred], axis=1)
    
    # 使用神经网络预测残差
    with torch.no_grad():
        nn_ensemble.eval()
        predicted_residual = nn_ensemble(
            torch.FloatTensor(X_test_features),
            torch.FloatTensor(all_preds)
        ).numpy()
    
    # 最终预测 = 神经网络预测 + 预测的残差
    final_pred_scaled = simple_nn_pred + predicted_residual
    
    # 反缩放预测结果
    final_pred = y_scaler.inverse_transform(final_pred_scaled)
    other_preds_unscaled = [y_scaler.inverse_transform(pred) for pred in other_preds]
    other_preds_unscaled.append(y_scaler.inverse_transform(simple_nn_pred))
    
    # 打印预测统计信息
    print("\n预测统计信息:")
    print(f"神经网络预测均值: {np.mean(simple_nn_pred):.4f}")
    print(f"神经网络预测标准差: {np.std(simple_nn_pred):.4f}")
    print(f"集成模型预测均值: {np.mean(final_pred_scaled):.4f}")
    print(f"集成模型预测标准差: {np.std(final_pred_scaled):.4f}")
    print(f"残差预测均值: {np.mean(predicted_residual):.4f}")
    print(f"残差预测标准差: {np.std(predicted_residual):.4f}")
    
    return final_pred.flatten(), [pred.flatten() for pred in other_preds_unscaled]

def main():
    # 加载数据
    df = pd.read_csv('total_cleaned.csv')
    
    # 准备数据
    (X_train, y_train, X_val, y_val, X_test, y_test), \
    (prophet_train, prophet_val, prophet_test), \
    (X_scaler, y_scaler), feature_cols = prepare_data(df)
    
    print("数据集大小:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 训练其他模型（只用于对比）
    print("\n训练对比模型...")
    xgb_model, lgb_model, rf_model = train_ensemble_models(X_train, y_train)
    
    print("\n训练神经网络模型...")
    simple_nn = train_neural_network(X_train.values, y_train, X_val.values, y_val)
    
    # 获取所有模型在验证集上的预测
    val_preds = []
    # 只使用RandomForest和SimpleNN进行集成
    for model in [rf_model]:
        pred = model.predict(X_val).reshape(-1, 1)
        val_preds.append(pred)
    
    # 添加SimpleNN的验证集预测
    with torch.no_grad():
        simple_nn.eval()
        simple_nn_val_pred = simple_nn(torch.FloatTensor(X_val.values)).numpy()
    val_preds.append(simple_nn_val_pred)
    
    val_preds = np.concatenate(val_preds, axis=1)
    
    # 训练神经网络集成器
    print("\n训练神经网络集成器...")
    nn_ensemble = train_neural_ensemble(
        val_preds, y_val, val_preds, y_val,
        X_val.values, X_val.values
    )
    
    # 使用神经网络生成最终预测
    ensemble_predictions, other_preds = make_ensemble_predictions(
        [rf_model], simple_nn,  # 只使用RandomForest
        X_test, X_test.values,
        y_scaler, nn_ensemble
    )
    
    # 获取实际值
    y_test_orig = y_scaler.inverse_transform(y_test).flatten()
    
    # 评估各个模型
    results = {}
    predictions = {}
    
    # 评估对比模型
    model_names = ['XGBoost', 'LightGBM', 'RandomForest', 'NeuralNetwork']
    for name, model in zip(model_names, [xgb_model, lgb_model, rf_model, simple_nn]):
        if name == 'NeuralNetwork':
            with torch.no_grad():
                model.eval()
                pred = model(torch.FloatTensor(X_test.values)).numpy()
        else:
            pred = model.predict(X_test)
        pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        results[name] = evaluate_model(y_test_orig, pred, name)
        predictions[name] = pred
    
    # 评估集成模型
    results['NeuralEnsemble'] = evaluate_model(y_test_orig, ensemble_predictions, 'NeuralEnsemble', predictions['NeuralNetwork'])
    predictions['NeuralEnsemble'] = ensemble_predictions
    
    # 绘制对比结果
    plot_model_comparison(results, y_test_orig, predictions)
    
    # 保存模型和结果
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    joblib.dump(lgb_model, 'models/lgb_model.joblib')
    joblib.dump(rf_model, 'models/rf_model.joblib')
    joblib.dump(X_scaler, 'models/X_scaler.joblib')
    joblib.dump(y_scaler, 'models/y_scaler.joblib')
    torch.save(simple_nn.state_dict(), 'models/nn_model.pth')
    torch.save(nn_ensemble.state_dict(), 'models/neural_ensemble.pth')
    
    # 保存预测结果
    results_df = pd.DataFrame({
        '实际值': y_test_orig,
        'XGBoost预测值': predictions['XGBoost'],
        'LightGBM预测值': predictions['LightGBM'],
        'RandomForest预测值': predictions['RandomForest'],
        'NeuralNetwork预测值': predictions['NeuralNetwork'],
        'NeuralEnsemble预测值': predictions['NeuralEnsemble']
    })
    results_df.to_csv('models/model_predictions.csv', index=False)
    
    print("\n模型和结果已保存到models目录")

if __name__ == "__main__":
    main()