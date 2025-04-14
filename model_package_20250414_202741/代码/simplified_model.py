import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import sys
import torch.nn.functional as F
import joblib  # 添加joblib用于保存scaler

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 简化的LSTM模型
class SimplifiedTemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.1):
        super(SimplifiedTemporalModel, self).__init__()
        
        # 简单的前馈网络
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # 只使用最后一个时间步的特征
        last_step = x[:, -1, :]
        return self.network(last_step)

# 安全的异常值处理函数
def handle_outliers(df, column, method='remove', threshold=3):
    """更严格的异常值处理方法"""
    sales_values = df[column].copy()
    df_copy = df.copy()
    
    if method == 'remove':
        # 使用IQR方法识别异常值
        Q1 = sales_values.quantile(0.25)
        Q3 = sales_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        print(f"异常值边界: 下界 = {lower_bound:.2f}, 上界 = {upper_bound:.2f}")
        
        # 创建异常值掩码
        outlier_mask = (sales_values >= lower_bound) & (sales_values <= upper_bound)
        
        # 统计和打印异常值信息
        outlier_count = (~outlier_mask).sum()
        total_count = len(sales_values)
        print(f"检测到 {outlier_count} 个异常值，占总数据的 {(outlier_count/total_count)*100:.2f}%")
        
        # 只保留非异常值的数据
        df_clean = df_copy[outlier_mask].copy()
        
        # 确保索引连续
        df_clean = df_clean.reset_index(drop=True)
        
        return df_clean
        
    elif method == 'winsorize':
        # 使用分位数进行截断
        lower_percentile = np.percentile(sales_values, 5)  # 使用5%分位数
        upper_percentile = np.percentile(sales_values, 95) # 使用95%分位数
        
        print(f"异常值处理: 下界 = {lower_percentile:.2f}, 上界 = {upper_percentile:.2f}")
        
        # 截断异常值
        sales_values[sales_values < lower_percentile] = lower_percentile
        sales_values[sales_values > upper_percentile] = upper_percentile
        df_copy[column] = sales_values
        
        return df_copy
    
    return df_copy

# 安全的特征工程
def create_safe_features(df, target_col='成交商品件数'):
    """创建安全的特征集，专注于短期预测"""
    df_copy = df.copy()
    
    # 转换日期列为datetime类型
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    
    # 1. 基础时间特征
    df_copy['dayofweek'] = df_copy['日期'].dt.dayofweek
    df_copy['is_weekend'] = df_copy['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. 滞后特征（只预测下一天）
    df_copy['lag_1'] = df_copy[target_col].shift(1)
    df_copy['lag_2'] = df_copy[target_col].shift(2)
    df_copy['lag_3'] = df_copy[target_col].shift(3)
    
    # 3. 简单的统计特征
    df_copy['rolling_mean_3'] = df_copy[target_col].rolling(window=3, min_periods=1).mean()
    df_copy['rolling_std_3'] = df_copy[target_col].rolling(window=3, min_periods=1).std()
    
    # 删除不需要的列
    df_copy = df_copy.drop(['日期', 'dayofweek'], axis=1)
    
    # 处理缺失值
    df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
    
    return df_copy

# 安全的数据缩放
def scale_data_safely(df):
    """安全的数据缩放函数"""
    # 分离特征和目标
    target = df['成交商品件数'].values.reshape(-1, 1)
    features = df.drop('成交商品件数', axis=1).values
    
    # 对特征进行稳健缩放
    feature_scaler = RobustScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    # 对目标进行稳健缩放
    target_scaler = RobustScaler()
    target_scaled = target_scaler.fit_transform(target)
    
    # 合并数据
    scaled_data = np.hstack([features_scaled, target_scaled])
    
    return scaled_data, feature_scaler, target_scaler

# 安全的数据序列生成
def create_sequences(data, seq_length, target_col_idx=-1):
    """创建时间序列的滑动窗口，预测下一天"""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # 获取输入序列
        sequence = data[i:(i + seq_length)]
        # 获取目标值（下一天的值）
        target = data[i + seq_length, target_col_idx]
        
        X.append(sequence)
        y.append(target)
    
    return np.array(X), np.array(y)

# 定义损失函数，添加L1正则化
class CustomLoss(nn.Module):
    def __init__(self, lambda_l1=0.01):
        super(CustomLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.mse = nn.MSELoss()
        
    def forward(self, output, target, model):
        mse_loss = self.mse(output, target)
        l1_reg = torch.tensor(0., device=output.device)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        return mse_loss + self.lambda_l1 * l1_reg

# 改进的训练函数
def train_model_safely(model, train_loader, val_loader, criterion, optimizer, device, 
                    num_epochs=50, patience=10, gradient_clip=0.5, scheduler=None):
    """安全地训练模型"""
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        batch_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            try:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                total_train_loss += loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"训练批次出错: {e}")
                continue
        
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                try:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    total_val_loss += loss.item()
                    val_batch_count += 1
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue
        
        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停触发 - 在{patience}轮内验证损失没有改善")
            break
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# 绘制结果函数
def plot_results_safely(predictions, actuals, train_losses, val_losses):
    """绘制训练结果和预测效果"""
    if predictions is None or actuals is None:
        print("没有有效的预测结果可以绘制")
        return
        
    try:
        plt.figure(figsize=(20, 12))
        
        # 1. 训练和验证损失
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练过程中的损失变化')
        plt.legend()
        plt.grid(True)
        
        # 2. 预测结果对比
        plt.subplot(2, 2, 2)
        # 只取每个样本的实际预测值（不是序列）
        plt.plot(actuals.reshape(-1), label='实际值', alpha=0.7)
        plt.plot(predictions.reshape(-1), label='预测值', alpha=0.7)
        plt.xlabel('样本')
        plt.ylabel('销量')
        plt.title('预测值与实际值对比')
        plt.legend()
        plt.grid(True)
        
        # 3. 预测误差分布
        plt.subplot(2, 2, 3)
        errors = actuals.reshape(-1) - predictions.reshape(-1)
        plt.hist(errors, bins=50, density=True, alpha=0.75)
        plt.xlabel('预测误差')
        plt.ylabel('密度')
        plt.title('预测误差分布')
        plt.grid(True)
        
        # 4. 散点图：预测值 vs 实际值
        plt.subplot(2, 2, 4)
        plt.scatter(actuals.reshape(-1), predictions.reshape(-1), alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')  # 对角线
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('预测值与实际值的散点图')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('simplified_model_results.png')
        plt.close()
        
        # 额外绘制一个详细的预测对比图
        plt.figure(figsize=(15, 6))
        # 选择最后100个样本点进行展示
        n_samples = 100
        x = np.arange(n_samples)
        y_true = actuals[-n_samples:].reshape(-1)
        y_pred = predictions[-n_samples:].reshape(-1)
        
        plt.plot(x, y_true, 'b-', label='实际值', alpha=0.7)
        plt.plot(x, y_pred, 'r--', label='预测值', alpha=0.7)
        plt.fill_between(x, y_true, y_pred, alpha=0.2, color='gray')
        
        plt.xlabel('样本序号')
        plt.ylabel('销量')
        plt.title('最后100天预测效果对比')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_detail.png')
        plt.close()
        
        # 打印评估指标
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        print("\n模型评估指标：")
        print(f"均方误差 (MSE): {mse:.2f}")
        print(f"平均绝对误差 (MAE): {mae:.2f}")
        print(f"R² 分数: {r2:.4f}")
        
    except Exception as e:
        print(f"绘图过程出错: {e}")

class EnsembleLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_models, output_size, dropout=0.1):
        super(EnsembleLSTM, self).__init__()
        
        self.num_models = num_models
        
        # 创建不同时间尺度的预测模型
        self.models = nn.ModuleList([
            # 模型1：快速响应模型（较小的隐藏层，单层LSTM）
            nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LSTM(
                    input_size=hidden_sizes[0],
                    hidden_size=hidden_sizes[0],
                    num_layers=1,
                    batch_first=True
                ),
                nn.Linear(hidden_sizes[0], output_size)
            ),
            # 模型2：平滑预测模型（中等隐藏层，多层LSTM）
            nn.Sequential(
                nn.Linear(input_size, hidden_sizes[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LSTM(
                    input_size=hidden_sizes[1],
                    hidden_size=hidden_sizes[1],
                    num_layers=2,
                    batch_first=True,
                    dropout=dropout
                ),
                nn.Linear(hidden_sizes[1], output_size)
            ),
            # 模型3：趋势预测模型（较大隐藏层，残差连接）
            nn.Sequential(
                nn.Linear(input_size, hidden_sizes[2]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LSTM(
                    input_size=hidden_sizes[2],
                    hidden_size=hidden_sizes[2],
                    num_layers=1,
                    batch_first=True
                ),
                nn.Linear(hidden_sizes[2], output_size)
            )
        ])
        
        # 自适应权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(input_size + 3, 64),  # 输入特征 + 三个模型的预测
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 获取每个模型的预测
        predictions = []
        for model in self.models:
            # 前向传播
            rnn_out, _ = model[3](model[2](model[1](model[0](x))))
            
            # 获取最后一个时间步
            last_hidden = rnn_out[:, -1, :]
            
            # 输出层
            pred = model[4](last_hidden)
            predictions.append(pred)
        
        # 将预测结果堆叠
        stacked_preds = torch.stack(predictions, dim=1)  # [batch_size, num_models, 1]
        
        # 构建权重网络的输入
        last_features = x[:, -1, :]  # 最后一个时间步的特征
        preds_flat = stacked_preds.squeeze(-1)  # [batch_size, num_models]
        weight_input = torch.cat([last_features, preds_flat], dim=1)
        
        # 计算动态权重
        weights = self.weight_net(weight_input)
        weights = weights.unsqueeze(-1)  # [batch_size, num_models, 1]
        
        # 加权平均
        weighted_pred = torch.sum(stacked_preds * weights, dim=1)
        
        return weighted_pred

def train_ensemble_safely(ensemble, train_loader, val_loader, criterion, optimizer, device, 
                        num_epochs=50, patience=10, gradient_clip=0.5, scheduler=None):
    """安全地训练集成模型"""
    ensemble = ensemble.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        ensemble.train()
        total_train_loss = 0.0
        batch_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            try:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = ensemble(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # 梯度裁剪（对每个模型分别进行）
                for model in ensemble.models:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()
                
                total_train_loss += loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"训练批次出错: {e}")
                continue
        
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        ensemble.eval()
        total_val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                try:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = ensemble(X_val)
                    loss = criterion(outputs, y_val)
                    total_val_loss += loss.item()
                    val_batch_count += 1
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue
        
        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = ensemble.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停触发 - 在{patience}轮内验证损失没有改善")
            break
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        ensemble.load_state_dict(best_model_state)
    
    return train_losses, val_losses

class BalancedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(BalancedLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # 平滑损失权重
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # 基础MSE损失
        mse_loss = self.mse(pred, target)
        
        # 平滑损失（惩罚预测值的剧烈变化）
        smoothness_loss = torch.mean(torch.abs(pred[1:] - pred[:-1]))
        
        # 组合损失
        total_loss = self.alpha * mse_loss + self.beta * smoothness_loss
        
        return total_loss

def evaluate_model_safely(model, test_loader, scaler_y, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            try:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 转换回原始尺度
                if scaler_y is not None:
                    output = output.cpu().numpy()
                    target = target.cpu().numpy()
                    
                    # 确保维度正确 (batch_size, 1)
                    output = output.reshape(-1, 1)
                    target = target.reshape(-1, 1)
                    
                    output = scaler_y.inverse_transform(output)
                    target = scaler_y.inverse_transform(target)
                else:
                    output = output.cpu().numpy().reshape(-1, 1)
                    target = target.cpu().numpy().reshape(-1, 1)
                
                predictions.append(output)
                actuals.append(target)
                
            except Exception as e:
                print(f"评估过程出错: {e}")
                continue
    
    if not predictions or not actuals:
        print("警告：没有有效的预测结果")
        return None, None, None, None, None
    
    try:
        # 合并所有批次的预测结果
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # 计算评估指标
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        print("\n模型评估指标：")
        print(f"均方误差 (MSE): {mse:.2f}")
        print(f"均方根误差 (RMSE): {rmse:.2f}")
        print(f"平均绝对误差 (MAE): {mae:.2f}")
        print(f"决定系数 (R²): {r2:.4f}")
        
        return predictions, actuals, mse, rmse, r2
        
    except Exception as e:
        print(f"计算评估指标时出错: {e}")
        return None, None, None, None, None

# 添加峰值感知损失函数
class PeakAwareLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=1.0):
        super(PeakAwareLoss, self).__init__()
        self.alpha = alpha  # 峰值权重
        self.beta = beta    # 基础MSE权重
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        # 基础MSE损失
        mse_loss = self.mse(pred, target)
        
        # 计算目标值的相对大小作为权重
        weights = 1.0 + self.alpha * torch.abs(target) / torch.mean(torch.abs(target))
        
        # 加权MSE损失
        weighted_loss = weights * mse_loss
        
        return self.beta * torch.mean(weighted_loss)

class SimplePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(SimplePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # 只使用最后一个时间步
        x = x[:, -1, :]  # 取最后一天的特征
        return self.model(x)

def create_features(df, target_col='成交商品件数'):
    """创建简单的特征集"""
    df_copy = df.copy()
    
    # 基础时间特征
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    df_copy['dayofweek'] = df_copy['日期'].dt.dayofweek
    df_copy['is_weekend'] = df_copy['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 滞后特征（只用最近3天）
    for i in range(1, 4):
        df_copy[f'lag_{i}'] = df_copy[target_col].shift(i)
    
    # 简单统计特征
    df_copy['rolling_mean_3'] = df_copy[target_col].rolling(window=3, min_periods=1).mean()
    df_copy['rolling_std_3'] = df_copy[target_col].rolling(window=3, min_periods=1).std()
    
    # 删除不需要的列并处理缺失值
    df_copy = df_copy.drop(['日期', 'dayofweek'], axis=1)
    df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
    
    return df_copy

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def prepare_data(df, target_col='成交商品件数', train_ratio=0.8):
    """准备训练和测试数据"""
    # 分离特征和目标
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    
    # 使用RobustScaler进行缩放
    X_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
    
    # 划分训练集和测试集
    train_size = int(len(X_scaled) * train_ratio)
    
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y_scaled[:train_size]
    y_test = y_scaled[train_size:]
    
    return (X_train, y_train, X_test, y_test), (X_scaler, y_scaler)

def train_model(model, X_train, y_train, X_test, y_test, device, 
                batch_size=32, epochs=50, patience=10):
    """训练模型"""
    # 创建保存模型的目录
    os.makedirs('models', exist_ok=True)
    
    train_data = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            X_val = torch.FloatTensor(X_test).to(device)
            y_val = torch.FloatTensor(y_test).to(device)
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {val_loss.item():.4f}')
        
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss.item(),
            }, 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def load_model(model, device):
    """加载保存的模型"""
    try:
        checkpoint = torch.load('models/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型成功，来自epoch {checkpoint['epoch']}")
        print(f"验证损失: {checkpoint['val_loss']:.4f}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        df = pd.read_csv('total_cleaned.csv')
        print("数据加载成功:", df.shape)
        
        df_features = create_features(df)
        print("特征工程完成:", df_features.shape)
        
        (X_train, y_train, X_test, y_test), (_, y_scaler) = prepare_data(df_features)
        print("数据准备完成")
        
        input_size = X_train.shape[1]
        model = SimpleNN(input_size).to(device)
        print("模型创建完成")
        
        train_losses, val_losses = train_model(
            model, X_train, y_train, X_test, y_test, device
        )
        print("模型训练完成")
        
        # 保存特征和目标的缩放器
        joblib.dump(y_scaler, 'models/y_scaler.joblib')
        
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            predictions = model(X_test_tensor).cpu().numpy()
        
        predictions = y_scaler.inverse_transform(predictions)
        y_test_orig = y_scaler.inverse_transform(y_test)
        
        mse = mean_squared_error(y_test_orig, predictions)
        mae = mean_absolute_error(y_test_orig, predictions)
        r2 = r2_score(y_test_orig, predictions)
        
        print("\n模型评估指标：")
        print(f"均方误差 (MSE): {mse:.2f}")
        print(f"平均绝对误差 (MAE): {mae:.2f}")
        print(f"R² 分数: {r2:.4f}")
        
        plot_results_safely(predictions.reshape(-1), y_test_orig.reshape(-1), train_losses, val_losses)
        print("\n预测完成！请查看 simplified_model_results.png 了解详细结果。")
        print("模型和缩放器已保存到 'models/best_model.pth' 和 'models/y_scaler.joblib'")
        
    except Exception as e:
        print(f"运行出错: {e}")
        return

if __name__ == "__main__":
    main() 