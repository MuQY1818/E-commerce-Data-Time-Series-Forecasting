import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader, Dataset
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from matplotlib.font_manager import FontProperties
import os

# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 时间注意力机制
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, query, key, value):
        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # 计算注意力分数
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(query.device)
        
        # 应用softmax获取注意力权重
        attention = torch.softmax(energy, dim=-1)
        
        # 应用注意力权重
        x = torch.matmul(attention, V)
        
        return x, attention

# 时间卷积网络模块
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                     stride=1, dilation=dilation_size, 
                                     padding=(kernel_size-1) * dilation_size, 
                                     dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# 完整的混合模型架构
class HybridTemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len, dropout=0.2):
        super(HybridTemporalModel, self).__init__()
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # 注意力层
        self.attention = TemporalAttention(hidden_size)
        
        # TCN层
        self.tcn = TemporalConvNet(hidden_size, [hidden_size, hidden_size, hidden_size], kernel_size=3, dropout=dropout)
        
        # Transformer编码器层
        encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*4, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
        
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # LSTM部分
        lstm_out, _ = self.lstm(x)
        
        # 注意力层
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # TCN部分 (需要调整维度)
        tcn_input = attn_out.transpose(1, 2)  # 从[batch, seq, features]到[batch, features, seq]
        tcn_out = self.tcn(tcn_input)
        tcn_out = tcn_out.transpose(1, 2)  # 转回[batch, seq, features]
        
        # Transformer部分 (需要调整维度)
        transformer_input = tcn_out.transpose(0, 1)  # 从[batch, seq, features]到[seq, batch, features]
        transformer_out = self.transformer_encoder(transformer_input)
        transformer_out = transformer_out.transpose(0, 1)  # 转回[batch, seq, features]
        
        # 只用最后一个时间步的输出进行预测
        out = self.fc(transformer_out[:, -1, :])
        
        return out

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return train_losses, val_losses

# 评估函数
def evaluate_model(model, test_loader, scaler_y, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 将预测值和真实值转换回原始尺度
            pred_np = output.cpu().numpy()
            target_np = target.cpu().numpy()
            
            predictions.append(pred_np)
            actuals.append(target_np)
    
    # 合并所有批次的预测结果
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # 将预测值和真实值转换回原始尺度
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    
    # 避免除零错误
    actuals_nonzero = np.copy(actuals)
    actuals_nonzero[actuals_nonzero == 0] = 1e-10
    mape = np.mean(np.abs((actuals_nonzero - predictions) / actuals_nonzero)) * 100
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
    print(f"决定系数 (R²): {r2:.4f}")
    
    return predictions, actuals, mse, rmse, r2

# 绘制结果函数
def plot_results(predictions, actuals, train_losses, val_losses):
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    try:
        # 尝试设置中文字体
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows系统中文字体路径
        if os.path.exists(font_path):
            chinese_font = FontProperties(fname=font_path)
            has_chinese_font = True
        else:
            has_chinese_font = False
    except:
        has_chinese_font = False

    # 创建一个包含3个子图的图表
    plt.figure(figsize=(18, 12))
    
    # 1. 绘制训练和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if has_chinese_font:
        plt.title('训练和验证损失', fontproperties=chinese_font)
    else:
        plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 绘制预测结果和真实值 - 全部数据
    plt.subplot(2, 2, 2)
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    if has_chinese_font:
        plt.title('真实值与预测值对比 (全部数据)', fontproperties=chinese_font)
    else:
        plt.title('Actual vs Predicted (All Data)')
    plt.legend()
    plt.grid(True)
    
    # 3. 绘制预测结果和真实值 - 缩放后的局部数据（最后100个样本）
    plt.subplot(2, 2, 3)
    last_n = min(100, len(actuals))
    plt.plot(actuals[-last_n:], label='Actual')
    plt.plot(predictions[-last_n:], label='Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    if has_chinese_font:
        plt.title(f'真实值与预测值对比 (最后{last_n}个样本)', fontproperties=chinese_font)
    else:
        plt.title(f'Actual vs Predicted (Last {last_n} Samples)')
    plt.legend()
    plt.grid(True)
    
    # 4. 绘制预测误差
    plt.subplot(2, 2, 4)
    error = actuals.flatten() - predictions.flatten()
    plt.scatter(np.arange(len(error)), error, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Samples')
    plt.ylabel('Error (Actual - Predicted)')
    if has_chinese_font:
        plt.title('预测误差分布', fontproperties=chinese_font)
    else:
        plt.title('Prediction Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('time_series_results.png')
    plt.close()
    
    # 生成更详细的残差分析图
    plt.figure(figsize=(12, 10))
    
    # 1. 残差对预测值的散点图
    plt.subplot(2, 1, 1)
    plt.scatter(predictions.flatten(), error, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    
    # 2. 残差直方图
    plt.subplot(2, 1, 2)
    plt.hist(error, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png')
    plt.close() 