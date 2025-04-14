import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
import os
import sys
import time

# 从简化版模型导入基础组件
try:
    from simplified_model import (TimeSeriesDataset, handle_outliers, 
                              create_safe_features, safe_scaling, 
                              create_sequences, train_model_safely, 
                              evaluate_model_safely, plot_results_safely)
except ImportError:
    print("警告: 无法导入简化版模型的组件，将重新定义这些函数")
    # 在这里可以重新定义这些函数

# 注意力机制
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_size=None):
        super(AttentionLayer, self).__init__()
        if attention_size is None:
            attention_size = hidden_size
        self.attention_size = attention_size
        
        # 投影层
        self.proj_query = nn.Linear(hidden_size, attention_size)
        self.proj_key = nn.Linear(hidden_size, attention_size)
        self.proj_value = nn.Linear(hidden_size, hidden_size)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.proj_query.weight)
        nn.init.xavier_uniform_(self.proj_key.weight)
        nn.init.xavier_uniform_(self.proj_value.weight)
        nn.init.zeros_(self.proj_query.bias)
        nn.init.zeros_(self.proj_key.bias)
        nn.init.zeros_(self.proj_value.bias)
        
        # 用于缩放点积注意力
        self.scale = torch.sqrt(torch.FloatTensor([attention_size]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 检查输入数据的有效性
        if torch.isnan(query).any() or torch.isinf(query).any():
            print("警告: 注意力层输入query包含NaN或Inf")
            query = torch.nan_to_num(query, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 投影查询、键、值
        Q = self.proj_query(query)  # (batch_size, seq_len, attention_size)
        K = self.proj_key(key)      # (batch_size, seq_len, attention_size)
        V = self.proj_value(value)  # (batch_size, seq_len, hidden_size)
        
        # 缩放点积注意力
        scale = self.scale.to(query.device)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (batch_size, seq_len, seq_len)
        
        # 检查注意力分数的有效性
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("警告: 注意力分数包含NaN或Inf")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 如果有掩码，应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 检查注意力权重的有效性
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            print("警告: 注意力权重包含NaN或Inf")
            attention_weights = torch.softmax(torch.zeros_like(scores), dim=-1)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)  # (batch_size, seq_len, hidden_size)
        
        return context, attention_weights

# 卷积模块
class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(ConvLayer, self).__init__()
        # 修改padding计算方式，确保输出序列长度与输入相同
        padding = (kernel_size - 1) // 2
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, 
                     padding=padding, dilation=dilation),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # 初始化权重
        for m in self.conv_layer:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x):
        """
        输入: [batch_size, seq_len, input_channels]
        输出: [batch_size, seq_len, output_channels]
        """
        # 转换维度顺序以适应Conv1d
        x = x.permute(0, 2, 1)  # [batch, input_channels, seq_len]
        
        # 检查输入数据的有效性
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告: 卷积层输入包含NaN或Inf")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 卷积操作
        x = self.conv_layer(x)  # [batch, output_channels, seq_len]
        
        # 转回原始维度顺序
        x = x.permute(0, 2, 1)  # [batch, seq_len, output_channels]
        
        return x

# 增强的LSTM模型
class EnhancedTemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(EnhancedTemporalModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # 卷积层
        self.conv_layer = ConvLayer(hidden_size, hidden_size, kernel_size=3, dropout=dropout)
        
        # LSTM层
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # 注意力层
        self.attention = AttentionLayer(hidden_size)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # 残差连接
        self.res_conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        # 输入投影层初始化
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # LSTM初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # 全连接层初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        # 残差连接初始化
        nn.init.xavier_uniform_(self.res_conv.weight)
        nn.init.zeros_(self.res_conv.bias)
        
    def forward(self, x):
        """
        输入: [batch_size, seq_len, input_size]
        输出: [batch_size, output_size]
        """
        batch_size, seq_len, _ = x.shape
            
        # 输入投影
        x_proj = self.input_proj(x)
        
        # 残差连接的预处理
        x_res = x.permute(0, 2, 1)  # [batch, input_size, seq_len]
        x_res = self.res_conv(x_res)  # [batch, hidden_size, seq_len]
        x_res = x_res.permute(0, 2, 1)  # [batch, seq_len, hidden_size]
        
        # 卷积特征提取
        x_conv = self.conv_layer(x_proj)
        
        # 残差连接
        x_combined = x_conv + x_res
        
        # LSTM处理
        lstm_out, _ = self.lstm(x_combined)
        
        # 注意力机制
        context, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 只使用最后一个时间步的输出进行预测
        out = context[:, -1, :]
        
        # 全连接层
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
            
        return out

# 安全的数据序列生成
def create_sequences(data, seq_length, target_col_idx=0):
    """创建时间序列的滑动窗口，确保数值稳定"""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # 获取输入序列
        X.append(data[i:i+seq_length])
        # 获取目标值（下一个时间步的值）
        y.append(data[i+seq_length, target_col_idx])
    
    X_array = np.array(X)
    y_array = np.array(y)
    
    # 检查生成的序列是否包含NaN或无限值
    if np.isnan(X_array).any() or np.isnan(y_array).any():
        print("警告：生成的序列包含NaN值")
        X_array = np.nan_to_num(X_array, nan=0.0)
        y_array = np.nan_to_num(y_array, nan=0.0)
        
    if np.isinf(X_array).any() or np.isinf(y_array).any():
        print("警告：生成的序列包含无限值")
        X_array = np.nan_to_num(X_array, posinf=1.0, neginf=-1.0)
        y_array = np.nan_to_num(y_array, posinf=1.0, neginf=-1.0)
    
    # 确保y的维度正确 [samples, 1]
    if len(y_array.shape) == 1:
        y_array = y_array.reshape(-1, 1)
    
    return X_array, y_array

def main():
    """训练增强版模型的主函数"""
    start_time = time.time()
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    try:
        file_path = '../total.csv'
        df = pd.read_csv(file_path)
        print(f"数据加载成功: {df.shape}行 x {df.columns.shape[0]}列")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 数据预处理
    sales_column = '成交商品件数'
    
    # 处理异常值
    try:
        df[sales_column] = handle_outliers(df, sales_column, method='winsorize')
    except Exception as e:
        print(f"异常值处理失败: {e}")
        df[sales_column] = np.log1p(df[sales_column])
    
    # 特征工程
    try:
        df_features = create_safe_features(df, sales_column)
        print(f"特征工程成功，特征数量: {df_features.shape[1]}")
    except Exception as e:
        print(f"特征工程失败: {e}")
        df_features = df[[sales_column]].copy()
    
    # 数据缩放
    try:
        scaled_data, scaler = safe_scaling(df_features)
        print(f"数据缩放成功")
    except Exception as e:
        print(f"数据缩放失败: {e}")
        scaled_data = df_features.values
        scaler = None
    
    # 目标变量单独缩放
    try:
        y_data = df_features[sales_column].values.reshape(-1, 1)
        y_scaled, scaler_y = safe_scaling(y_data)
    except Exception as e:
        print(f"目标变量缩放失败: {e}")
        y_scaled = y_data
        scaler_y = None
    
    # 创建序列数据
    try:
        seq_length = 14
        X, y = create_sequences(scaled_data, seq_length, 
                              target_col_idx=df_features.columns.get_loc(sales_column))
        print(f"序列数据创建成功: X形状={X.shape}, y形状={y.shape}")
    except Exception as e:
        print(f"序列数据创建失败: {e}")
        return
    
    # 划分数据集
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, shuffle=False)
        
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
        
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print("数据集划分和加载器创建成功")
    except Exception as e:
        print(f"数据集划分失败: {e}")
        return
    
    # 创建增强版模型
    try:
        input_size = X.shape[2]
        hidden_size = 128    # 增加隐藏层大小，提升模型容量
        num_layers = 2       # 使用2层LSTM，增强序列建模能力
        output_size = 1
        dropout = 0.3        # 适当减小dropout
        
        model = EnhancedTemporalModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
        
        print("增强版模型创建成功:")
        print(model)
    except Exception as e:
        print(f"模型创建失败: {e}")
        return
    
    # 定义损失函数和优化器
    # 使用自定义损失函数，加强对峰值的关注
    def custom_loss(pred, target):
        mse_loss = nn.MSELoss()(pred, target)
        # 计算峰值权重
        peak_weight = 1.0 + torch.abs(target) / torch.mean(torch.abs(target))
        weighted_mse = torch.mean(peak_weight * (pred - target) ** 2)
        return 0.7 * mse_loss + 0.3 * weighted_mse

    criterion = custom_loss
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0005,     # 减小学习率，使训练更稳定
        weight_decay=1e-4  # 减小权重衰减
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=200,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # 训练模型
    try:
        train_losses, val_losses = train_model_safely(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,  # 添加调度器
            device=device,
            num_epochs=200,
            patience=25,       # 增加耐心值
            gradient_clip=1.0
        )
        
        print("模型训练完成")
    except Exception as e:
        print(f"模型训练失败: {e}")
        return
    
    # 加载最佳模型
    try:
        model.load_state_dict(torch.load('best_simplified_model.pth'))
        print("加载最佳模型成功")
    except Exception as e:
        print(f"加载最佳模型失败: {e}")
    
    # 评估模型
    try:
        predictions, actuals, mse, rmse, r2 = evaluate_model_safely(
            model=model,
            test_loader=test_loader,
            scaler_y=scaler_y,
            device=device
        )
        
        print("模型评估完成")
    except Exception as e:
        print(f"模型评估失败: {e}")
        predictions, actuals = None, None
    
    # 结果可视化
    try:
        plot_results_safely(predictions, actuals, train_losses, val_losses)
        print("结果可视化完成")
    except Exception as e:
        print(f"结果可视化失败: {e}")
    
    # 保存预测结果
    try:
        if predictions is not None and actuals is not None:
            results_df = pd.DataFrame({
                'Actual': actuals,
                'Predicted': predictions
            })
            results_df.to_csv('enhanced_prediction_results.csv', index=False)
            print("预测结果已保存")
    except Exception as e:
        print(f"保存预测结果失败: {e}")
    
    # 保存模型
    try:
        torch.save(model.state_dict(), 'enhanced_model.pth')
        print("增强版模型已保存")
    except Exception as e:
        print(f"保存模型失败: {e}")
    
    end_time = time.time()
    print(f"总运行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
