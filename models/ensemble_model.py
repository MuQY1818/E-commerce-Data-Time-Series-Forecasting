import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from sklearn.ensemble import VotingRegressor
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

def prepare_ensemble_data(df, target_col='成交商品件数', train_ratio=0.7, val_ratio=0.15):
    """准备集成模型的数据"""
    # 创建高级特征
    df_features = create_advanced_features(df, target_col)
    
    # 准备Prophet数据
    prophet_df = pd.DataFrame({
        'ds': df_features['日期'],
        'y': df_features[target_col]
    })
    
    # 准备其他模型的特征
    feature_cols = [col for col in df_features.columns 
                   if col not in ['日期', target_col, 'dayofweek']]
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    # 数据分割为训练集、验证集和测试集
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

class EnsembleNN(nn.Module):
    def __init__(self, n_models, hidden_size=64):
        super(EnsembleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_models, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class EnsembleDataset(Dataset):
    def __init__(self, predictions, targets):
        self.predictions = torch.FloatTensor(predictions)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.predictions[idx], self.targets[idx]

def train_simple_nn(X_train, y_train, X_val, y_val, epochs=500, batch_size=32, lr=0.001, patience=20):
    """训练简单神经网络"""
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

def train_nn_ensemble(base_predictions, y_true, val_predictions, y_val, epochs=500, batch_size=32, lr=0.001, patience=20):
    """训练神经网络集成器"""
    # 准备数据
    train_dataset = EnsembleDataset(base_predictions, y_true)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = EnsembleDataset(val_predictions, y_val)
    
    # 初始化模型
    model = EnsembleNN(n_models=base_predictions.shape[1])
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
            val_outputs = model(torch.FloatTensor(val_predictions))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_val).view(-1, 1))
        
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

def make_ensemble_predictions(prophet_model, other_models, simple_nn, X_test, prophet_test, y_scaler, nn_ensemble=None):
    """使用神经网络生成集成预测"""
    # Prophet预测
    future = prophet_model.make_future_dataframe(periods=len(prophet_test))
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast.tail(len(prophet_test))['yhat'].values
    
    # 其他模型预测
    other_preds = []
    for model in other_models:
        pred = model.predict(X_test)
        other_preds.append(pred)
    
    # SimpleNN预测
    with torch.no_grad():
        simple_nn.eval()
        simple_nn_pred = simple_nn(torch.FloatTensor(X_test.values)).numpy()
    other_preds.append(simple_nn_pred.flatten())
    
    # 将所有预测值转换为相同的形状
    prophet_pred = prophet_pred.reshape(-1, 1)
    other_preds = [pred.reshape(-1, 1) for pred in other_preds]
    
    # 对Prophet预测值进行缩放以匹配其他模型的尺度
    prophet_pred_scaled = y_scaler.transform(prophet_pred)
    
    # 组合所有预测结果
    all_preds = np.concatenate([prophet_pred_scaled] + other_preds, axis=1)
    
    if nn_ensemble is not None:
        # 使用神经网络进行集成
        with torch.no_grad():
            nn_ensemble.eval()
            ensemble_pred_scaled = nn_ensemble(torch.FloatTensor(all_preds)).numpy()
    else:
        # 如果没有神经网络模型，使用简单平均
        ensemble_pred_scaled = np.mean(all_preds, axis=1).reshape(-1, 1)
    
    # 反缩放预测值
    ensemble_pred = y_scaler.inverse_transform(ensemble_pred_scaled).flatten()
    
    return ensemble_pred

def plot_ensemble_results(y_true, y_pred, title="集成模型预测结果"):
    """绘制集成模型预测结果"""
    plt.figure(figsize=(15, 10))
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 第一个子图：实际值和预测值
    ax1.plot(y_true, label='实际值', color='#1f77b4', 
             marker='o', markersize=4, linewidth=1.5)
    ax1.plot(y_pred, label='预测值', color='#ff7f0e', 
             marker='s', markersize=4, linewidth=1.5)
    ax1.set_title(title, fontsize=14, pad=20)
    ax1.set_xlabel('样本索引', fontsize=12)
    ax1.set_ylabel('销量', fontsize=12)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 第二个子图：预测误差
    errors = y_true - y_pred
    ax2.plot(errors, color='#2ca02c', label='预测误差',
             marker='o', markersize=4, linewidth=1.5)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax2.set_title('预测误差', fontsize=14, pad=20)
    ax2.set_xlabel('样本索引', fontsize=12)
    ax2.set_ylabel('误差值', fontsize=12)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('models/ensemble_predictions.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()

def evaluate_ensemble(y_true, y_pred):
    """评估集成模型性能"""
    # 计算相对误差
    relative_error = np.abs(y_true - y_pred) / y_true
    mape = np.mean(relative_error) * 100
    
    # 计算其他指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\n集成模型评估指标：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    print(f"R² 分数: {r2:.4f}")
    
    return mse, rmse, mae, r2, mape

def main():
    # 加载数据
    df = pd.read_csv('total_cleaned.csv')
    
    # 准备数据
    (X_train, y_train, X_val, y_val, X_test, y_test), \
    (prophet_train, prophet_val, prophet_test), \
    (X_scaler, y_scaler), feature_cols = prepare_ensemble_data(df)
    
    print("数据集大小:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 训练Prophet模型
    print("\n训练Prophet模型...")
    prophet_model = train_prophet_model(prophet_train)
    
    # 训练其他模型
    print("\n训练其他模型...")
    xgb_model, lgb_model, rf_model = train_ensemble_models(X_train, y_train)
    
    # 训练SimpleNN
    print("\n训练SimpleNN...")
    simple_nn = train_simple_nn(X_train.values, y_train, X_val.values, y_val)
    
    # 获取所有模型在验证集上的预测
    val_preds = []
    for model in [prophet_model] + [xgb_model, lgb_model, rf_model]:
        if isinstance(model, Prophet):
            future = model.make_future_dataframe(periods=len(X_val))
            forecast = model.predict(future)
            pred = y_scaler.transform(forecast.tail(len(X_val))['yhat'].values.reshape(-1, 1))
        else:
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
    nn_ensemble = train_nn_ensemble(val_preds, y_val, val_preds, y_val)
    
    # 使用神经网络生成最终预测
    ensemble_predictions = make_ensemble_predictions(
        prophet_model, [xgb_model, lgb_model, rf_model], simple_nn,
        X_test, prophet_test, y_scaler, nn_ensemble
    )
    
    # 获取实际值
    y_test_orig = y_scaler.inverse_transform(y_test).flatten()
    
    # 评估模型
    evaluate_ensemble(y_test_orig, ensemble_predictions)
    
    # 绘制预测结果
    plot_ensemble_results(y_test_orig, ensemble_predictions)
    
    # 保存模型和结果
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/ensemble_xgb.joblib')
    joblib.dump(lgb_model, 'models/ensemble_lgb.joblib')
    joblib.dump(rf_model, 'models/ensemble_rf.joblib')
    joblib.dump(X_scaler, 'models/ensemble_X_scaler.joblib')
    joblib.dump(y_scaler, 'models/ensemble_y_scaler.joblib')
    
    # 保存神经网络模型
    torch.save(simple_nn.state_dict(), 'models/ensemble_simple_nn.pth')
    torch.save(nn_ensemble.state_dict(), 'models/ensemble_nn.pth')
    
    # 保存预测结果
    results_df = pd.DataFrame({
        '实际值': y_test_orig,
        '预测值': ensemble_predictions,
        '预测误差': y_test_orig - ensemble_predictions
    })
    results_df.to_csv('models/ensemble_results.csv', index=False)
    
    print("\n模型和结果已保存到models目录")

if __name__ == "__main__":
    main() 