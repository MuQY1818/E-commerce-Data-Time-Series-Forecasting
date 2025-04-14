import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from simplified_model import create_features  # 复用特征工程
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

def prepare_data(df, target_col='成交商品件数', train_ratio=0.8, val_ratio=0.1, use_last_year=True):
    """准备数据，复用之前的数据处理逻辑"""
    df_copy = df.copy()
    
    if use_last_year:
        df_copy['日期'] = pd.to_datetime(df_copy['日期'])
        last_date = df_copy['日期'].max()
        one_year_ago = last_date - pd.DateOffset(years=1)
        df_copy = df_copy[df_copy['日期'] >= one_year_ago]
        print(f"\n使用从 {one_year_ago.date()} 到 {last_date.date()} 的数据")
        print(f"数据集大小: {len(df_copy)} 条记录")
    
    # 分离特征和目标
    X = df_copy.drop([target_col, '日期'], axis=1).values
    y = df_copy[target_col].values
    
    # 使用RobustScaler进行缩放
    X_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
    
    # 计算划分点
    total_size = len(X_scaled)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # 划分数据集
    X_train = X_scaled[:train_size]
    X_val = X_scaled[train_size:train_size+val_size]
    X_test = X_scaled[train_size+val_size:]
    
    y_train = y_scaled[:train_size]
    y_val = y_scaled[train_size:train_size+val_size]
    y_test = y_scaled[train_size+val_size:]
    
    print(f"\n数据集划分：")
    print(f"训练集: {len(X_train)} 样本 ({len(X_train)/total_size*100:.1f}%)")
    print(f"验证集: {len(X_val)} 样本 ({len(X_val)/total_size*100:.1f}%)")
    print(f"测试集: {len(X_test)} 样本 ({len(X_test)/total_size*100:.1f}%)")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), (X_scaler, y_scaler)

def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} 评估指标：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"R² 分数: {r2:.4f}")
    
    return mse, rmse, mae, r2

def plot_predictions(y_true, y_pred, model_name):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='实际值', alpha=0.7)
    plt.plot(y_pred, label='预测值', alpha=0.7)
    plt.title(f'{model_name} 预测结果对比')
    plt.xlabel('样本')
    plt.ylabel('销量')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_predictions.png')
    plt.close()

def train_xgboost(X_train, y_train, X_val, y_val):
    """训练XGBoost模型"""
    model = xgb.XGBRegressor(
        n_estimators=1000,  # 设置较大的数值，让early_stopping生效
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10  # 移到这里
    )
    
    model.fit(
        X_train, y_train.ravel(),
        eval_set=[(X_val, y_val.ravel())],
        verbose=False
    )
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """训练LightGBM模型"""
    model = lgb.LGBMRegressor(
        n_estimators=1000,  # 设置较大的数值，让early_stopping生效
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_round=10  # 修改参数名
    )
    
    model.fit(
        X_train, y_train.ravel(),
        eval_set=[(X_val, y_val.ravel())]
    )
    
    return model

def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train.ravel())
    return model

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_nn(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """训练全连接神经网络"""
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = SimpleNN(X_train.shape[1])
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
    
    return model

def evaluate_nn(model, X_test, y_test):
    """评估神经网络模型"""
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
    
    # 确保y_pred和y_test的维度一致
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    
    return y_pred  # 只返回预测值，让main函数处理评估

def plot_all_results(y_test, y_pred_xgb, y_pred_lgb, y_pred_rf, y_pred_nn):
    """绘制所有模型的预测结果对比图"""
    plt.figure(figsize=(20, 12))
    
    # 设置样式
    plt.style.use('default')  # 使用默认样式
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 绘制实际值
    plt.plot(y_test, label='实际值', color=colors[0], linewidth=2, marker='o', markersize=6, alpha=0.8)
    
    # 绘制预测值
    plt.plot(y_pred_xgb, label='XGBoost预测', color=colors[1], linestyle='--', marker='s', markersize=6, alpha=0.7)
    plt.plot(y_pred_lgb, label='LightGBM预测', color=colors[2], linestyle='--', marker='^', markersize=6, alpha=0.7)
    plt.plot(y_pred_rf, label='随机森林预测', color=colors[3], linestyle='--', marker='D', markersize=6, alpha=0.7)
    plt.plot(y_pred_nn, label='神经网络预测', color=colors[4], linestyle='--', marker='*', markersize=6, alpha=0.7)
    
    # 添加标题和标签
    plt.title('不同模型预测结果对比', fontsize=16, pad=20)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('销售额', fontsize=12)
    
    # 设置图例
    plt.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加背景色
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('white')
    
    # 确保models目录存在
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存图片
    plt.savefig('models/all_models_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("\n对比图已保存到 models/all_models_comparison.png")

def main():
    # 加载数据
    df = pd.read_csv('total_cleaned.csv')
    
    # 特征工程
    df_features = create_features(df)
    
    # 准备数据
    (X_train, y_train, X_val, y_val, X_test, y_test), (_, y_scaler) = prepare_data(
        df_features, 
        target_col='成交商品件数',
        train_ratio=0.8,
        val_ratio=0.1,
        use_last_year=False  # 使用所有数据
    )
    
    # 训练和评估XGBoost
    print("\n训练XGBoost模型...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    y_pred_xgb = xgb_model.predict(X_test)
    # 反缩放
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred_xgb_orig = y_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1)).reshape(-1)
    mse_xgb, rmse_xgb, mae_xgb, r2_xgb = evaluate_model(y_test_orig, y_pred_xgb_orig, "XGBoost")
    print(f"XGBoost - MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, R²: {r2_xgb:.4f}")
    
    # 训练和评估LightGBM
    print("\n训练LightGBM模型...")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    y_pred_lgb = lgb_model.predict(X_test)
    # 反缩放
    y_pred_lgb_orig = y_scaler.inverse_transform(y_pred_lgb.reshape(-1, 1)).reshape(-1)
    mse_lgb, rmse_lgb, mae_lgb, r2_lgb = evaluate_model(y_test_orig, y_pred_lgb_orig, "LightGBM")
    print(f"LightGBM - MSE: {mse_lgb:.2f}, RMSE: {rmse_lgb:.2f}, MAE: {mae_lgb:.2f}, R²: {r2_lgb:.4f}")
    
    # 训练和评估随机森林
    print("\n训练随机森林模型...")
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    # 反缩放
    y_pred_rf_orig = y_scaler.inverse_transform(y_pred_rf.reshape(-1, 1)).reshape(-1)
    mse_rf, rmse_rf, mae_rf, r2_rf = evaluate_model(y_test_orig, y_pred_rf_orig, "RandomForest")
    print(f"随机森林 - MSE: {mse_rf:.2f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.4f}")
    
    # 训练和评估神经网络
    print("\n训练神经网络模型...")
    nn_model = train_nn(X_train, y_train, X_val, y_val)
    y_pred_nn = evaluate_nn(nn_model, X_test, y_test)
    # 反缩放
    y_pred_nn_orig = y_scaler.inverse_transform(y_pred_nn.reshape(-1, 1)).reshape(-1)
    mse_nn, rmse_nn, mae_nn, r2_nn = evaluate_model(y_test_orig, y_pred_nn_orig, "NeuralNetwork")
    print(f"神经网络 - MSE: {mse_nn:.2f}, RMSE: {rmse_nn:.2f}, MAE: {mae_nn:.2f}, R²: {r2_nn:.4f}")
    
    # 绘制所有模型的预测结果对比图（使用原始尺度的数据）
    plot_all_results(y_test_orig, y_pred_xgb_orig, y_pred_lgb_orig, y_pred_rf_orig, y_pred_nn_orig)
    
    # 保存模型
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    joblib.dump(lgb_model, 'models/lgb_model.joblib')
    joblib.dump(rf_model, 'models/rf_model.joblib')
    torch.save(nn_model.state_dict(), 'models/nn_model.pth')
    joblib.dump(y_scaler, 'models/y_scaler.joblib')
    
    print("\n所有模型已保存到models目录")

if __name__ == "__main__":
    main() 