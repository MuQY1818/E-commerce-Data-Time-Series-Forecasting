import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
from simplified_model import (
    SimplifiedTemporalModel, TimeSeriesDataset, 
    create_sequences, scale_data_safely, create_safe_features,
    evaluate_model_safely, plot_results_safely
)
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class HyperparameterTuner:
    def __init__(self, X, y, device):
        # 数据验证
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("输入数据包含NaN值")
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("输入数据包含无限值")
            
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.device = device
        self.best_params = None
        self.best_score = float('inf')
        self.results = []
        
    def create_model(self, params):
        model = SimplifiedTemporalModel(
            input_size=self.X.shape[2],
            hidden_size=params['hidden_size'],
            output_size=params['output_size'],
            dropout=params['dropout']
        ).to(self.device)
        return model
    
    def train_model(self, model, train_loader, val_loader, params):
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params['learning_rate'],
            epochs=params['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 使用tqdm显示训练进度
        epoch_pbar = tqdm(range(params['epochs']), desc='训练进度', leave=False)
        for epoch in epoch_pbar:
            # 训练阶段
            model.train()
            train_loss = 0
            batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{params["epochs"]}', leave=False)
            for X_batch, y_batch in batch_pbar:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                
                # 确保输出和目标维度一致
                if output.dim() != y_batch.dim():
                    if output.dim() == 2:
                        output = output.unsqueeze(-1)
                    elif y_batch.dim() == 2:
                        y_batch = y_batch.unsqueeze(-1)
                
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 验证阶段
            model.eval()
            val_loss = 0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.float().to(self.device)
                    y_val = y_val.float().to(self.device)
                    
                    output = model(X_val)
                    
                    # 确保输出和目标维度一致
                    if output.dim() != y_val.dim():
                        if output.dim() == 2:
                            output = output.unsqueeze(-1)
                        elif y_val.dim() == 2:
                            y_val = y_val.unsqueeze(-1)
                    
                    val_loss += criterion(output, y_val).item()
                    predictions.extend(output.cpu().numpy())
                    actuals.extend(y_val.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # 安全计算评估指标
            if len(predictions) > 0 and len(actuals) > 0:
                predictions = np.array(predictions, dtype=np.float32)
                actuals = np.array(actuals, dtype=np.float32)
                
                # 计算平滑度（使用安全的方式）
                try:
                    pred_diff = np.diff(predictions.flatten())
                    smoothness = np.mean(np.abs(pred_diff)) if len(pred_diff) > 0 else 0
                except:
                    smoothness = 0
                
                # 计算敏感度
                try:
                    sensitivity = np.mean(np.abs(predictions - actuals))
                except:
                    sensitivity = 0
                
                # 综合评分（添加安全检查）
                try:
                    combined_score = val_loss + 0.3 * smoothness + 0.7 * sensitivity
                    if np.isnan(combined_score) or np.isinf(combined_score):
                        combined_score = float('inf')
                except:
                    combined_score = float('inf')
                
                epoch_pbar.set_postfix({
                    'val_loss': f'{val_loss:.4f}',
                    'smoothness': f'{smoothness:.4f}',
                    'sensitivity': f'{sensitivity:.4f}'
                })
            else:
                print(f"警告：Epoch {epoch+1} 没有有效的预测结果")
                combined_score = float('inf')
                smoothness = 0
                sensitivity = 0
            
            if combined_score < best_val_loss:
                best_val_loss = combined_score
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= params['patience']:
                print(f"早停：{patience_counter}轮未改善")
                break
                
        return best_val_loss, smoothness, sensitivity
    
    def tune(self, param_grid):
        # 数据集划分
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False)
        
        # 创建数据加载器
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        # 生成所有参数组合
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in product(*param_grid.values())]
        
        # 使用tqdm显示参数搜索进度
        param_pbar = tqdm(param_combinations, desc='参数搜索进度')
        for params in param_pbar:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=params['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=params['batch_size'], 
                shuffle=False
            )
            
            # 训练和评估模型
            model = self.create_model(params)
            val_loss, smoothness, sensitivity = self.train_model(
                model, train_loader, val_loader, params)
            
            # 记录结果
            result = {
                'params': params,
                'val_loss': val_loss if not np.isnan(val_loss) and not np.isinf(val_loss) else float('inf'),
                'smoothness': smoothness if not np.isnan(smoothness) and not np.isinf(smoothness) else 0,
                'sensitivity': sensitivity if not np.isnan(sensitivity) and not np.isinf(sensitivity) else 0,
                'combined_score': val_loss + 0.3 * smoothness + 0.7 * sensitivity if not np.isnan(val_loss) and not np.isinf(val_loss) else float('inf')
            }
            self.results.append(result)
            
            # 更新最佳参数
            if result['combined_score'] < self.best_score:
                self.best_score = result['combined_score']
                self.best_params = params
            
            # 更新进度条描述
            param_pbar.set_postfix({
                'val_loss': f'{result["val_loss"]:.4f}',
                'smoothness': f'{result["smoothness"]:.4f}',
                'sensitivity': f'{result["sensitivity"]:.4f}'
            })
        
        # 保存调参结果
        self.save_results()
        return self.best_params
    
    def save_results(self):
        # 将结果保存为CSV文件
        results_df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'tuning_results_{timestamp}.csv', index=False)
        
        # 保存最佳参数
        with open(f'best_params_{timestamp}.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)

def create_safe_features(df, target_col):
    """创建安全的时间序列特征
    
    Args:
        df (pd.DataFrame): 包含日期和目标列的数据框
        target_col (str): 目标列名
        
    Returns:
        pd.DataFrame: 包含特征的数据框
    """
    features = pd.DataFrame()
    
    try:
        # 基本特征
        features['target'] = df[target_col]
        
        # 移动平均特征
        for window in [3, 7, 14]:
            features[f'ma_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            # 处理可能的NaN值
            features[f'ma_{window}'] = features[f'ma_{window}'].fillna(method='ffill').fillna(method='bfill')
        
        # 移动标准差特征
        for window in [7, 14]:
            features[f'std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
            # 处理可能的NaN值
            features[f'std_{window}'] = features[f'std_{window}'].fillna(method='ffill').fillna(method='bfill')
        
        # 移动最大最小值特征
        for window in [7, 14]:
            features[f'max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
            features[f'min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
            # 处理可能的NaN值
            features[f'max_{window}'] = features[f'max_{window}'].fillna(method='ffill').fillna(method='bfill')
            features[f'min_{window}'] = features[f'min_{window}'].fillna(method='ffill').fillna(method='bfill')
        
        # 差分特征
        features['diff_1'] = df[target_col].diff().fillna(0)
        features['diff_7'] = df[target_col].diff(7).fillna(0)
        
        # 周期性特征
        features['day_of_week'] = pd.to_datetime(df['日期']).dt.dayofweek
        features['day_of_month'] = pd.to_datetime(df['日期']).dt.day
        features['month'] = pd.to_datetime(df['日期']).dt.month
        
        # 趋势特征
        features['trend'] = np.arange(len(df))
        
        # 检查并处理无效值
        for col in features.columns:
            if features[col].isnull().any():
                print(f"警告：特征 {col} 包含空值，使用前向填充和后向填充处理")
                features[col] = features[col].fillna(method='ffill').fillna(method='bfill')
            if np.isinf(features[col]).any():
                print(f"警告：特征 {col} 包含无限值，使用有限值替换")
                features[col] = features[col].replace([np.inf, -np.inf], np.nan)
                features[col] = features[col].fillna(method='ffill').fillna(method='bfill')
        
        # 打印特征统计信息
        print("\n特征统计信息:")
        for col in features.columns:
            print(f"{col}: 最小值={features[col].min():.4f}, 最大值={features[col].max():.4f}, "
                  f"平均值={features[col].mean():.4f}, 标准差={features[col].std():.4f}")
        
        print(f"\n创建了 {len(features.columns)} 个特征")
        return features
        
    except Exception as e:
        print(f"特征工程过程中出错: {str(e)}")
        raise

def scale_data_safely(df):
    """安全地缩放数据
    
    Args:
        df (pd.DataFrame): 输入数据框
        
    Returns:
        tuple: (缩放后的数据数组, 缩放器对象)
    """
    try:
        # 创建缩放器
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 确保数据是数值类型
        df_numeric = df.select_dtypes(include=[np.number])
        
        # 检查并处理无效值
        if df_numeric.isnull().any().any():
            print("警告：数据包含空值，使用前向填充和后向填充处理")
            df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
        
        if np.isinf(df_numeric).any().any():
            print("警告：数据包含无限值，使用有限值替换")
            df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
            df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
        
        # 缩放数据
        scaled_data = scaler.fit_transform(df_numeric)
        
        # 再次检查缩放后的数据
        if np.isnan(scaled_data).any():
            print("警告：缩放后的数据包含NaN值，使用0填充")
            scaled_data = np.nan_to_num(scaled_data, nan=0.0)
        
        if np.isinf(scaled_data).any():
            print("警告：缩放后的数据包含无限值，使用有限值替换")
            scaled_data = np.nan_to_num(scaled_data, posinf=1.0, neginf=0.0)
        
        # 打印数据统计信息
        print("\n数据缩放统计信息:")
        print(f"原始数据 - 最小值: {df_numeric.min().min():.4f}, 最大值: {df_numeric.max().max():.4f}")
        print(f"缩放后数据 - 最小值: {scaled_data.min():.4f}, 最大值: {scaled_data.max():.4f}")
        
        return scaled_data, scaler
        
    except Exception as e:
        print(f"数据缩放过程中出错: {str(e)}")
        raise

def create_sequences(data, seq_length, pred_length=7):
    """创建序列数据，用于预测连续多天的数据
    
    Args:
        data (np.ndarray): 输入数据数组
        seq_length (int): 输入序列长度
        pred_length (int): 预测序列长度
        
    Returns:
        tuple: (X序列数据, y目标值)
    """
    try:
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[(i + seq_length):(i + seq_length + pred_length), 0])  # 预测未来7天的数据
            
        return np.array(X), np.array(y)
        
    except Exception as e:
        print(f"创建序列数据时出错: {str(e)}")
        raise

def main():
    try:
        # 加载数据
        print("正在加载数据...")
        df = pd.read_csv('total_cleaned.csv')
        
        # 检查并处理日期
        print("正在处理日期...")
        df['日期'] = pd.to_datetime(df['日期'])
        date_range = pd.date_range(start=df['日期'].min(), end=df['日期'].max())
        if len(date_range) != len(df):
            print(f"发现日期缺失，正在填充...")
            df = df.set_index('日期').reindex(date_range).reset_index()
            df = df.rename(columns={'index': '日期'})
            df['成交商品件数'] = df['成交商品件数'].ffill().bfill()
        
        # 创建特征
        print("正在创建特征...")
        features = create_safe_features(df, '成交商品件数')
        print(f"创建了 {features.shape[1]} 个特征")
        print(f"特征形状: {features.shape}")
        
        # 缩放数据
        print("正在缩放数据...")
        scaled_data, scaler = scale_data_safely(features)
        
        # 创建序列
        print("正在创建序列...")
        seq_length = 14  # 使用14天的数据
        pred_length = 7  # 预测未来7天
        X, y = create_sequences(scaled_data, seq_length, pred_length)
        print(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 定义超参数搜索空间（基于最佳参数进行微调）
        param_grid = {
            'hidden_size': [256],  # 固定为最佳值
            'learning_rate': [0.001],  # 固定为最佳值
            'batch_size': [64],  # 固定为最佳值
            'dropout': [0.2],  # 固定为最佳值
            'epochs': [100],  # 增加训练轮数
            'patience': [15],  # 增加早停耐心值
            'gradient_clip': [0.5],  # 固定为最佳值
            'weight_decay': [0.01],  # 固定为最佳值
            'output_size': [7]  # 固定输出为7天
        }
        
        # 创建调优器实例
        tuner = HyperparameterTuner(X, y, device)
        
        # 执行超参数调优
        print("开始超参数调优...")
        best_params = tuner.tune(param_grid)
        print(f"最佳参数: {best_params}")
        
        # 保存结果
        tuner.save_results()
        
        print("\n最佳参数组合:")
        print(json.dumps(best_params, indent=2))
        
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 