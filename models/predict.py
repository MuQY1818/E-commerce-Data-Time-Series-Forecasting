import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import matplotlib.pyplot as plt
import argparse
from matplotlib.font_manager import FontProperties

# 添加父目录到系统路径，以便导入模型模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.time_series_model import HybridTemporalModel

def load_model(model_path, input_size, hidden_size, num_layers, output_size, seq_len):
    """加载训练好的模型"""
    model = HybridTemporalModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        seq_len=seq_len
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def prepare_data(file_path, window_size):
    """准备预测所需的数据"""
    # 加载CSV文件
    df = pd.read_csv(file_path)
    
    # 提取销售数据
    sales_data = df.iloc[:, 1].values
    
    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaled = scaler.fit_transform(sales_data.reshape(-1, 1))
    
    # 获取最后window_size天的数据作为预测输入
    last_window = sales_scaled[-window_size:].reshape(1, window_size, 1)
    
    return torch.FloatTensor(last_window), scaler, df.iloc[-window_size:, 0].values

def predict_next_day(model, input_data, scaler):
    """预测下一天的销售量"""
    with torch.no_grad():
        prediction = model(input_data)
        # 将预测结果转换回原始尺度
        prediction = scaler.inverse_transform(prediction.numpy())
        
    return prediction[0][0]

def visualize_prediction(historical_dates, historical_values, predicted_date, predicted_value):
    """可视化历史数据和预测结果"""
    plt.figure(figsize=(12, 6))
    
    # 设置中文显示 - 使用英文替代中文标签避免字体问题
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
    
    # 绘制历史数据
    plt.plot(historical_dates, historical_values, 'b-o', label='Historical Sales')
    
    # 绘制预测数据
    plt.plot(predicted_date, predicted_value, 'r*', markersize=10, label='Predicted Sales')
    
    # 设置标题和标签 - 使用英文避免字体问题
    if has_chinese_font:
        plt.title('销售量预测', fontproperties=chinese_font)
        plt.xlabel('日期', fontproperties=chinese_font)
        plt.ylabel('销售量', fontproperties=chinese_font)
    else:
        plt.title('Sales Prediction')
        plt.xlabel('Date')
        plt.ylabel('Sales Volume')
        
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('prediction_visualization.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict next day sales')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Model path')
    parser.add_argument('--data_path', type=str, default='total.csv', help='Data file path')
    parser.add_argument('--window_size', type=int, default=14, help='Input window size')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--next_date', type=str, default=None, help='Prediction date (format: YYYY/MM/DD)')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(
        model_path=args.model_path,
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=1,
        seq_len=args.window_size
    )
    
    # 准备数据
    input_data, scaler, historical_dates = prepare_data(args.data_path, args.window_size)
    
    # 获取历史数据的原始值
    last_window_original = scaler.inverse_transform(input_data.squeeze().numpy().reshape(-1, 1))
    
    # 预测下一天
    predicted_value = predict_next_day(model, input_data, scaler)
    
    # 如果未提供预测日期，则基于最后一天的日期生成下一天
    if args.next_date is None:
        last_date = historical_dates[-1]
        # 简单处理，假设格式为YYYY/MM/DD
        year, month, day = map(int, last_date.split('/'))
        # 非常简化的日期处理，仅作示例
        day += 1
        if day > 30:  # 简化处理
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
        next_date = f"{year}/{month}/{day}"
    else:
        next_date = args.next_date
    
    # 输出预测结果
    print(f"Based on data from the past {args.window_size} days:")
    for i, date in enumerate(historical_dates):
        print(f"{date}: {last_window_original[i][0]:.2f}")
    
    print(f"\nPredicted sales for {next_date}: {predicted_value:.2f}")
    
    # 可视化
    visualize_prediction(historical_dates, last_window_original.flatten(), next_date, predicted_value)
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'Date': [next_date],
        'Predicted_Sales': [predicted_value]
    })
    result_df.to_csv('next_day_prediction.csv', index=False)
    print(f"Prediction result saved to next_day_prediction.csv")

if __name__ == "__main__":
    main() 