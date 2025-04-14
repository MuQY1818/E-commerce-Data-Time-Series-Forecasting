import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def prepare_prophet_data(df):
    """准备Prophet所需的数据格式"""
    # Prophet要求数据列名为'ds'和'y'
    prophet_df = df.copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['日期'])
    prophet_df['y'] = prophet_df['成交商品件数']
    
    # 只保留必要的列
    prophet_df = prophet_df[['ds', 'y']]
    
    return prophet_df

def add_custom_features(df):
    """添加自定义特征"""
    # 添加中国特殊节日
    holidays = pd.DataFrame({
        'holiday': 'chinese_holiday',
        'ds': pd.to_datetime([
            '2023-01-01',  # 元旦
            '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27',  # 春节
            '2023-04-05',  # 清明节
            '2023-05-01', '2023-05-02', '2023-05-03',  # 劳动节
            '2023-06-22', '2023-06-23', '2023-06-24',  # 端午节
            '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05',  # 国庆节
            '2023-12-31'  # 跨年
        ]),
        'lower_window': -1,  # 节日前一天
        'upper_window': 1,   # 节日后一天
    })
    
    return holidays

def train_prophet_model(train_df, holidays=None):
    """训练Prophet模型"""
    # 初始化模型
    model = Prophet(
        yearly_seasonality=20,    # 增加年度季节性的复杂度
        weekly_seasonality=10,    # 增加周度季节性的复杂度
        daily_seasonality=False,  # 关闭日度季节性，减少过拟合
        holidays=holidays,        # 节假日
        changepoint_prior_scale=0.01,  # 降低趋势变化点的灵活度，使趋势更平滑
        seasonality_prior_scale=15,    # 增加季节性的强度
        holidays_prior_scale=20,       # 增加节假日效应的强度
        changepoint_range=0.9,         # 允许在90%的数据范围内检测变化点
        seasonality_mode='additive'    # 改用加法季节性模式
    )
    
    # 添加月度季节性
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=8,        # 增加傅里叶阶数，捕捉更复杂的月度模式
        prior_scale=15
    )
    
    # 添加季度季节性
    model.add_seasonality(
        name='quarterly',
        period=91.25,           # 365.25/4
        fourier_order=6,
        prior_scale=15
    )
    
    print("开始训练Prophet模型...")
    model.fit(train_df)
    return model

def make_predictions(model, periods=30):
    """生成预测"""
    # 创建未来日期的数据框
    future = model.make_future_dataframe(periods=periods)
    
    # 进行预测
    forecast = model.predict(future)
    return forecast

def evaluate_prophet(y_true, y_pred):
    """评估Prophet模型的性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\nProphet模型评估指标：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"R² 分数: {r2:.4f}")
    
    return mse, rmse, mae, r2

def plot_prophet_results(model, forecast, train_df, test_df=None):
    """绘制Prophet预测结果"""
    plt.figure(figsize=(15, 10))
    
    # 设置全局字体
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'
    
    # 绘制训练数据
    plt.plot(train_df['ds'], train_df['y'], 
            label='训练数据', color='#1f77b4', alpha=0.7, 
            marker='o', markersize=4, linewidth=1.5)
    
    # 如果有测试数据，绘制测试数据
    if test_df is not None:
        plt.plot(test_df['ds'], test_df['y'], 
                label='测试数据', color='#2ca02c', alpha=0.7,
                marker='s', markersize=4, linewidth=1.5)
    
    # 绘制预测结果
    plt.plot(forecast['ds'], forecast['yhat'], 
            label='预测值', color='#ff7f0e', 
            linestyle='--', linewidth=2)
    
    # 绘制置信区间
    plt.fill_between(forecast['ds'], 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    color='#ff7f0e', 
                    alpha=0.2, 
                    label='95% 置信区间')
    
    plt.title('Prophet模型预测结果', fontsize=14, pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('销量', fontsize=12)
    plt.legend(fontsize=10, loc='best', frameon=True, 
              facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 调整x轴日期显示
    plt.gcf().autofmt_xdate()
    
    # 确保models目录存在
    os.makedirs('models', exist_ok=True)
    
    # 保存图片
    plt.savefig('models/prophet_predictions.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    print("\n预测结果图表已保存到 models/prophet_predictions.png")

def analyze_components(model, forecast):
    """分析并绘制Prophet模型的各个组件"""
    # 设置组件图的中文标签
    components = ['趋势', '年度', '季度', '月度', '周度', '节假日']
    
    # 绘制趋势、季节性等组件
    fig = model.plot_components(forecast)
    
    # 修改子图的标题
    for i, ax in enumerate(fig.axes):
        if i < len(components):
            ax.set_title(components[i], fontsize=12)
    
    # 保存组件分析图
    plt.savefig('models/prophet_components.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    print("\n模型组件分析图已保存到 models/prophet_components.png")

def main():
    # 加载数据
    df = pd.read_csv('total_cleaned.csv')
    
    # 准备Prophet数据
    prophet_df = prepare_prophet_data(df)
    
    # 准备节假日数据
    holidays = add_custom_features(prophet_df)
    
    # 划分训练集和测试集
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:train_size]
    test_df = prophet_df[train_size:]
    
    print("数据集大小:")
    print(f"训练集: {len(train_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    
    # 训练模型
    print("\n开始训练Prophet模型...")
    model = train_prophet_model(train_df, holidays)
    
    # 生成预测
    forecast = make_predictions(model, periods=len(test_df))
    
    # 评估模型
    y_true = test_df['y'].values
    y_pred = forecast.tail(len(test_df))['yhat'].values
    evaluate_prophet(y_true, y_pred)
    
    # 绘制预测结果
    plot_prophet_results(model, forecast, train_df, test_df)
    
    # 分析模型组件
    analyze_components(model, forecast)
    
    # 保存模型和预测结果
    os.makedirs('models', exist_ok=True)
    
    # 保存预测结果
    forecast.to_csv('models/prophet_forecast.csv', index=False)
    
    # 保存模型参数
    with open('models/prophet_params.txt', 'w') as f:
        f.write(str(model.params))
        
    print("\nProphet模型预测结果已保存到 models/prophet_forecast.csv")
    print("模型参数已保存到 models/prophet_params.txt")

if __name__ == "__main__":
    main() 