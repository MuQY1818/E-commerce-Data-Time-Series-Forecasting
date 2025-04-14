# 商品销售量时序预测模型 | Time Series Sales Forecasting Model

[English](#english) | [中文](#chinese)

<a name="chinese"></a>
## 中文版

这是一个基于深度学习的时序预测项目，用于预测商品销售量。模型采用了复杂的混合架构，结合LSTM、注意力机制、时间卷积网络(TCN)和Transformer，提供高精度的销售量预测。

### 模型架构

该项目实现了一个名为`HybridTemporalModel`的混合时序预测模型，它由以下组件构成：

1. **LSTM层**：捕获时间序列的长期依赖关系
2. **时间注意力机制**：动态关注序列中的重要时间步
3. **时间卷积网络(TCN)**：通过扩张卷积捕获多尺度时间模式
4. **Transformer编码器**：利用自注意力机制进一步增强对序列关系的建模能力
5. **全连接输出层**：将所有特征整合并输出最终预测

### 项目结构

```
.
├── total.csv                 # 销售量数据
├── models/
│   ├── time_series_model.py  # 模型定义文件
│   └── train.py              # 训练脚本
├── best_model.pth            # 训练过程中保存的最佳模型
├── prediction_results.csv    # 预测结果
├── time_series_results.png   # 结果可视化图像
└── README.md                 # 项目说明文档
```

### 环境要求

本项目需要以下Python库：

- PyTorch >= 1.7.0
- NumPy
- Pandas
- Matplotlib
- scikit-learn

可以通过以下命令安装依赖：

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

### 使用方法

#### 1. 数据准备

确保您的CSV文件格式为：第一列是日期，第二列是销售量数据。

#### 2. 训练模型

```bash
cd models
python train.py
```

### 3. 训练参数说明

训练脚本中的主要参数：

- `seq_length`: 用于预测的历史时间窗口长度（默认14天）
- `batch_size`: 批次大小（默认32）
- `hidden_size`: 隐藏层维度（默认64）
- `num_layers`: LSTM层数（默认2）
- `num_epochs`: 训练轮数（默认50）
- `patience`: 早停机制的耐心值（默认15）

### 4. 评估指标

模型使用以下指标评估预测性能：

- 均方误差(MSE)
- 均方根误差(RMSE)
- 决定系数(R²)

### 结果可视化

训练后，脚本将生成`time_series_results.png`文件，其中包含：

1. 训练和验证损失曲线
2. 实际值与预测值的对比图

### 先进特性

1. **混合架构设计**: 结合多种先进的时序建模技术
2. **注意力机制**: 动态关注重要的时间步
3. **梯度裁剪**: 防止训练过程中的梯度爆炸
4. **学习率调度**: 自动调整学习率以获得更好的收敛性
5. **早停机制**: 避免过拟合
6. **完整的评估指标**: 提供多种评估模型性能的指标

### 模型改进方向

1. 增加更多特征输入，如产品价格、促销活动、季节因素等
2. 实现贝叶斯优化来自动调整超参数
3. 添加集成学习方法，如模型组合或堆叠
4. 考虑多步预测而不仅是单步预测

---

<a name="english"></a>
## English Version

This is a deep learning-based time series forecasting project for predicting product sales volume. The model employs a sophisticated hybrid architecture combining LSTM, attention mechanisms, Temporal Convolutional Networks (TCN), and Transformers to provide high-accuracy sales predictions.

### Model Architecture

The project implements a hybrid temporal model called `HybridTemporalModel`, which consists of the following components:

1. **LSTM Layer**: Captures long-term dependencies in time series
2. **Temporal Attention Mechanism**: Dynamically focuses on important time steps
3. **Temporal Convolutional Network (TCN)**: Captures multi-scale temporal patterns using dilated convolutions
4. **Transformer Encoder**: Further enhances the modeling of sequence relationships using self-attention mechanisms
5. **Fully Connected Output Layer**: Integrates all features and outputs the final prediction

### Project Structure

```
.
├── total.csv                 # Sales data
├── models/
│   ├── time_series_model.py  # Model definition
│   ├── train.py              # Training script
│   └── predict.py            # Prediction script
├── best_model.pth            # Best model saved during training
├── prediction_results.csv    # Prediction results
├── time_series_results.png   # Results visualization
└── README.md                 # Project documentation
```

### Requirements

This project requires the following Python libraries:

- PyTorch >= 1.7.0
- NumPy
- Pandas
- Matplotlib
- scikit-learn

You can install the dependencies using:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

### Usage

#### 1. Data Preparation

Ensure your CSV file is formatted with dates in the first column and sales volume data in the second column.

#### 2. Training the Model

```bash
cd models
python train.py
```

#### 3. Predicting Future Sales

```bash
cd models
python predict.py --data_path ../total.csv
```

#### 4. Training Parameters

Key parameters in the training script:

- `seq_length`: Historical time window length for prediction (default 14 days)
- `batch_size`: Batch size (default 32)
- `hidden_size`: Hidden dimension (default 64)
- `num_layers`: Number of LSTM layers (default 2)
- `num_epochs`: Number of training epochs (default 50)
- `patience`: Patience value for early stopping (default 15)

#### 5. Evaluation Metrics

The model uses the following metrics to evaluate prediction performance:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### Results Visualization

After training, the script will generate a `time_series_results.png` file containing:

1. Training and validation loss curves
2. Comparison of actual values and predicted values

### Advanced Features

1. **Hybrid Architecture Design**: Combines multiple advanced time series modeling techniques
2. **Attention Mechanism**: Dynamically focuses on important time steps
3. **Gradient Clipping**: Prevents gradient explosion during training
4. **Learning Rate Scheduling**: Automatically adjusts learning rate for better convergence
5. **Early Stopping**: Prevents overfitting
6. **Comprehensive Evaluation Metrics**: Provides multiple metrics for evaluating model performance

### Future Improvements

1. Add more feature inputs such as product prices, promotional activities, seasonal factors, etc.
2. Implement Bayesian optimization for automatic hyperparameter tuning
3. Add ensemble learning methods such as model combination or stacking
4. Consider multi-step prediction rather than just single-step prediction 