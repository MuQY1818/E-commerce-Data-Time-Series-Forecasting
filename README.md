# 销量预测集成模型

这个项目实现了一个高级销量预测系统，通过集成多个机器学习模型来提供更准确的预测结果。

## 模型架构

该系统集成了以下五个基础模型：

1. **Prophet模型**
   - 处理时间序列的趋势和季节性
   - 自动处理节假日效应
   - 可以捕捉长期和短期的周期性变化

2. **XGBoost模型**
   - 处理非线性特征关系
   - 自动特征选择
   - 对异常值具有鲁棒性

3. **LightGBM模型**
   - 高效处理大规模数据
   - 自动处理类别特征
   - 较低的内存使用

4. **随机森林模型**
   - 提供稳定的基准预测
   - 降低过拟合风险
   - 特征重要性分析

5. **简单神经网络(SimpleNN)**
   - 三层全连接网络
   - 处理复杂的非线性关系
   - Dropout层防止过拟合

## 特征工程

系统包含丰富的特征工程：

1. **时间特征**
   - 星期几、是否周末
   - 月份、日期
   - 季节性编码（正弦/余弦变换）

2. **滞后特征**
   - 前15天的历史数据
   - 多个时间窗口的移动平均
   - 趋势指标

3. **统计特征**
   - 7/14/30天滑动窗口统计
   - 差分特征
   - 比率特征

## 模型集成方法

使用神经网络进行模型集成：

- 双层集成架构
- 动态权重分配
- 验证集指导的早停机制
- HuberLoss损失函数

## 环境要求

```bash
# 主要依赖
pandas
numpy
prophet
scikit-learn
xgboost
lightgbm
torch
matplotlib
```

## 使用方法

1. **数据准备**
```python
# 数据格式要求：CSV文件包含'日期'和'成交商品件数'列
df = pd.read_csv('total_cleaned.csv')
```

2. **运行预测**
```python
# 直接运行ensemble_model.py
python models/ensemble_model.py
```

3. **输出结果**
- 预测结果保存在 `models/ensemble_results.csv`
- 可视化结果保存在 `models/ensemble_predictions.png`
- 模型文件保存在 `models/` 目录下

## 模型评估

系统提供多个评估指标：
- 均方误差 (MSE)
- 均方根误差 (RMSE)
- 平均绝对误差 (MAE)
- 平均绝对百分比误差 (MAPE)
- R² 分数

## 目录结构

```
├── models/
│   ├── ensemble_model.py     # 主要模型代码
│   ├── ensemble_xgb.joblib   # XGBoost模型
│   ├── ensemble_lgb.joblib   # LightGBM模型
│   ├── ensemble_rf.joblib    # 随机森林模型
│   ├── ensemble_simple_nn.pth # 简单神经网络模型
│   ├── ensemble_nn.pth       # 集成神经网络
│   └── ensemble_results.csv  # 预测结果
├── total_cleaned.csv         # 输入数据
└── README.md                 # 项目说明
```

## 注意事项

1. 确保输入数据的质量和格式正确
2. 模型训练可能需要较长时间
3. 建议使用GPU加速训练过程
4. 定期更新模型以保持预测准确性

## 未来改进

1. 添加更多的基础模型
2. 优化特征工程过程
3. 实现自动化的超参数调优
4. 添加在线学习功能
5. 改进模型集成策略 