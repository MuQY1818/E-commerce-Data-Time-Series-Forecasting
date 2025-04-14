# 深度学习模型NaN损失问题分析与解决方案

## 问题分析

训练过程中出现NaN损失值通常由以下几个原因导致：

1. **梯度爆炸**: 当梯度过大时，权重更新会变得极端，导致模型参数溢出。
2. **数据预处理问题**: 特征或目标值中存在异常值、零值或无穷大值。
3. **学习率过高**: 较大的学习率导致优化过程不稳定。
4. **数值溢出**: 在计算过程中（如log、除法）可能出现除零或取负数的对数。
5. **数据标准化问题**: 如果标准差接近0，标准化可能产生极大值。
6. **网络结构复杂度**: 模型过于复杂，导致数值不稳定。

## 针对当前项目的解决方案

基于对`models/train.py`和`models/time_series_model.py`的分析，建议采取以下措施：

### 1. 数据预处理改进

```python
# 改进数据处理方式 - 使用更保守的异常值处理
def handle_outliers(df, column, method='winsorize', threshold=3):
    sales_values = df[column].copy()
    
    if method == 'winsorize':
        # 百分位数截断，更保守的方式
        lower_percentile = np.percentile(sales_values, 1)  # 使用1%截断而非5%
        upper_percentile = np.percentile(sales_values, 99)  # 使用99%截断而非95%
        
        # 截断异常值
        sales_values[sales_values < lower_percentile] = lower_percentile
        sales_values[sales_values > upper_percentile] = upper_percentile
    
    return sales_values
```

### 2. 缩放方法调整

```python
# 使用更稳健的数据缩放方法
from sklearn.preprocessing import RobustScaler

# 确保输出的缩放数据没有极端值
def safe_scaling(data, scaler):
    scaled_data = scaler.fit_transform(data)
    # 额外保护：限制缩放后的极端值
    scaled_data = np.clip(scaled_data, -10, 10)
    return scaled_data, scaler
```

### 3. 模型稳定性改进

```python
# 在HybridTemporalModel的forward方法中添加数值稳定性检查
def forward(self, x):
    batch_size, seq_len, _ = x.shape
    
    # 添加输入数据检查
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("警告：输入数据包含NaN或Inf值")
        # 替换NaN和Inf值
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # LSTM部分
    lstm_out, _ = self.lstm(x)
    # 添加层间检查
    lstm_out = torch.nan_to_num(lstm_out, nan=0.0)
    
    # [其他层保持不变...]
    
    return out
```

### 4. 训练过程改进

```python
# 训练函数添加梯度检查和学习率调整
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    # ...现有代码...
    
    # 添加学习率预热
    warmup_epochs = 5
    initial_lr = 0.0001  # 开始使用较小的学习率
    target_lr = 0.001
    
    for epoch in range(num_epochs):
        # 学习率预热
        if epoch < warmup_epochs:
            current_lr = initial_lr + (target_lr - initial_lr) * epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # ...训练循环...
            
        # 增强梯度裁剪保护
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 更严格的梯度裁剪
        except Exception as e:
            print(f"梯度裁剪出错: {e}")
            # 如果发生异常，重置梯度并继续
            optimizer.zero_grad()
            continue
```

### 5. 简化模型结构

考虑暂时移除复杂组件，从简单模型开始逐步增加复杂性：

```python
# 简化的模型版本
class SimplifiedTemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len):
        super(SimplifiedTemporalModel, self).__init__()
        
        # 仅保留LSTM层和输出层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 基本的LSTM预测
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
```

### 6. 检查特征工程

确保生成的特征不会导致数值不稳定：

```python
# 安全的特征创建
def create_safe_features(df):
    # 添加特征前的数据检查
    if df.isnull().any().any():
        print("警告：输入数据包含NaN值")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 防止除零错误的百分比变化计算
    df['pct_change'] = df['成交商品件数'].pct_change()
    df['pct_change'] = df['pct_change'].replace([np.inf, -np.inf], 0)
    
    # 其他特征计算...
    
    return df
```

## 调试建议

1. **增量调试**：先删除所有复杂组件，确保基本模型能正常工作，再逐步添加组件。

2. **检查权重初始化**：尝试不同的初始化方法：
   ```python
   def weight_init(m):
       if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
           nn.init.kaiming_normal_(m.weight)
           if m.bias is not None:
               nn.init.zeros_(m.bias)
   
   model.apply(weight_init)
   ```

3. **记录中间值**：在每个组件处理后打印数据统计信息：
   ```python
   def print_stats(tensor, name):
       print(f"{name} stats: min={tensor.min().item()}, max={tensor.max().item()}, "
             f"mean={tensor.mean().item()}, std={tensor.std().item()}")
   ```

4. **使用混合精度训练**：如果是因为数值精度问题，可以尝试混合精度训练。

5. **逐层检查**：如果怀疑某个具体层导致问题，可以替换为更简单的实现进行测试。

## 下一步建议

1. 首先实现简化模型和数据预处理改进
2. 添加数值稳定性检查和异常处理
3. 使用更保守的训练设置（较小学习率、较强梯度裁剪）
4. 成功训练基本模型后，逐步恢复复杂组件 