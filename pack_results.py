import os
import shutil
from datetime import datetime

def pack_model_results():
    # 创建打包文件夹，使用时间戳命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = f"model_package_{timestamp}"
    os.makedirs(package_dir, exist_ok=True)
    
    # 需要复制的文件列表
    files_to_copy = [
        ("models/simplified_model.py", "代码/simplified_model.py"),
        ("models/best_model.pth", "模型/best_model.pth"),
        ("models/y_scaler.joblib", "模型/y_scaler.joblib"),
        ("simplified_model_results.png", "结果/simplified_model_results.png"),
        ("prediction_detail.png", "结果/prediction_detail.png"),
        ("total_cleaned.csv", "数据/total_cleaned.csv")
    ]
    
    # 创建子文件夹
    subdirs = ["代码", "模型", "结果", "数据"]
    for subdir in subdirs:
        os.makedirs(os.path.join(package_dir, subdir), exist_ok=True)
    
    # 复制文件
    for src, dst in files_to_copy:
        try:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(package_dir, dst))
                print(f"已复制: {src} -> {dst}")
            else:
                print(f"文件不存在: {src}")
        except Exception as e:
            print(f"复制文件时出错 {src}: {e}")
    
    # 创建说明文件
    readme_content = """模型包说明
==========

文件结构：
- 代码/: 模型源代码
- 模型/: 训练好的模型和数据缩放器
- 结果/: 预测结果可视化
- 数据/: 用于训练的数据集

使用方法：
1. 确保安装了所需的Python包（torch, pandas, numpy, scikit-learn, matplotlib）
2. 运行 simplified_model.py 进行预测
3. 结果将保存在 '结果/' 目录下

注意事项：
- best_model.pth 是训练好的模型权重
- y_scaler.joblib 用于数据缩放
- 预测结果可在 simplified_model_results.png 中查看
"""
    
    with open(os.path.join(package_dir, "说明.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n打包完成！所有文件已保存到文件夹: {package_dir}")
    return package_dir

if __name__ == "__main__":
    pack_model_results() 