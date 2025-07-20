import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 配置参数
base_dir = "/root/autodl-tmp/open-unlearning/saves/eval"  # 修改为你的大文件夹路径
methods = ["Original", "retain", "GradDiff", "NPO+Entropy", "SimNPO"]
temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
metrics = ["forget_Q_A_ROUGE", "forget_Q_A_gibberish", "forget_quality", "retain_Truth_Ratio"]
colors = ['b', 'g', 'r', 'c', 'm']  # 每种方法的颜色
line_styles = ['-', '--', ':', '-.', '--']  # 线型
markers = ['o', 's', '^', 'D', 'x']  # 标记样式

# 准备存储数据的字典
data = {method: {metric: [] for metric in metrics} for method in methods}

# 读取所有JSON文件
for method in methods:
    for temp in temperatures:
        filename = f"TOFU_temperature{temp}SUMMARY.json"
        filepath = os.path.join(base_dir, method, filename)
        
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
                
            for metric in metrics:
                if metric in json_data:
                    data[method][metric].append(json_data[metric])
                else:
                    data[method][metric].append(np.nan)
        except FileNotFoundError:
            print(f"警告: 文件 {filepath} 未找到")
            for metric in metrics:
                data[method][metric].append(np.nan)

# 创建四个子图，每个指标一个
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison Across Methods and Temperatures', fontsize=16)

# 绘制每个指标的图表
for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    for j, method in enumerate(methods):
        ax.plot(temperatures, data[method][metric], 
                label=method, 
                color=colors[j],
                linestyle=line_styles[j],
                marker=markers[j],
                linewidth=2,
                markersize=8)
    
    ax.set_xlabel('Temperature', fontsize=10)
    ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
    ax.set_title(metric.replace('_', ' '), fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(temperatures)
    
    # 对forget_quality使用对数坐标（因为值非常小）
    if metric == "forget_quality":
        ax.set_yscale('log')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图表
output_path = os.path.join(base_dir, "all_methods_metrics_comparison.png")
plt.savefig(output_path, dpi=300)
print(f"图表已保存至: {output_path}")

# 显示图表
plt.show()