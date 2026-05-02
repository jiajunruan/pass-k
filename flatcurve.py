import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# =========================
# 1️⃣ 数据
# =========================
data = {
    "NPO": [(0.745, 0.367), (0.718, 0.361), (0.032, 0.369), (0.015, 0.366)],
    "SimNPO": [(0.806, 0.368), (0.796, 0.392), (0.758, 0.411), (0.744, 0.418)],
    "GradDiff": [(0.835, 0.364), (0.110, 0.382), (0.048, 0.391), (0.040, 0.397)],
    "RMU": [(0.886, 0.377), (0.822, 0.361), (0.322, 0.372), (0.222, 0.364)],
}

# =========================
# 2️⃣ 样式配置
# =========================
colors = {
    "NPO": "#e41a1c", "SimNPO": "#a65628",
    "GradDiff": "#4daf4a", "RMU": "#7f7f7f",
}
markers = {"NPO": "v", "SimNPO": "P", "GradDiff": "^", "RMU": "h"}

start_size = 40
end_size = 180

# =========================
# 3️⃣ 开始画图
# =========================
plt.figure(figsize=(9, 7))

for method, points in data.items():
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # 绘制趋势线 (不在此处加 label)
    plt.plot(x, y, color=colors[method], linewidth=3, zorder=2)
    
    # 绘制变大的散点
    sizes = np.linspace(start_size, end_size, len(points))
    plt.scatter(x, y, s=sizes, color=colors[method], edgecolor="black", 
                linewidth=0.8, marker=markers[method], zorder=3)

# =========================
# 4️⃣ 关键修改：手动构建图例标签
# =========================
legend_handles = []
for method in data.keys():
    # 创建一个结合了线和形状的代理句柄
    handle = mlines.Line2D([], [], color=colors[method], marker=markers[method],
                           markersize=10,        # 图例中形状的大小
                           linewidth=3,          # 图例中线条的粗细
                           markeredgecolor='black', 
                           markeredgewidth=0.8,
                           label=method)
    legend_handles.append(handle)

# 使用自定义句柄生成图例
plt.legend(handles=legend_handles, fontsize=14, loc="upper right", 
           frameon=True, shadow=False, borderpad=1)

# =========================
# 5️⃣ 坐标轴 & 风格 (保持与你要求一致)
# =========================
plt.xlabel(r"$\widehat{leak@128}$-ES", fontsize=20)
plt.ylabel("Retain Score (Average ES)", fontsize=20)

plt.xlim(0, 1.0)
plt.ylim(0.2, 0.6)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("legend_fixed.pdf", format='pdf', bbox_inches='tight')
plt.show()