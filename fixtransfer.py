import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1️⃣ 数据 (保持不变)
# =========================
data = {
    "NPO": [(0.745, 0.367), (0.718, 0.361), (0.032, 0.369), (0.015, 0.366)],
    "SimNPO": [(0.806, 0.368), (0.796, 0.392), (0.758, 0.411), (0.744, 0.418)],
    "GradDiff": [(0.835, 0.364), (0.110, 0.382), (0.048, 0.391), (0.040, 0.397)],
    "RMU": [(0.886, 0.377), (0.822, 0.361), (0.322, 0.372), (0.222, 0.364)],
}

colors = {"NPO": "#e41a1c", "SimNPO": "#a65628", "GradDiff": "#4daf4a", "RMU": "#7f7f7f"}
markers = {"NPO": "v", "SimNPO": "P", "GradDiff": "^", "RMU": "h"}
start_size, end_size = 40, 180

# =========================
# 2️⃣ 画图
# =========================
plt.figure(figsize=(10, 7))

for method, points in data.items():
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # 绘制线条 (注意这里加了 label)
    # 我们给 plot 加上 marker，但在图中只画一条线，真正的 marker 由 scatter 画
    line, = plt.plot(x, y, color=colors[method], linewidth=3, label=method, zorder=2)
    
    # 绘制变大尺寸的散点
    sizes = np.linspace(start_size, end_size, len(points))
    plt.scatter(x, y, s=sizes, color=colors[method], edgecolor="black", 
                linewidth=0.8, marker=markers[method], zorder=3)

# =========================
# 3️⃣ 关键修正：图例句柄合并
# =========================
# 获取当前的图例句柄和标签
handles, labels = plt.gca().get_legend_handles_labels()

# 强制为图例中的线条添加 marker
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines

new_handles = []
for i, method in enumerate(data.keys()):
    # 创建一个同时包含线和特定 marker 的代理对象
    handle = mlines.Line2D([], [], color=colors[method], marker=markers[method],
                           markersize=10, linewidth=3, markeredgecolor='black')
    new_handles.append(handle)

plt.legend(new_handles, data.keys(), fontsize=16, loc="upper right", frameon=True)

# =========================
# 4️⃣ 细节微调
# =========================
plt.xlabel(r"$\widehat{leak@128}$-ES", fontsize=20)
plt.ylabel("Retain Score (Average ES)", fontsize=20)
plt.xlim(0, 1.0)
plt.ylim(0.2, 0.6)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("fix_legend.pdf", format='pdf', bbox_inches='tight')
plt.show()