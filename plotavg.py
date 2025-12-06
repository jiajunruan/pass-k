import json
import matplotlib.pyplot as plt

# ======= 在这里填你的两个 JSON 文件路径 =======
json_file_1 = "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=0.0top_p=0.0/generations_n200.json"
json_file_2 = "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=1.0top_p=1.0/generations_n200.json"
# ===========================================

# 读取第一个 JSON
with open(json_file_1, "r", encoding="utf-8") as f:
    json1 = json.load(f)

# 读取第二个 JSON
with open(json_file_2, "r", encoding="utf-8") as f:
    json2 = json.load(f)

# 将 key 按数字排序
x1 = sorted(map(int, json1.keys()))
y1 = [json1[str(k)] for k in x1]

x2 = sorted(map(int, json2.keys()))
y2 = [json2[str(k)] for k in x2]

# 画图
plt.figure(figsize=(10, 5))
plt.plot(x1, y1, label="Greedy Decoding")
plt.plot(x2, y2, label="Probabilistic Decoding")

plt.xlabel("Sample n")
plt.ylabel("Leakage")
plt.title("Average Leakage Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()

# === 保存图片 ===
plt.savefig("avg.png", dpi=300)


