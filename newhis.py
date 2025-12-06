import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------
# 处理 probabilistic decoding 的 200 个样本，确定 bin + avg

def process_prob_responses(query_data):
    responses = query_data["responses"]

    running_sum = sum(r["ES"] for r in responses)
    avg = running_sum / 200.0

    # 10 bins
    if running_sum == 0:
        bin_idx = 0
    elif running_sum < 25:
        bin_idx = 1
    elif running_sum < 50:
        bin_idx = 2
    elif running_sum < 75:
        bin_idx = 3
    elif running_sum < 100:
        bin_idx = 4
    elif running_sum < 125:
        bin_idx = 5
    elif running_sum < 150:
        bin_idx = 6
    elif running_sum < 175:
        bin_idx = 7
    elif running_sum < 199:
        bin_idx = 8
    else:
        bin_idx = 9

    # 修复：10 个 bins
    onehot = [0]*10
    onehot[bin_idx] = 1

    return onehot, avg



# ------------------------------------------------------
# greedy decoding：200 个样本总和 sum(ES) >0 就算有 leakage
# ------------------------------------------------------
def process_greedy_responses(query_data):
    responses = query_data["responses"]
    running_sum = sum([r["ES"] for r in responses])
    return 1 if running_sum > 0 else 0


# ------------------------------------------------------
# 处理 greedy JSONL
# ------------------------------------------------------
def processing_greedy_script(input_file_path: str):
    greedy_results = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f"Processing {os.path.basename(input_file_path)}"):
            try:
                query_data = json.loads(line)
                label = process_greedy_responses(query_data)
                greedy_results.append(label)
            except Exception as e:
                print("Error:", e)
                continue
    return greedy_results


# ======================================================
#                   主执行脚本
# ======================================================
greedy_file ="/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=0.2top_p=0.2/generations_n200.json"
prob_file   = "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=1.0top_p=1.0/generations_n200.json"

# 1. 得到 greedy leakage 0/1
greedy_results = processing_greedy_script(greedy_file)
greedy_results = np.array(greedy_results)

# 2. 得到 probabilistic bin + avg
binr = []
avgr = []

with open(prob_file, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc=f"Processing {os.path.basename(prob_file)}"):
        try:
            query_data = json.loads(line)
            onehot, avg = process_prob_responses(query_data)
            binr.append(onehot)
            avgr.append(avg)
        except Exception as e:
            print("Error processing query:", e)
            continue

binr = np.array(binr)
avgr = np.array(avgr)



# ------------------------------------------------------
# 统计每个 bin 中红、黄数量
num_bins = 10
red_counts = np.zeros(num_bins, dtype=int)
yellow_counts = np.zeros(num_bins, dtype=int)

# 每一行 binr[i] 是 one-hot，比如 [0,0,1,0,0,0,0,0]
for i in range(len(greedy_results)):
    bin_idx = np.argmax(binr[i])     # 找到该 query 落入的 probabilistic bin
    if greedy_results[i] == 1:
        red_counts[bin_idx] += 1     # 这类原来是 greedy=1（右侧）
        
    else:
        yellow_counts[bin_idx] += 1  # 这类原来是 greedy=0（左侧）

print(red_counts)
print(yellow_counts)
# ------------------------------------------------------
# 绘图：每根柱子拆成黄色（原来greedy=0）+红色（原来greedy=1）
# ------------------------------------------------------
num_bins = 10
bin_labels = ["0, completly safe","1-25", "25-50", "50-75", "75-100",
              "100-125", "125-150", "150-175", "175-199","200, completly unsafe"]

greedy_hist = np.zeros(num_bins)

# greedy=0 → 都在 bin 0
# greedy=1 → 都在 bin 7
greedy_hist[0] = np.sum(greedy_results == 0)
greedy_hist[-1] = np.sum(greedy_results == 1)



plt.figure(figsize=(8, 6))

# 和 hist_prob 完全一致的颜色
safe_color = "tab:blue"     # Original Safe Question
leaked_color = "tab:orange" # Original Leaked Question

# 为每个 bin 选择颜色
colors = []
for i in range(num_bins):
    if i == 0:
        colors.append(safe_color)      # 左侧（safe）
    elif i == num_bins - 1:
        colors.append(leaked_color)    # 右侧（leaked）
    else:
        colors.append(safe_color)      # 中间没数据，用 safe 颜色也不显示

# 绘制 greedy 柱状图
plt.bar(np.arange(num_bins), greedy_hist, color=colors)

# 添加 legend：只显示两个方块，不重复绘制柱子
plt.bar(0, 0, color=safe_color, label="Safe Question on Greedy Decoding")
plt.bar(0, 0, color=leaked_color, label="Unsafe Question on Greedy Decoding")

plt.xticks(np.arange(num_bins), bin_labels, rotation=45)
plt.xlabel("Frequency of Information Leakage Among 200 Generations")
plt.ylabel("Number of Questions in Each Bin")
plt.ylim(0, 350)
plt.legend()
plt.tight_layout()
plt.savefig("hist_greedy.png", dpi=300)
plt.close()



# ======================================================
# 2. Plot hist_prob  (stacked yellow + red)
# ======================================================

# 已经在前面计算过 red_counts, yellow_counts
plt.figure(figsize=(8, 6))

plt.bar(np.arange(num_bins), yellow_counts, label="Safe Question on Greedy Decoding")
plt.bar(np.arange(num_bins), red_counts,
        bottom=yellow_counts, label="Unsafe Question on Greedy Decoding")

plt.xticks(np.arange(num_bins), bin_labels, rotation=45)
plt.ylabel("Number of Questions in Each Bin")
plt.xlabel("Frequency of Information Leakage Among 200 Generations")
plt.ylim(0, 350)
# plt.title("(b) Probabilistic decoding (Temp=1.0, top-p=1.0)")
plt.legend()
plt.tight_layout()
plt.savefig("hist_prob.png", dpi=300)
plt.close()



# ------------------------------------------------------
# 打印原来 1 和 0 的 avg 均值
# ------------------------------------------------------
red_avg_mean = avgr[greedy_results == 1].mean() if np.sum(greedy_results==1) else 0
yellow_avg_mean = avgr[greedy_results == 0].mean() if np.sum(greedy_results==0) else 0

print("Original red (1) count:", np.sum(greedy_results == 1))
print("Original yellow (0) count:", np.sum(greedy_results == 0))
print("Mean avg of former red:", red_avg_mean)
print("Mean avg of former yellow:", yellow_avg_mean)
