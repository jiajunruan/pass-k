import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def process_prob_responses(query_data):
    responses = query_data["responses"]

    running_sum = sum(r["ES"] for r in responses)


    return running_sum


def process_greedy_responses(query_data):
    responses = query_data["responses"]
    running_sum = sum([r["ES"] for r in responses])
    return 1 if running_sum > 0 else 0


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



greedy_file ="/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=0.0top_p=0.0/generations_n200.json"
prob_file   = "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=1.0top_p=1.0/generations_n200.json"

greedy_results = processing_greedy_script(greedy_file)
greedy_results = np.array(greedy_results)

cate = []

with open(prob_file, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc=f"Processing {os.path.basename(prob_file)}"):
        try:
            query_data = json.loads(line)
            category = process_prob_responses(query_data)
            cate.append(category)
            
        except Exception as e:
            print("Error processing query:", e)
            continue

cate = np.array(cate)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

safe_cate = cate[greedy_results == 0]
unsafe_cate = cate[greedy_results == 1]

bins = np.arange(0, 202)
safe_hist, _ = np.histogram(safe_cate, bins=bins)
unsafe_hist, _ = np.histogram(unsafe_cate, bins=bins)


offset = 250

# ==========================================
#   0   → green (#008800)
#   100 → yellow (#ffff00)
#   200 → red (#ff0000)
# ==========================================
cmap = mcolors.LinearSegmentedColormap.from_list(
    "green_yellow_red",
    ["#008800", "#ffff00", "#ff0000"]  # green → yellow → red
)

norm = plt.Normalize(0, 200)

left_colors = cmap(norm(bins[:-1]))
right_colors = cmap(norm(bins[:-1]))


fig, ax = plt.subplots(figsize=(12, 6))


ax.bar(
    bins[:-1],
    safe_hist,
    width=1.0,
    color=left_colors,
    edgecolor="none",
    label="safe"
)


ax.bar(
    bins[:-1] + offset,
    unsafe_hist,
    width=1.0,
    color=right_colors,
    edgecolor="none",
    label="unsafe"
)


ax.set_title("Leakage Frequency Distribution")
ax.set_xlabel("Leaking Frequency (Numbers of Leaking Responses)")
ax.set_ylabel("Number of Questions")

ax.set_xticks(
    [
        0, 50, 100, 150, 200,      
        offset + 0,
        offset + 50,
        offset + 100,
        offset + 150,
        offset + 200              
    ]
)

ax.set_xticklabels(
    [
        "0", "50", "100", "150", "200",
        "0", "50", "100", "150", "200"
    ]
)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([]) 
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Leaking responses (0=green, 100=yellow, 200=red)")

plt.tight_layout()
plt.show()



plt.savefig("his.png", dpi=300)












