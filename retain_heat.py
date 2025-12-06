import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

temperatures_and_topps = [
    "temperature=0.0top_p=0.0",
    "temperature=0.2top_p=0.2",
    "temperature=0.2top_p=0.8",
    "temperature=0.2top_p=1.0",
    "temperature=0.8top_p=0.2",
    "temperature=0.8top_p=0.8",
    "temperature=0.8top_p=1.0",
    "temperature=1.0top_p=0.2",
    "temperature=1.0top_p=0.8",
    "temperature=1.0top_p=1.0"
]

temp_top_labels = [
    "(0.0, 0.0)",
    "(0.2, 0.2)",
    "(0.2, 0.8)",
    "(0.2, 1.0)",
    "(0.8, 0.2)",
    "(0.8, 0.8)",
    "(0.8, 1.0)",
    "(1.0, 0.2)",
    "(1.0, 0.8)",
    "(1.0, 1.0)"
]

subfolder_config_names = [
    "LoUK", "BLUR-NPO", "Retrain", "NPO", "NPO+ENT", "SimNPO", "GradDiff", "RMU", "Original"
]

gen_index = 200
metric_key = "mean_mean_rougeL_recall"
base_dir = "/users/2/jruan/pass-k/saves/eval"
output_plot = "results_plots/retain_heatmap_gen_scores_rougeL.pdf"

def save_retain_heatmap(data_matrix, x_labels, y_labels, save_path, title, metric_label):
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = plt.imshow(data_matrix, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, fontsize=19)
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, fontsize=16)

    # Move x-axis label to top
    plt.xlabel("(temperature, top-p)", fontsize=19, labelpad=20)
    # plt.gca().xaxis.set_label_position('top')
    # plt.gca().xaxis.tick_top()

    plt.ylabel("Method", fontsize=19)

    cbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', pad=0.02)
    cbar.set_label(metric_label, fontsize=19)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()


def collect_rougel_scores_from_gen_scores():
    data_matrix = np.zeros((len(subfolder_config_names), len(temperatures_and_topps)))

    for j, temp_top in enumerate(temperatures_and_topps):
        for i, method in enumerate(subfolder_config_names):
            json_path = os.path.join(base_dir, method, "retain", temp_top, "avg_summary.json")
            if not os.path.exists(json_path):
                print(f"⚠️ Missing: {json_path}")
                data_matrix[i, j] = np.nan
                continue

            with open(json_path, "r") as f:
                scores = json.load(f)
                # 读取 avg_es_avg_per_generation 下 generations_n128 的值
#                 gen_data = scores.get("avg_es_avg_per_generation", {})
#                 value = gen_data.get("generations_n128", np.nan)
                value = scores.get("200")
                data_matrix[i, j] = value

    return data_matrix


if __name__ == "__main__":
    retain_matrix = collect_rougel_scores_from_gen_scores()
    save_retain_heatmap(
        data_matrix=retain_matrix,
        x_labels=temp_top_labels,
        y_labels=subfolder_config_names,
        save_path=output_plot,
        title=f"Mean Entailment Score @ Generation {gen_index}",
        metric_label="Average ES"
    )