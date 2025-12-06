import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

temperatures_and_topps = [
    # "temperature=0.0top_p=0.0",
    "temperature=0.2top_p=0.2",
    # "temperature=0.2top_p=0.8",
    "temperature=0.2top_p=1.0",
    # "temperature=0.8top_p=0.2",
    # "temperature=0.8top_p=0.8",
    # "temperature=0.8top_p=1.0",
    # "temperature=1.0top_p=0.2",
    # "temperature=1.0top_p=0.8",
    "temperature=1.0top_p=1.0"
]

subfolder_config_names = [
    "LoUK","BLUR-NPO", "Retrain", "NPO", "NPO+ENT", "SimNPO", "GradDiff", "RMU", "Original"
]

generation_indices = [1, 2, 4, 8, 16, 32, 64, 128]

metrics_to_plot = [
    # ("score_at_k_n200", "$\widehat{leak@k}$-RS", "un_score")
    ("best_es_avg_per_generation", "$\widehat{leak@k}$-LJ", "best_es"),
]

output_base_dir = "plotting2"
base_dir = "/users/2/jruan/pass-k/saves/eval"
# base_dir = "/users/2/jruan/pass-k/LLMjudgeresult/eval"

def save_detailed_heatmap(x_vals, y_vals, colors, x_tick_labels, y_tick_labels,
                          metric_title, save_path, super_title):
    x_unique = sorted(list(set(x_vals)))
    y_unique = sorted(list(set(y_vals)))
    heatmap = np.zeros((len(y_unique), len(x_unique)))

    for xi, yi, ci in zip(x_vals, y_vals, colors):
        x_idx = x_unique.index(xi)
        y_idx = y_unique.index(yi)
        heatmap[y_idx, x_idx] = ci

    plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap('RdBu_r')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = plt.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    plt.xticks(
        ticks=np.arange(len(x_unique)),
        labels=[str(n) for n in x_unique],
        rotation=45,
        fontsize=19
    )
    plt.yticks(
        ticks=np.arange(len(y_tick_labels)),
        labels=y_tick_labels,
        fontsize=16
    )

    plt.xlabel("Number of Generations $k$", fontsize=19)
    plt.ylabel("Method", fontsize=19)
    plt.title(super_title, fontsize=19)

    cbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', pad=0.02)
    cbar.set_label(metric_title, fontsize=19)
    # cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()

def process_and_plot_all_eval_data(temp_topps, config_names_list, gen_indices, metrics):
    for json_key_prefix, plot_metric_title, filename_prefix in metrics:
        for temp_top in temp_topps:
            print(f"Processing {temp_top} - {plot_metric_title}")

            collected_data = []
            x_coords = []
            y_coords = []

            for config_idx, config_name in enumerate(config_names_list):
                json_file = os.path.join(base_dir, config_name, "forget", temp_top, "rougeL_summary.json")
                print("json_path:", json_file)
                # json_file = os.path.join(full_path, f"{json_key_prefix}.json")

                if not os.path.exists(json_file):
                    print(f"Missing {json_file}, skipping.")
                    continue

                with open(json_file, 'r') as f:
                    summary_data = json.load(f)

                for n in gen_indices:
                    val = summary_data.get(str(n), 0.0)
                    collected_data.append(val)
                    x_coords.append(n)
                    y_coords.append(config_idx)

            if not collected_data:
                continue

            filename = f"{filename_prefix}_{temp_top}.pdf"
            detailed_save_path = os.path.join(output_base_dir, filename)
            os.makedirs(output_base_dir, exist_ok=True)

            # === Extract temperature and top_p for supertitle ===
            temp_val = temp_top[temp_top.find("temperature=") + len("temperature="): temp_top.find("top_p=")]
            top_p_val = temp_top[temp_top.find("top_p=") + len("top_p="):]

            save_detailed_heatmap(
                x_coords, y_coords, collected_data,
                gen_indices, config_names_list,
                metric_title=plot_metric_title,
                save_path=detailed_save_path,
                super_title=f"(temperature, top-p) = ({temp_val}, {top_p_val})"
            )


if __name__ == "__main__":
    process_and_plot_all_eval_data(
        temp_topps=temperatures_and_topps,
        config_names_list=subfolder_config_names,
        gen_indices=generation_indices,
        metrics=metrics_to_plot
    )