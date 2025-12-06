import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# === Configuration ===

temperatures_and_topps = [
    "temperature=0.0top_p=0.0",
    "temperature=0.2top_p=0.2",
    "temperature=0.2top_p=1.0",
    "temperature=0.8top_p=0.2",
    "temperature=0.8top_p=1.0",
    "temperature=1.0top_p=1.0",
    "temperature=0.2top_p=0.8",
    "temperature=0.8top_p=0.8",
    "temperature=1.0top_p=0.8",
    "temperature=1.0top_p=0.2",
]

subfolder_config_names = [
    "BLUR-NPO", "Retrain", "NPO", "NPO+ENT", "SimNPO", "GradDiff", "RMU", "Original"
]

generation_indices = [1, 2, 4, 8, 16, 32, 64, 128]

metrics_to_plot = [
    # ("generations_n", "ROUGE-L Recall", "rouge_l"),
    # ("best_cs_avg_per_generation", "Best Cosine Similarity", "best_cs"),
    # ("avg_cs_avg_per_generation", "Average Cosine Similarity", "avg_cs"),
    ("best_es_avg_per_generation", r"$\widehat{leak@k-ES}$", "best_es"),
    # ("avg_es_avg_per_generation", "Average Entailment Score", "avg_es")
]

output_base_dir = "forget_heatmap"
base_dir = "/users/2/jruan/pass-k/saves/eval"

# === Plotting Function ===

# def save_detailed_heatmap(x_vals, y_vals, colors, x_tick_labels, y_tick_labels,
#                           metric_title, save_path, super_title):
#     x_unique = sorted(list(set(x_vals)))
#     y_unique = sorted(list(set(y_vals)))
#     heatmap = np.zeros((len(y_unique), len(x_unique)))

#     for xi, yi, ci in zip(x_vals, y_vals, colors):
#         x_idx = x_unique.index(xi)
#         y_idx = y_unique.index(yi)
#         heatmap[y_idx, x_idx] = ci

#     plt.figure(figsize=(8, 5))
#     cmap = plt.get_cmap('RdBu_r')
#     norm = mcolors.Normalize(vmin=0, vmax=1)

#     im = plt.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', origin='lower')

#     plt.xticks(
#         ticks=np.arange(len(x_unique)),
#         labels=[str(n) for n in x_unique],
#         rotation=45,
#         fontsize=19
#     )
#     plt.yticks(
#         ticks=np.arange(len(y_tick_labels)),
#         labels=y_tick_labels,
#         fontsize=16
#     )

#     plt.xlabel("Number of Generations $k$", fontsize=19)
#     plt.ylabel("Method", fontsize=19)
#     plt.title(super_title, fontsize=19)

#     cbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', pad=0.02)
#     cbar.set_label(metric_title, fontsize=19)
#     cbar.ax.tick_params(labelsize=16)

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300, format='pdf')
#     plt.close()

def save_detailed_heatmap(x_vals, y_vals, colors, x_tick_labels, y_tick_labels,
                          metric_title, save_path, super_title,
                          show_x=True, show_y=True, show_cbar=True):
    x_unique = sorted(list(set(x_vals)))
    y_unique = sorted(list(set(y_vals)))
    heatmap = np.zeros((len(y_unique), len(x_unique)))

    for xi, yi, ci in zip(x_vals, y_vals, colors):
        x_idx = x_unique.index(xi)
        y_idx = y_unique.index(yi)
        heatmap[y_idx, x_idx] = ci
    print("super_title:", super_title)
    if "0.8)" in super_title: 
        
        plt.figure(figsize=(6, 5))
    else:
        plt.figure(figsize=(8, 5))
    cmap = plt.get_cmap('RdBu_r')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = plt.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    if show_x:
        plt.xticks(
            ticks=np.arange(len(x_unique)),
            labels=[str(n) for n in x_unique],
            rotation=45,
            fontsize=19
        )
        plt.xlabel("Number of Generations $k$", fontsize=19)
    else:
        plt.xticks([])
        plt.xlabel("")

    if show_y:
        plt.yticks(
            ticks=np.arange(len(y_tick_labels)),
            labels=y_tick_labels,
            fontsize=16
        )
        plt.ylabel("Method", fontsize=19)
    else:
        plt.yticks([])
        plt.ylabel("")

    plt.title(super_title, fontsize=19)

    if show_cbar:
        cbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', pad=0.02)
        cbar.set_label(metric_title, fontsize=19)
        cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()


# === Main Loop ===

def process_and_plot_all_eval_data(temp_topps, config_names_list, gen_indices, metrics):
    for json_key_prefix, plot_metric_title, filename_prefix in metrics:
        for temp_top in temp_topps:
            print(f"Processing {temp_top} - {plot_metric_title}")

            collected_data = []
            x_coords = []
            y_coords = []

            for config_idx, config_name in enumerate(config_names_list):
                full_path = os.path.join(base_dir, f"{config_name}","forget")
                summary_file_path = os.path.join(full_path, temp_top, "rougeL_summary.json")
                print(f"  Processing config: {config_name} at {full_path}")

                if not os.path.exists(summary_file_path):
                    print(f"⚠️ Missing {summary_file_path}, skipping.")
                    continue

                with open(summary_file_path, 'r') as f:
                    summary_data = json.load(f)

                if json_key_prefix == "generations_n":
                    for n in gen_indices:
                        collected_data.append(summary_data.get(f"generations_n{n}", 0.0))
                else:
                    nested_data = summary_data.get(json_key_prefix, {})
                    for n in gen_indices:
                        collected_data.append(nested_data.get(f"generations_n{n}", 0.0))

                x_coords.extend(gen_indices)
                y_coords.extend([config_idx] * len(gen_indices))

            if not collected_data:
                continue

            # === Construct output file path ===
            filename = f"{filename_prefix}_{temp_top}.pdf"
            detailed_save_path = os.path.join(output_base_dir, filename)
            os.makedirs(output_base_dir, exist_ok=True)

            # === Extract temperature and top_p for supertitle ===
            temp_val = temp_top.split("temperature=")[1].split("top_p=")[0]
            top_p_val = temp_top.split("top_p=")[1]

            # 控制显示
            show_x = True
            show_y = True
            show_cbar = True
            if (temp_val, top_p_val) in [("0.2", "0.2"), ("0.8", "0.2"), ("1.0", "0.2")]:
                show_x = True
                show_cbar = False
            elif (temp_val, top_p_val) in [("0.2", "1.0"), ("0.8", "1.0"), ("1.0", "1.0")]:
                show_y = False
            elif (temp_val, top_p_val) in [("0.2", "0.8"), ("0.8", "0.8"), ("1.0", "0.8")]:
                show_x = True
                show_y = False
                show_cbar = False

            save_detailed_heatmap(
                x_coords, y_coords, collected_data,
                gen_indices, config_names_list,
                metric_title=plot_metric_title,
                save_path=detailed_save_path,
                super_title=f"(temperature, top-p) = ({temp_val}, {top_p_val})",
                show_x=show_x,
                show_y=show_y,
                show_cbar=show_cbar
            )

# === Entry Point ===

if __name__ == "__main__":
    process_and_plot_all_eval_data(
        temp_topps=temperatures_and_topps,
        config_names_list=subfolder_config_names,
        gen_indices=generation_indices,
        metrics=metrics_to_plot
    )