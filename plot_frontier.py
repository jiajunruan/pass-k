import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from scipy.interpolate import make_interp_spline

# === Configuration ===

temperatures_and_topps = [
    "temperature=0.0top_p=0.0",
    "temperature=0.2top_p=0.2",
    "temperature=0.2top_p=1.0",
    "temperature=0.8top_p=0.2",
    "temperature=0.8top_p=1.0",
    "temperature=1.0top_p=1.0"
]

subfolder_config_names = [
    "BLUR-NPO", "NPO", "NPO+ENT", "NPO+ENT+TMP", "SimNPO", "GradDiff"
]

generation_indices = [1, 2, 4, 8, 16, 32, 64, 128]

metrics_to_plot = [
    ("best_es_avg_per_generation", "Best Entailment Score", "best_es"),
]

output_base_dir = "forget_frontiers_normalized_final"
base_dir = "/users/2/jruan/pass-k/saves/eval"

# === Plotting Function ===

from matplotlib.patches import FancyArrowPatch

def save_smooth_frontier_plot(generations, all_model_data, upper_frontier, lower_frontier, config_names, metric_title, save_path, super_title):
    plt.figure(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(config_names)))

    # 绘制所有模型的曲线
    for i, config_name in enumerate(config_names):
        data = all_model_data.get(config_name, [])
        if data:
            plt.plot(generations, data, marker='o', linestyle='-', color=colors[i], alpha=0.6, label=config_name)

    # === 绘制平滑 Frontier 曲线 ===
    log_generations = np.log2(generations)

    # 拟合上边界
    poly_upper = np.poly1d(np.polyfit(log_generations, upper_frontier, 2))
    x_smooth = np.linspace(log_generations.min(), log_generations.max(), 300)
    y_smooth_upper = poly_upper(x_smooth)

    # 画带箭头的上边界曲线
    plt.plot(2**x_smooth, y_smooth_upper, color='red', linewidth=3)
    arrow = FancyArrowPatch(
        posA=(2**x_smooth[-20], y_smooth_upper[-20]),  # 箭头起点（靠近末尾）
        posB=(2**x_smooth[-1], y_smooth_upper[-1]),    # 箭头终点（末尾）
        arrowstyle='-|>', mutation_scale=20, color='red', linewidth=3
    )
    plt.gca().add_patch(arrow)

    # 拟合下边界
    poly_lower = np.poly1d(np.polyfit(log_generations, lower_frontier, 2))
    y_smooth_lower = poly_lower(x_smooth)

    # 画带箭头的下边界曲线
    plt.plot(2**x_smooth, y_smooth_lower, color='black', linewidth=3)
    arrow = FancyArrowPatch(
        posA=(2**x_smooth[-20], y_smooth_lower[-20]),
        posB=(2**x_smooth[-1], y_smooth_lower[-1]),
        arrowstyle='-|>', mutation_scale=20, color='black', linewidth=3
    )
    plt.gca().add_patch(arrow)

    # 标注
    plt.text(2**x_smooth[len(x_smooth)//2] * 0.9, y_smooth_upper[len(x_smooth)//2],
             "Worst Leak@k Performance", color='red', fontsize=14, ha='right', va='bottom')

    plt.text(2**x_smooth[len(x_smooth)//2] * 1.1, y_smooth_lower[len(x_smooth)//2],
             "Best Leak@k Performance", color='black', fontsize=14, ha='left', va='top')

    plt.xscale('log', base=2)
    plt.xticks(generations, labels=[str(n) for n in generations], fontsize=16)

    plt.xlabel("Number of Generations ($k$)", fontsize=16)
    plt.ylabel(f"Increment of {metric_title}", fontsize=16)
    plt.title(super_title, fontsize=18)

    plt.ylim([-0.1, 0.75])

    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()


# === Main Loop ===

def process_and_plot_all_eval_data(temp_topps, config_names_list, gen_indices, metrics):
    for json_key_prefix, plot_metric_title, filename_prefix in metrics:
        for temp_top in temp_topps:
            print(f"Processing {temp_top} - {plot_metric_title}")

            raw_model_data = {}
            for config_name in config_names_list:
                raw_model_data[config_name] = []

            for config_name in config_names_list:
                full_path = os.path.join(base_dir, f"{config_name}","forget")
                summary_file_path = os.path.join(full_path, temp_top, "rougeL_summary.json")

                if not os.path.exists(summary_file_path):
                    print(f"⚠️ Missing {summary_file_path}, skipping.")
                    continue

                with open(summary_file_path, 'r') as f:
                    summary_data = json.load(f)

                if json_key_prefix == "generations_n":
                    for n in gen_indices:
                        raw_model_data[config_name].append(summary_data.get(f"generations_n{n}", 0.0))
                else:
                    nested_data = summary_data.get(json_key_prefix, {})
                    for n in gen_indices:
                        raw_model_data[config_name].append(nested_data.get(f"generations_n{n}", 0.0))
            
            all_model_data_normalized = {}
            valid_raw_data = []

            for config_name, data in raw_model_data.items():
                if data and len(data) == len(gen_indices):
                    k1_value = data[gen_indices.index(1)]
                    normalized_data = [x - k1_value for x in data]
                    all_model_data_normalized[config_name] = normalized_data
                    valid_raw_data.append(normalized_data)
                
            if not valid_raw_data:
                print(f"No valid data found for {temp_top}, skipping plot.")
                continue

            valid_models_data = np.array(valid_raw_data)
            
            upper_frontier = np.max(valid_models_data, axis=0)
            lower_frontier = np.min(valid_models_data, axis=0)

            filename = f"{filename_prefix}_{temp_top}_frontiers_normalized_final.pdf"
            frontier_save_path = os.path.join(output_base_dir, filename)
            os.makedirs(output_base_dir, exist_ok=True)

            temp_val = temp_top.split("temperature=")[1].split("top_p=")[0]
            top_p_val = temp_top.split("top_p=")[1]
            super_title = f"Entailment Score Increment at (temperature, top-p) = ({temp_val}, {top_p_val})"

            save_smooth_frontier_plot(
                generations=gen_indices,
                all_model_data=all_model_data_normalized,
                upper_frontier=upper_frontier,
                lower_frontier=lower_frontier,
                config_names=config_names_list,
                metric_title=plot_metric_title,
                save_path=frontier_save_path,
                super_title=super_title
            )

# === Entry Point ===

if __name__ == "__main__":
    process_and_plot_all_eval_data(
        temp_topps=temperatures_and_topps,
        config_names_list=subfolder_config_names,
        gen_indices=generation_indices,
        metrics=metrics_to_plot
    )