import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.colors as mcolors

# --- Configuration ---
my_main_folders = [
    "/users/2/jruan/pass-k/saves/eval/BLUR-NPO",
    "/users/2/jruan/pass-k/saves/eval/GradDiff",
    "/users/2/jruan/pass-k/saves/eval/NPO",
    "/users/2/jruan/pass-k/saves/eval/NPO+ENT",
    "/users/2/jruan/pass-k/saves/eval/Original",
    "/users/2/jruan/pass-k/saves/eval/Retrain",
    "/users/2/jruan/pass-k/saves/eval/SimNPO",
    "/users/2/jruan/pass-k/saves/eval/RMU"
]

subfolder_config_names = [
    "temperature=0.2top_p=0.2",
    "temperature=0.8top_p=0.2",
    "temperature=0.2top_p=1.0",
    "temperature=0.8top_p=1.0",
    "temperature=1.0top_p=1.0"
]

generation_indices = [1, 2, 4, 8, 16, 32, 64, 128]

metrics_to_plot = [
    ("generations_n", "ROUGE-L Recall", "rouge_l"),
    ("best_cs_avg_per_generation", "Best Cosine Similarity", "best_cs"),
    ("avg_cs_avg_per_generation", "Average Cosine Similarity", "avg_cs"),
    ("best_es_avg_per_generation", "Best Entailment Score", "best_es"),
    ("avg_es_avg_per_generation", "Average Entailment Score", "avg_es")
]

output_base_dir = "results_plots"

# --- Save single detailed heatmap with ticks, title, colorbar ---
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
    cmap = plt.cm.get_cmap('RdBu_r')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = plt.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    plt.xticks(ticks=np.arange(len(x_unique)), labels=[str(n) for n in x_unique], rotation=45)
    plt.yticks(ticks=np.arange(len(y_tick_labels)), labels=y_tick_labels)

    plt.xlabel("Number of Generations (n)")
    plt.ylabel("Generation Config (Temperature & Top_p)")
    plt.title(super_title, fontsize=14)

    cbar = plt.colorbar(im, ax=plt.gca(), orientation='vertical', pad=0.02)
    cbar.set_label(metric_title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved detailed heatmap: {save_path}")

# --- Save minimalist square heatmap (no ticks, no internal title) for collage ---
def save_minimal_heatmap_for_collage(x_vals, y_vals, colors, save_path):
    x_unique = sorted(list(set(x_vals)))
    y_unique = sorted(list(set(y_vals)))
    heatmap = np.zeros((len(y_unique), len(x_unique)))

    for xi, yi, ci in zip(x_vals, y_vals, colors):
        x_idx = x_unique.index(xi)
        y_idx = y_unique.index(yi)
        heatmap[y_idx, x_idx] = ci

    plt.figure(figsize=(4, 3))
    cmap = plt.cm.get_cmap('RdBu_r')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = plt.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    plt.xticks([])
    plt.yticks([])

    # Add colorbar only on the rightmost images when collage
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved minimal heatmap: {save_path}")

from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np # Import numpy for creating dummy data if needed

def combine_images_grid_with_titles(image_paths, titles, grid_shape, save_path, super_title):
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1]*4, grid_shape[0]*3))

    # Determine the number of actual image plots
    num_images = len(image_paths)

    # We need to store the mappable object for the colorbar
    # This will be the last heatmap plotted before the "extra" axes if any
    last_im = None

    for idx, ax in enumerate(axes.flatten()):
        if idx < num_images:
            img_path = image_paths[idx]
            title = titles[idx]

            # Open the image. Assuming these are single-channel (grayscale) heatmaps.
            img = Image.open(img_path)
            # Convert to a NumPy array for imshow
            img_data = np.array(img)

            # Display the image with the RdBu_r colormap
            # vmin and vmax ensure the color mapping is consistent across all heatmaps
            im = ax.imshow(img_data, cmap='RdBu_r', vmin=0, vmax=1) # 0 is blue, 1 is red
            ax.axis('off')
            ax.set_title(title, fontsize=10)
            last_im = im # Keep track of the last image plot for the colorbar
        else:
            # Remove any extra empty axes
            ax.axis('off')

    # Add a single global colorbar on the right middle
    if last_im: # Only add colorbar if there was at least one image
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # left, bottom, width, height
        fig.colorbar(last_im, cax=cbar_ax) # Use the last_im for the colorbar

    fig.suptitle(super_title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # leave space for colorbar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved combined grid image with centered colorbar: {save_path}")



# --- Main Processing ---
def process_and_plot_all_eval_data(main_folders_list, config_names_list, gen_indices, metrics):
    for data_type in ["forget", "retain"]:
        for json_key_prefix, plot_metric_title, filename_prefix in metrics:
            image_paths_for_collage = []
            titles_for_collage = []

            for main_folder_path in main_folders_list:
                main_model_name = os.path.basename(main_folder_path)
                print(f"Processing {main_model_name} - {data_type} - {plot_metric_title}")

                data_type_path = os.path.join(main_folder_path, data_type)
                if not os.path.isdir(data_type_path):
                    print(f"⚠️ Skipping missing {data_type} folder in {main_model_name}.")
                    continue

                collected_data = []
                x_coords = []
                y_coords = []

                for config_idx, subfolder_name in enumerate(config_names_list):
                    subfolder_path = os.path.join(data_type_path, subfolder_name)
                    summary_file_path = os.path.join(subfolder_path, "rougeL_summary.json")

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

                # Save detailed heatmap with ticks and title
                detailed_save_path = os.path.join(
                    output_base_dir, data_type, main_model_name, f"{filename_prefix}_full.png"
                )
                save_detailed_heatmap(
                    x_coords, y_coords, collected_data,
                    gen_indices, config_names_list,
                    metric_title=plot_metric_title,
                    save_path=detailed_save_path,
                    super_title=f"{plot_metric_title} for {main_model_name} ({data_type.capitalize()})"
                )

                # Save minimal heatmap for collage
                minimal_save_path = os.path.join(
                    output_base_dir, data_type, main_model_name, f"{filename_prefix}_mini.png"
                )
                save_minimal_heatmap_for_collage(
                    x_coords, y_coords, collected_data, minimal_save_path
                )
                image_paths_for_collage.append(minimal_save_path)
                titles_for_collage.append(main_model_name)

            # Combine collage
            combined_save_path = os.path.join(output_base_dir, data_type, f"{filename_prefix}_combined.png")
            combine_images_grid_with_titles(
                image_paths_for_collage, titles_for_collage,
                grid_shape=(2, 4),
                save_path=combined_save_path,
                super_title=plot_metric_title
            )

    print("\n✅ All processing complete.")

# --- Run ---
if __name__ == "__main__":
    process_and_plot_all_eval_data(
        main_folders_list=my_main_folders,
        config_names_list=subfolder_config_names,
        gen_indices=generation_indices,
        metrics=metrics_to_plot
    )
