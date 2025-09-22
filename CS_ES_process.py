import os
import json
import numpy as np
from tqdm import tqdm

def aggregate_metrics_for_subfolder(subfolder_path):
    """
    Reads all generation_*.json files in a given subfolder,
    aggregates Best CS, Avg CS, Best ES, Avg ES for each 'i',
    and updates the rougeL_summary.json file.
    """
    print(f"\nProcessing subfolder: {subfolder_path}")

    # Initialize dictionaries to store metrics for each 'i'
    best_cs_data = {}
    avg_cs_data = {}
    best_es_data = {}
    avg_es_data = {}

    # Define the 'i' values
    generation_indices = [1, 2, 4, 8, 16, 32, 64, 128]
#     generation_indices

    for i in tqdm(generation_indices, desc="Aggregating generation files"):
        file_name = f"generations_n{i}.json"
        file_path = os.path.join(subfolder_path, file_name)

        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}. Skipping.")
            continue

        current_best_cs_values = []
        current_avg_cs_values = []
        current_best_es_values = []
        current_avg_es_values = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Extract metrics, handling potential None values
                    if data.get("Best CS") is not None:
                        current_best_cs_values.append(data["Best CS"])
                    if data.get("Avg CS") is not None:
                        current_avg_cs_values.append(data["Avg CS"])
                    if data.get("Best ES") is not None:
                        current_best_es_values.append(data["Best ES"])
                    if data.get("Avg ES") is not None:
                        current_avg_es_values.append(data["Avg ES"])

            # Calculate average for the current 'i'
            # Using .item() to convert numpy scalars to native Python floats for JSON serialization
            best_cs_data[f"generations_n{i}"] = np.mean(current_best_cs_values).item() if current_best_cs_values else 0.0
            avg_cs_data[f"generations_n{i}"] = np.mean(current_avg_cs_values).item() if current_avg_cs_values else 0.0
            best_es_data[f"generations_n{i}"] = np.mean(current_best_es_values).item() if current_best_es_values else 0.0
            avg_es_data[f"generations_n{i}"] = np.mean(current_avg_es_values).item() if current_avg_es_values else 0.0

        except json.JSONDecodeError as e:
            print(f"  Error: Malformed JSON in {file_path}. Skipping. Error: {e}")
        except Exception as e:
            print(f"  An unexpected error occurred processing {file_path}: {e}")

    # Load existing summary and update
    summary_file_path = os.path.join(subfolder_path, "rougeL_summary.json")
    summary_data = {}
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            try:
                summary_data = json.load(f)
            except json.JSONDecodeError:
                print(f"  Warning: {summary_file_path} is malformed or empty. Overwriting with new data.")
                summary_data = {}
    
    # Add new metrics to the summary data
    # Using descriptive keys for clarity
    summary_data["best_cs_avg_per_generation"] = best_cs_data
    summary_data["avg_cs_avg_per_generation"] = avg_cs_data
    summary_data["best_es_avg_per_generation"] = best_es_data
    summary_data["avg_es_avg_per_generation"] = avg_es_data

    # Save the updated summary
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Updated {summary_file_path} with CS and ES averages.")


def process_all_main_folders(main_folder_paths: list):
    """
    Processes a list of main folders. For each main folder, it processes
    both the 'forget' and 'retain' subdirectories.
    """
    print("Starting batch processing for all main folders...")
    print("-" * 50)

    for root_folder in main_folder_paths:
        if not os.path.isdir(root_folder):
            print(f"Error: Main folder '{root_folder}' not found. Skipping.")
            continue

        print(f"\nProcessing main folder: {root_folder}")
        
        # Process both 'forget' and 'retain' folders
        for data_type_folder_name in ["forget", "retain"]:
            data_type_path = os.path.join(root_folder, data_type_folder_name)
            
            if not os.path.isdir(data_type_path):
                print(f"  Warning: '{data_type_folder_name}' folder not found in {root_folder}. Skipping.")
                continue

            print(f"  Entering {data_type_folder_name} directory.")
            
            # Iterate through the five subfolders within 'forget'/'retain'
            for item in os.listdir(data_type_path):
                subfolder_path = os.path.join(data_type_path, item)
                if os.path.isdir(subfolder_path): # Ensure it's actually a directory
                    aggregate_metrics_for_subfolder(subfolder_path)
        
        print(f"Finished processing all relevant folders in {root_folder}.")
        print("-" * 50)
    
    print("All specified main folders processed successfully!")


# --- Example Usage ---
if __name__ == "__main__":
  


    # Define your list of main_folder paths here
    my_main_folders = [
        # "/users/2/jruan/pass-k/saves/eval/BLUR-NPO",
        # "/users/2/jruan/pass-k/saves/eval/GradDiff",
        # "/users/2/jruan/pass-k/saves/eval/NPO",
        # "/users/2/jruan/pass-k/saves/eval/NPO+ENT",
        "/users/2/jruan/pass-k/saves/eval/ada",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha4gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha2gamma1"
        # "/users/2/jruan/pass-k/saves/eval/beta1alpha2gamma1ada",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha2gamma1ada",
        # "/users/2/jruan/pass-k/saves/eval/Original",
        "/users/2/jruan/pass-k/saves/eval/Retrain",
        # "/users/2/jruan/pass-k/saves/eval/SimNPO",
        # "/users/2/jruan/pass-k/saves/eval/RMU",
        # "/users/2/jruan/pass-k/saves/eval/GradAscent"
        # "/users/2/jruan/pass-k/saves/eval/beta1alpha4gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta1alpha2gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha4gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha2gamma1",
        ]
    
    # Call the main processing function
    process_all_main_folders(my_main_folders)

