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
    # generation_indices = [1,2,4,8,16,32,64,128]
    generation_indices = [200]
#     generation_indices

    for i in tqdm(generation_indices, desc="Aggregating generation files"):
        file_name = f"generations_n{i}.json"
        file_path = os.path.join(subfolder_path, file_name)

        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}. Skipping.")
            file_path = os.path.join(subfolder_path, "generations_n200.json")
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
                    try:
                        data = json.loads(line)

                        # Check if data is a list or dictionary
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if item.get("Best CS") is not None:
                                        current_best_cs_values.append(item["Best CS"])
                                    if item.get("Avg CS") is not None:
                                        current_avg_cs_values.append(item["Avg CS"])
                                    if item.get("Best ES") is not None:
                                        current_best_es_values.append(item["Best ES"])
                                    if item.get("Avg ES") is not None:
                                        current_avg_es_values.append(item["Avg ES"])
                        elif isinstance(data, dict):
                            if data.get("Best CS") is not None:
                                current_best_cs_values.append(data["Best CS"])
                            if data.get("Avg CS") is not None:
                                current_avg_cs_values.append(data["Avg CS"])
                            if data.get("Best ES") is not None:
                                current_best_es_values.append(data["Best ES"])
                            if data.get("Avg ES") is not None:
                                current_avg_es_values.append(data["Avg ES"])
                        else:
                            print(f"  Warning: Unexpected JSON structure in {file_path}. Skipping.")
                    except json.JSONDecodeError as e:
                        print(f"  Error: Malformed JSON in {file_path}. Skipping line. Error: {e}")
        except Exception as e:
            print(f"  An unexpected error occurred processing {file_path}: {e}")

        # Calculate average for the current 'i'
        # Using .item() to convert numpy scalars to native Python floats for JSON serialization
        best_cs_data[f"generations_n{i}"] = np.mean(current_best_cs_values).item() if current_best_cs_values else 0.0
        avg_cs_data[f"generations_n{i}"] = np.mean(current_avg_cs_values).item() if current_avg_cs_values else 0.0
        best_es_data[f"generations_n{i}"] = np.mean(current_best_es_values).item() if current_best_es_values else 0.0
        avg_es_data[f"generations_n{i}"] = np.mean(current_avg_es_values).item() if current_avg_es_values else 0.0
        print(current_avg_es_values)

    # Define the path for the summary file
    summary_file_path = os.path.join(subfolder_path, "avg_summary.json")

    summary_data = {}  # Ensure summary_data is a dictionary

    # Prepare the output in the desired format using only best_es_data
    for i in generation_indices:
        summary_data[str(i)] = avg_es_data.get(f"generations_n{i}", 0.0)
        print(avg_es_data.get(f"generations_n{i}", 0.0))

    # Directly overwrite the file with new data
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Overwritten {summary_file_path} with new data in the desired format using only ES best.")


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
        for data_type_folder_name in ["retain"]:
            data_type_path = os.path.join(root_folder, data_type_folder_name)
            print(f"Entering {data_type_path} directory.")
            
            if not os.path.isdir(data_type_path):
                print(f"  Warning: '{data_type_folder_name}' folder not found in {root_folder}. Skipping.")
                continue

            print(f"  Entering {data_type_folder_name} directory.")
            
            # Iterate through the five subfolders within 'forget'/'retain'
            for item in os.listdir(data_type_path):
                subfolder_path = os.path.join(data_type_path, item)
                # Check if the folder name contains the specified temperature and top_p values
                if any(f"temperature={temp}top_p={top_p}" in item for temp, top_p in [(1.0, 0.8)]):
                    print(f"Found matching subfolder: {item}")
                    print(subfolder_path)
                    if os.path.isdir(subfolder_path):  # Ensure it's actually a directory
                        print(subfolder_path)
                        aggregate_metrics_for_subfolder(subfolder_path)

                aggregate_metrics_for_subfolder(subfolder_path)
                
        
        print(f"Finished processing all relevant folders in {root_folder}.")
        print("-" * 50)
    
    print("All specified main folders processed successfully!")


# --- Example Usage ---
if __name__ == "__main__":
  


    # Define your list of main_folder paths here
    my_main_folders = [
        "/users/2/jruan/pass-k/saves/eval/BLUR-NPO",
        "/users/2/jruan/pass-k/saves/eval/GradDiff",
        "/users/2/jruan/pass-k/saves/eval/LoUK",
#         "/users/2/jruan/pass-k/saves/eval/NPO+ENT",
        "/users/2/jruan/pass-k/saves/eval/NPO",
#         "/users/2/jruan/pass-k/saves/eval/ada",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha4gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha2gamma1"
        # "/users/2/jruan/pass-k/saves/eval/beta1alpha2gamma1ada",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha2gamma1ada",
        "/users/2/jruan/pass-k/saves/eval/Original",
        "/users/2/jruan/pass-k/saves/eval/Retrain",
        "/users/2/jruan/pass-k/saves/eval/SimNPO",
        "/users/2/jruan/pass-k/saves/eval/RMU",
#         "/users/2/jruan/pass-k/saves/eval/Original",
        # "/users/2/jruan/pass-k/saves/eval/GradAscent"
        # "/users/2/jruan/pass-k/saves/eval/beta1alpha4gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta1alpha2gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha4gamma1",
        # "/users/2/jruan/pass-k/saves/eval/beta0.1alpha2gamma1",
        ]
    
    # Call the main processing function
    process_all_main_folders(my_main_folders)

