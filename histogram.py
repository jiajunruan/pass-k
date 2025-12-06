import os
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
# Note: sklearn.metrics.pairwise.cosine_similarity is fine, but for PyTorch tensors,
# torch.nn.functional.cosine_similarity is often more direct.
# My `eval_cosine_similarity_with_model` already uses the torch version.
# from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
def process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task="retain", tofu=True):
    """
    Processes multiple generated responses for a single query,
    calculates and adds CS and ES metrics, then summarizes best/avg CS/ES.
    """

    responses = query_data["responses"]  # len = 200

    # ---- 你要的 prefix ES 平均值计算 ----
    prefix_es = []
    running_sum = 0.0

    for i in range(1, 201):  # i = 1..200
        running_sum += responses[i-1]["ES"]   # 加入第 i 个 response 的 ES
            # 前 i 个 ES 的平均值
    if running_sum >=0 and running_sum<25:
        return 1
    elif running_sum >=25 and running_sum<50:
        return 2
    elif running_sum >=50 and running_sum<75:
        return 3
    elif running_sum >=75 and running_sum<100:
        return 4
    elif running_sum >=100 and running_sum<125:
        return 5
    elif running_sum >=125 and running_sum<150:
        return 6
    elif running_sum >=150 and running_sum<175:
        return 7
    else:
        return 8

from collections import Counter # 记得在文件开头导入这个

def main_processing_script(input_file_path: str, output_file_path: str, nli_pipeline, sentence_transformer_model, eval_task: str = "retain", tofu: bool = True):
    """
    Loads, processes, and saves a single JSONL/JSON file with CS/ES metrics.
    """
    processed_results = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f"Processing {os.path.basename(input_file_path)}"):
            try:
                query_data = json.loads(line)
                # 这里返回的是 1, 2, 3, 4, 5 中的一个数字
                prefix_es = process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task, tofu)
                processed_results.append(prefix_es)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line: {e}")
                continue
            except Exception as e:
                print(f"Error processing query: {e}")
                continue
    
    # ---- 新增统计代码 ----
    print(f"\n[Statistics] Results for {os.path.basename(input_file_path)}:")
    counts = Counter(processed_results)
    total = len(processed_results)
    
    # 按照 1 到 5 的顺序打印，如果没有该数字则显示 0
    for i in range(1, 9):
        count = counts.get(i, 0) # 获取次数，默认为0
        percentage = (count / total * 100) if total > 0 else 0
        print(f"Category {i}: {count} ({percentage:.2f}%)")
    
    return processed_results

#     # ---- 新增：跨 400 行聚合 ----
#     num_queries = len(processed_results)   # 比如 400
#     final_avg = []

#     for i in range(200):   # i = 0~199 → prefix index
#         s = 0.0
#         for q in range(num_queries):
#             s += processed_results[q][i]   # 第 q 行的第 i 个 prefix_es
#         final_avg.append(s / num_queries)

#     # 转为 {"1": ..., "2": ..., ..., "200": ...}
#     output_dict = {str(i + 1): final_avg[i] for i in range(200)}

#     # ---- 输出 ----
#     with open(output_file_path, 'w', encoding='utf-8') as outfile:
#         outfile.write(json.dumps(output_dict, ensure_ascii=False))

    
    # print(f"Updated: {os.path.basename(output_file_path)}") # Print concise message per file
    # Only print this for the overall script, not each file processing.

def process_all_generation_jsons(root_dirs: list, eval_task: str = "forget", tofu: bool = True):
    """
    Traverses given root directories, finds all JSON files with "generation" in their name,
    and processes them in-place with CS/ES metrics.
    """
    print("Loading SentenceTransformer model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)
    print("Loading NLI pipeline...")
    nli_pipeline = pipeline("text-classification", model="sileod/deberta-v3-base-tasksource-nli", device=device)
    print("-" * 50)

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"Warning: Root directory '{root_dir}' not found. Skipping.")
            continue
        
        print(f"Searching in: {root_dir}")
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                # Check for .json extension and "generation" in filename
                if filename.endswith(".json") and "generations_n200" in filename:
                    file_path = os.path.join(dirpath, filename)
                    output_file_path = os.path.join(dirpath, "rougeL_summary.json")
                    print(f"Processing: {file_path}")
                    try:
                        main_processing_script(
                            input_file_path=file_path,
                            output_file_path=output_file_path, # Overwrite the file
                            nli_pipeline=nli_pipeline,
                            sentence_transformer_model=sentence_transformer_model,
                            eval_task=eval_task,
                            tofu=tofu
                        )
                        print(f"Successfully updated: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                # else: # Optional: uncomment to see skipped files
                #     print(f"Skipping {filename} in {dirpath}")
        print("-" * 50)
    print("All specified directories processed.")

# --- Example Usage ---
if __name__ == "__main__":

    my_root_directories = [
#         "/users/2/jruan/pass-k/saves/eval/Lora/forget/temperature=0.2top_p=0.2",
#         "/users/2/jruan/pass-k/saves/eval/NPO-fix/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/forget/temperature=1.top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/forget/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=1.0top_p=1.0",
        "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=0.8top_p=1.0",        
        "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=0.2top_p=0.2",
#         "/users/2/jruan/pass-k/saves/eval/RMU/forget/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/Original/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/Original/forget/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/SimNPO/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/SimNPO/forget/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/NPO+ENT/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/NPO+ENT/forget/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/GradDiff/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/GradDiff/forget/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=1.0top_p=0.8",
#         "/users/2/jruan/pass-k/saves/eval/NPO-fix/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/NPO-fix/retain/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/pmc-lm/retain/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/pmc-lm/retain/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/Lora/retain/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/Lora/retain/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=1.0top_p=0.2",
        # "/users/2/jruan/pass-k/saves/eval/Retrain/forget/temperature=0.8top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/Retrain/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=1.0top_p=0.8",
#         "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=1.0top_p=0.2",
#         "/users/2/jruan/pass-k/saves/eval/NPO+gen3/forget/temperature=1.0top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/NPO2/retain/temperature=1.0top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/ada/retain",
        # "/users/2/jruan/pass-k/saves/eval/GradDiff/forget/temperature=0.8top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/GradDiff/forget/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/GradDiff/retain/temperature=1.0top_p=0.8",
#         "/users/2/jruan/pass-k/saves/eval/GradDiff/retain/temperature=1.0top_p=0.2",
        # "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=0.8top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/NPO/forget/temperature=1.0top_p=1.0",
        
        
# #         "/users/2/jruan/pass-k/saves/eval/NPO/retain/temperature=1.0top_p=0.8",
# #         "/users/2/jruan/pass-k/saves/eval/NPO/retain/temperature=1.0top_p=0.2",
        
        
        
#         "/users/2/jruan/pass-k/saves/eval/NPO+ENT/forget/temperature=0.8top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/NPO+ENT/forget/temperature=1.0top_p=1.0",
# #         "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=1.0top_p=0.8",
# #         "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=1.0top_p=0.2",
# #         "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=0.2top_p=0.2",
#         # "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=0.0top_p=0.0",
#         "/users/2/jruan/pass-k/saves/eval/SimNPO/forget/temperature=0.8top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/SimNPO/forget/temperature=1.0top_p=1.0",
# #         "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=1.0top_p=0.8",
# #         "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=1.0top_p=0.2",
#         "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/forget/temperature=0.8top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/forget/temperature=1.0top_p=1.0",
# #         "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/retain/temperature=1.0top_p=0.8",
# #         "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/retain/temperature=1.0top_p=0.2",
#         # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.2top_p=0.2",
#         # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.2top_p=1.0",
#         # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.8top_p=0.2",
#         # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.8top_p=1.0", 
#         # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=1.0top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/RMU/forget/temperature=0.8top_p=1.0",
#         "/users/2/jruan/pass-k/saves/eval/RMU/forget/temperature=1.0top_p=1.0",
# #         "/users/2/jruan/pass-k/saves/eval/RMU/retain/temperature=1.0top_p=0.8",
# #         "/users/2/jruan/pass-k/saves/eval/RMU/retain/temperature=1.0top_p=0.2",
#         # "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=0.0top_p=0.0",
#         # "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=0.0top_p=0.0",
#         # "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=0.0top_p=0.0",
# #         "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/forget/temperature=0.2top_p=0.2"

    ]
    # Run the main processing function
    process_all_generation_jsons(
        root_dirs=my_root_directories,
        eval_task="forget", # Set your eval_task
        tofu=True            # Set your tofu preference
    )