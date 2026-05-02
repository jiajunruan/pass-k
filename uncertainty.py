import subprocess
import time
import os
import json
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm

# --- Phase 2: Evaluation Helper Functions ---
# (These remain exactly as you wrote them, placed here so workers can use them)

def eval_cosine_similarity_with_model(gen_outputs, ground_truths, model):
    scores = []
    with torch.no_grad():
        for gen, gt in zip(gen_outputs, ground_truths):
            gen_str = str(gen) if gen is not None else ""
            gt_str = str(gt) if gt is not None else ""
            if not gen_str or not gt_str:
                scores.append(0.0)
                continue
            gen_embedding = model.encode(gen_str, show_progress_bar=False, convert_to_tensor=True)
            gt_embedding = model.encode(gt_str, show_progress_bar=False, convert_to_tensor=True)
            cosine_sim = torch.nn.functional.cosine_similarity(gen_embedding, gt_embedding, dim=0).item()
            scores.append(float(max(0, cosine_sim)))
    return {'cosine_similarity': scores}

def get_entailment_score(entailment_labels):
    correct = sum(1 for label in entailment_labels if label == 'entailment')
    return correct / len(entailment_labels) if len(entailment_labels) > 0 else 0.0

def get_entailment_results_with_pipeline(pipe, gen_outputs, ground_truths, eval_task, rouge_scores, bs=30, tofu=True):
    results = []
    if len(gen_outputs) != len(ground_truths):
        ground_truths = [ground_truths[0]] * len(gen_outputs)

    for i in range(0, len(gen_outputs), bs):
        targets_batch = ground_truths[i:i + bs]
        outputs_batch = gen_outputs[i:i + bs]
        rouge_scores_batch = rouge_scores[i:i + bs]

        data_list = []
        for j in range(len(targets_batch)):
            output_text = str(outputs_batch[j]) if outputs_batch[j] is not None else ""
            target_text = str(targets_batch[j]) if targets_batch[j] is not None else ""

            if not output_text or not target_text:
                results.append({'label': 'none', 'score': 0.0})
                continue

            if not tofu:
                data_list.append({'text': output_text, 'text_pair': target_text})
            else:
                if 'forget' in eval_task:
                    data_list.append({'text': output_text, 'text_pair': target_text})
                else:
                    data_list.append({'text': target_text, 'text_pair': output_text})
        
        if data_list:
            batch_results = pipe(data_list)
            filtered_batch_results = []
            valid_data_list_idx = 0
            for j in range(len(targets_batch)):
                if not (str(outputs_batch[j]) and str(targets_batch[j])):
                    continue
                if rouge_scores_batch[valid_data_list_idx] < 0.1:
                    filtered_batch_results.append({'label': 'none', 'score': 0.0})
                else:
                    filtered_batch_results.append(batch_results[valid_data_list_idx])
                valid_data_list_idx += 1
            results.extend(filtered_batch_results)
        else:
            for _ in range(len(outputs_batch)):
                if not (str(outputs_batch[_]) and str(targets_batch[_])):
                     results.append({'label': 'none', 'score': 0.0})
                     
    entailment_labels = [result['label'] for result in results]
    return {'entailment_labels': entailment_labels}

def process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task="retain", tofu=True):
    ground_truth = query_data["ground_truth"]
    responses = query_data["responses"]

    gen_outputs = [res["response"] for res in responses]
    rougeL_recalls = [res["rougeL_recall"] for res in responses]

    cs_scores_dict = eval_cosine_similarity_with_model(gen_outputs, [ground_truth] * len(gen_outputs), sentence_transformer_model)
    cs_scores = cs_scores_dict['cosine_similarity']

    es_labels_dict = get_entailment_results_with_pipeline(nli_pipeline, gen_outputs, [ground_truth] * len(gen_outputs), eval_task, rougeL_recalls, bs=30, tofu=tofu)
    es_labels = es_labels_dict['entailment_labels']
    es_scores = [1.0 if label == 'entailment' else 0.0 for label in es_labels]

    for i, res in enumerate(responses):
        res["CS"] = cs_scores[i]
        res["ES"] = es_scores[i]

    if cs_scores:
        query_data["Best CS"] = max(cs_scores)
        query_data["Avg CS"] = np.mean(cs_scores).item()
    else:
        query_data["Best CS"] = None
        query_data["Avg CS"] = None

    if es_scores:
        query_data["Best ES"] = max(es_scores)
        query_data["Avg ES"] = np.mean(es_scores).item()
    else:
        query_data["Best ES"] = None
        query_data["Avg ES"] = None
    
    query_data["responses"] = responses
    return query_data

def main_processing_script(input_file_path: str, output_file_path: str, nli_pipeline, sentence_transformer_model, eval_task: str = "retain", tofu: bool = True):
    processed_results = []
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f"Processing {os.path.basename(input_file_path)}", leave=False):
            try:
                query_data = json.loads(line)
                processed_query_data = process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task, tofu)
                processed_results.append(processed_query_data)
            except Exception as e:
                continue

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for data in processed_results:
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# --- Multiprocessing Worker for Phase 2 ---
def run_post_evaluation_worker(seed, gpu_id):
    """
    This runs in a completely separate process.
    It restricts visibility to a single GPU and loads the models there.
    """
    # 1. Isolate to the assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Imports inside the worker to prevent PyTorch CUDA fork initialization issues
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    
    print(f"\n[GPU {gpu_id}] 🧠 Loading Eval Models for seed {seed}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)
    nli_pipeline = pipeline("text-classification", model="sileod/deberta-v3-base-tasksource-nli", device=device)

    # 2. Build the exact path based on your format
    # Using lowercase 'npo_' to match the task_name created in Phase 1
    root_dir = f"saves/eval/npo_{seed}/forget/temperature=1.0top_p=1.0"

    if not os.path.isdir(root_dir):
        print(f"[GPU {gpu_id}] ⚠️ Directory not found: {root_dir}")
        return

    # 3. Find and process the generation files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json") and "generations_n200" in filename:
                file_path = os.path.join(dirpath, filename)
                print(f"[GPU {gpu_id}] ⚙️ Processing: {file_path}")
                try:
                    main_processing_script(
                        input_file_path=file_path,
                        output_file_path=file_path, 
                        nli_pipeline=nli_pipeline,
                        sentence_transformer_model=sentence_transformer_model,
                        eval_task="forget",
                        tofu=True
                    )
                    print(f"[GPU {gpu_id}] ✅ Successfully updated: {file_path}")
                except Exception as e:
                    print(f"[GPU {gpu_id}] ❌ Error processing {file_path}: {e}")

# ==========================================
#               MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    # REQUIRED for PyTorch multiprocessing with CUDA
    multiprocessing.set_start_method('spawn', force=True)

    model_path = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10"
#     model_path = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10"
    available_gpus = [0,1,2,3]
    total_seeds = 25

    # ---------------------------------------------------------
    # PHASE 1: GENERATION QUEUE (Subprocesses)
    # ---------------------------------------------------------
#     print(f"=== PHASE 1: Starting generation for {total_seeds} seeds ===")
#     free_gpus = available_gpus.copy()
#     seeds_to_run = list(range(1, total_seeds + 1))
#     active_processes = []

#     while seeds_to_run or active_processes:
#         # Check finished processes
#         for p_info in active_processes[:]:
#             process, task_name, gpu_id = p_info
#             if process.poll() is not None:
#                 retcode = process.returncode
#                 if retcode == 0:
#                     print(f"✅ '{task_name}' on GPU {gpu_id} finished successfully.")
#                 else:
#                     print(f"❌ '{task_name}' on GPU {gpu_id} exited with return code {retcode}.")
#                 active_processes.remove(p_info)
#                 free_gpus.append(gpu_id)
                
#         # Launch new processes
#         while free_gpus and seeds_to_run:
#             gpu_id = free_gpus.pop(0)
#             seed = seeds_to_run.pop(0)
#             task_name = f"npo_{seed}"
#             log_file = f"{task_name}.log"
            
#             command = (
#                 f"CUDA_VISIBLE_DEVICES={gpu_id} "
#                 "python src/eval.py "
#                 "--config-name=eval.yaml "
#                 "experiment=eval/tofu/default "
#                 "model=Llama-3.2-1B-Instruct "
#                 f"model.model_args.pretrained_model_name_or_path={model_path} "
#                 "retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json "
#                 f"task_name={task_name} "
#                 f"seed={seed} "  
#                 f"> {log_file} 2>&1"
#             )

#             print(f"🚀 Launching Generation '{task_name}' on GPU {gpu_id}...")
#             process = subprocess.Popen(command, shell=True)
#             active_processes.append((process, task_name, gpu_id))
#             time.sleep(2) # stagger launches
            
#         time.sleep(1)

#     print("\n🎉 Phase 1 (Generation) complete!")
    
    # ---------------------------------------------------------
    # PHASE 2: POST-EVALUATION QUEUE (Multiprocessing)
    # ---------------------------------------------------------
    print(f"\n=== PHASE 2: Starting Post-Evaluation for {total_seeds} seeds ===")
    
    free_gpus = available_gpus.copy()
    seeds_to_eval = list(range(1, total_seeds + 1))
    active_eval_processes = []

    while seeds_to_eval or active_eval_processes:
        # Check finished eval processes
        for p_info in active_eval_processes[:]:
            process, seed, gpu_id = p_info
            if not process.is_alive():
                print(f"✅ Post-Eval for seed {seed} on GPU {gpu_id} finished.")
                process.join() # Cleanly close the process
                active_eval_processes.remove(p_info)
                free_gpus.append(gpu_id)

        # Launch new eval processes
        while free_gpus and seeds_to_eval:
            gpu_id = free_gpus.pop(0)
            seed = seeds_to_eval.pop(0)

            # Spawn a distinct python process for the evaluation
            process = multiprocessing.Process(target=run_post_evaluation_worker, args=(seed, gpu_id))
            process.start()
            active_eval_processes.append((process, seed, gpu_id))

            time.sleep(3) # Stagger heavy model loading (DeBERTa + SentenceTransformer)
            
        time.sleep(1)

    print("\n🎉🎉 All 25 seeds fully generated and evaluated!")