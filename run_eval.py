import os
import sys
import json
import time
import subprocess
import argparse
import numpy as np
from tqdm import tqdm

# ==========================================
#        CORE EVALUATION FUNCTIONS
# ==========================================
# (No global torch/transformers imports to protect the Master GPU memory)

def eval_cosine_similarity_with_model(gen_outputs, ground_truths, model):
    import torch
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

def process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task="forget", tofu=True):
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

def main_processing_script(input_file_path: str, output_file_path: str, nli_pipeline, sentence_transformer_model, eval_task: str = "forget", tofu: bool = True):
    processed_results = []
    
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f"Processing {os.path.basename(input_file_path)}", leave=False):
            try:
                query_data = json.loads(line)
                processed_query_data = process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task, tofu)
                processed_results.append(processed_query_data)
            except Exception as e:
                print(f"⚠️ Error processing a line in {input_file_path}: {e}")
                continue

    # --- SAFETY CHECK: Do not write if processing failed completely ---
    if len(processed_results) == 0:
        print(f"❌ CRITICAL: No valid data processed for {input_file_path}. Aborting write to prevent empty file creation.")
        return False

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for data in processed_results:
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    return True


# ==========================================
#        ISOLATED WORKER PROCESS
# ==========================================
def run_worker(seed):
    """
    Runs in total isolation. Imports PyTorch safely on the masked GPU.
    """
    import torch
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n[Seed {seed}] 🧠 Loading Eval Models onto GPU...")
    sentence_transformer_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)
    nli_pipeline = pipeline("text-classification", model="sileod/deberta-v3-base-tasksource-nli", device=device)

    root_dir = f"saves/eval/grad_{seed}/forget/temperature=1.0top_p=1.0"

    if not os.path.isdir(root_dir):
        print(f"[Seed {seed}] ⚠️ Directory not found: {root_dir}")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Only process the original file, ignore already evaluated ones
            if filename == "generations_n200.json":
                input_file = os.path.join(dirpath, filename)
                
                # Create a perfectly safe output name
                output_file = os.path.join(dirpath, "generations_n200_evaluated.json")
                
                print(f"[Seed {seed}] ⚙️ Evaluating: {input_file}")
                
                success = main_processing_script(
                    input_file_path=input_file,
                    output_file_path=output_file, 
                    nli_pipeline=nli_pipeline,
                    sentence_transformer_model=sentence_transformer_model,
                    eval_task="forget",
                    tofu=True
                )
                
                if success:
                    print(f"[Seed {seed}] ✅ Safely saved results to: {output_file}")


# ==========================================
#             MAIN ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # If this argument is present, the script knows it is a worker
    parser.add_argument("--worker_seed", type=int, default=None, help="Run evaluation for a specific seed")
    args = parser.parse_args()

    # --- WORKER MODE ---
    if args.worker_seed is not None:
        run_worker(args.worker_seed)
        sys.exit(0)

    # --- ORCHESTRATOR MODE ---
    available_gpus = [0, 1, 2, 3,4,5,6,7]
    total_seeds = 25
    
    print(f"=== Starting Isolated GPU Post-Evaluation Queue for {total_seeds} seeds ===")
    
    free_gpus = available_gpus.copy()
    seeds_to_eval = list(range(1, total_seeds + 1))
#     seeds_to_eval = list(range(2,3))
    active_eval_processes = []
    
    script_name = sys.argv[0]

    while seeds_to_eval or active_eval_processes:
        # 1. Check for finished processes
        for p_info in active_eval_processes[:]:
            process, seed, gpu_id = p_info
            
            if process.poll() is not None:
                retcode = process.returncode
                if retcode == 0:
                    print(f"✅ Worker for seed {seed} on GPU {gpu_id} finished gracefully.")
                else:
                    print(f"❌ Worker for seed {seed} on GPU {gpu_id} crashed (Code {retcode}).")
                
                active_eval_processes.remove(p_info)
                free_gpus.append(gpu_id)

        # 2. Launch new isolated processes
        while free_gpus and seeds_to_eval:
            gpu_id = free_gpus.pop(0)
            seed = seeds_to_eval.pop(0)

            # Mask everything except the assigned GPU, then call this exact script in worker mode
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} python {script_name} --worker_seed {seed}"

            print(f"🚀 Launching Post-Eval for seed {seed} on GPU {gpu_id}...")
            process = subprocess.Popen(command, shell=True)
            active_eval_processes.append((process, seed, gpu_id))

            # Stagger launches so all 4 GPUs don't try to load DeBERTa from disk at the exact same millisecond
            time.sleep(5) 
            
        time.sleep(1)

    print("\n🎉🎉 All 25 seeds fully evaluated!")