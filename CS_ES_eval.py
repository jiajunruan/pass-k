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

# --- Core Processing Function ---
def process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task="retain", tofu=True):
    """
    Processes multiple generated responses for a single query,
    calculates and adds CS and ES metrics, then summarizes best/avg CS/ES.
    """
    ground_truth = query_data["ground_truth"]
    responses = query_data["responses"]

    gen_outputs = [res["response"] for res in responses]
    rougeL_recalls = [res["rougeL_recall"] for res in responses]

    # 1. Calculate CS for all responses
    cs_scores_dict = eval_cosine_similarity_with_model(gen_outputs, [ground_truth] * len(gen_outputs), sentence_transformer_model)
    cs_scores = cs_scores_dict['cosine_similarity']

    # 2. Calculate ES for all responses
    es_labels_dict = get_entailment_results_with_pipeline(nli_pipeline, gen_outputs, [ground_truth] * len(gen_outputs), eval_task, rougeL_recalls, bs=30, tofu=tofu)
    es_labels = es_labels_dict['entailment_labels']
    es_scores = [1.0 if label == 'entailment' else 0.0 for label in es_labels]

    # 3. Add CS and ES to each response entry
    for i, res in enumerate(responses):
        res["CS"] = cs_scores[i]
        res["ES"] = es_scores[i]

    # 4. Calculate Best CS, Best ES, Avg CS, Avg ES for the query
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

# --- Helper Functions (Modified to accept pre-loaded models) ---

def eval_cosine_similarity_with_model(gen_outputs, ground_truths, model):
    """Calculates cosine similarity between generated texts and ground truths."""
    scores = []
    with torch.no_grad():
        for gen, gt in zip(gen_outputs, ground_truths):
            # Ensure inputs are strings, sometimes they can be None or other types after processing
            gen_str = str(gen) if gen is not None else ""
            gt_str = str(gt) if gt is not None else ""
            
            # Handle empty strings to prevent errors in model.encode
            if not gen_str or not gt_str:
                scores.append(0.0)
                continue

            gen_embedding = model.encode(gen_str, show_progress_bar=False, convert_to_tensor=True)
            gt_embedding = model.encode(gt_str, show_progress_bar=False, convert_to_tensor=True)
            cosine_sim = torch.nn.functional.cosine_similarity(gen_embedding, gt_embedding, dim=0).item()
            scores.append(float(max(0, cosine_sim)))
    return {'cosine_similarity': scores}

def get_entailment_score(entailment_labels):
    """Calculates the proportion of 'entailment' labels."""
    correct = 0
    for label in entailment_labels:
        if label == 'entailment':
            correct += 1
    return correct / len(entailment_labels) if len(entailment_labels) > 0 else 0.0

def get_entailment_results_with_pipeline(pipe, gen_outputs, ground_truths, eval_task, rouge_scores, bs=30, tofu=True):
    """Evaluates entailment relationship between generated and ground truth texts."""
    results = []
    # Ensure ground_truths matches gen_outputs length
    if len(gen_outputs) != len(ground_truths):
        # This shouldn't happen if `[ground_truth] * len(gen_outputs)` is used correctly
        # but is a safeguard.
        ground_truths = [ground_truths[0]] * len(gen_outputs)

    for i in range(0, len(gen_outputs), bs):
        targets_batch = ground_truths[i:i + bs]
        outputs_batch = gen_outputs[i:i + bs]
        rouge_scores_batch = rouge_scores[i:i + bs] # Extract corresponding rouge scores

        data_list = []
        for j in range(len(targets_batch)):
            # Ensure inputs are strings and handle potential None values
            output_text = str(outputs_batch[j]) if outputs_batch[j] is not None else ""
            target_text = str(targets_batch[j]) if targets_batch[j] is not None else ""

            if not output_text or not target_text: # Skip empty inputs for NLI
                # If either is empty, assume no entailment and add a dummy result
                results.append({'label': 'none', 'score': 0.0}) # NLI pipe would likely fail on empty strings
                continue

            if not tofu:
                data_list.append({'text': output_text, 'text_pair': target_text})
            else:
                if 'forget' in eval_task:
                    # For forget set, usually we check if generation entails GT (undesired)
                    data_list.append({'text': output_text, 'text_pair': target_text})
                else:
                    # For retain/real author/real world, check if GT entails generation (desired)
                    data_list.append({'text': target_text, 'text_pair': output_text})
        
        if data_list: # Only run pipeline if there's data to process
            batch_results = pipe(data_list)
            
            # Apply ROUGE filter based on original index in `rouge_scores_batch`
            filtered_batch_results = []
            valid_data_list_idx = 0
            for j in range(len(targets_batch)):
                # If original output/target was empty, we added a 'none' dummy result.
                # Skip to the next valid NLI result from `batch_results`.
                if not (str(outputs_batch[j]) and str(targets_batch[j])):
                    # This implies a 'none' was already appended in the previous step
                    # based on the empty string check.
                    continue
                
                # Apply the ROUGE filter for valid NLI results
                if rouge_scores_batch[valid_data_list_idx] < 0.1:
                    filtered_batch_results.append({'label': 'none', 'score': 0.0})
                else:
                    filtered_batch_results.append(batch_results[valid_data_list_idx])
                valid_data_list_idx += 1
            results.extend(filtered_batch_results)
        # Handle cases where data_list might be empty but original batch was not (due to empty strings)
        else:
            # If all inputs in the batch were empty, ensure `results` gets enough 'none' entries
            for _ in range(len(outputs_batch)):
                if not (str(outputs_batch[_]) and str(targets_batch[_])):
                     results.append({'label': 'none', 'score': 0.0}) # Append dummy for each skipped empty input
    
    entailment_labels = [result['label'] for result in results]
    return {'entailment_labels': entailment_labels}


# --- Main Processing Functions ---

def main_processing_script(input_file_path: str, output_file_path: str, nli_pipeline, sentence_transformer_model, eval_task: str = "retain", tofu: bool = True):
    """
    Loads, processes, and saves a single JSONL/JSON file with CS/ES metrics.
    Accepts pre-loaded models for efficiency.
    """
    processed_results = []
    # Use 'r' mode for reading input_file_path
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc=f"Processing {os.path.basename(input_file_path)}"):
            try:
                query_data = json.loads(line)
                processed_query_data = process_query_responses(query_data, nli_pipeline, sentence_transformer_model, eval_task, tofu)
                processed_results.append(processed_query_data)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line in {input_file_path}: {line.strip()} - Error: {e}")
                continue
            except Exception as e:
                print(f"Error processing query in {input_file_path}: {line.strip()} - Error: {e}")
                continue

    # Use 'w' mode for writing to output_file_path (overwrites existing file)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for data in processed_results:
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    
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
                if filename.endswith(".json") and "generation" in filename:
                    file_path = os.path.join(dirpath, filename)
                    print(f"Processing: {file_path}")
                    try:
                        main_processing_script(
                            input_file_path=file_path,
                            output_file_path=file_path, # Overwrite the file
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
        "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=0.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=1.0top_p=0.2",
        "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=1.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=1.0top_p=0.2",
#         "/users/2/jruan/pass-k/saves/eval/NPO+gen3/forget/temperature=1.0top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/NPO2/retain/temperature=1.0top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/ada/retain",
        "/users/2/jruan/pass-k/saves/eval/GradDiff/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/GradDiff/retain/temperature=0.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/GradDiff/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/GradDiff/retain/temperature=1.0top_p=0.2",
        "/users/2/jruan/pass-k/saves/eval/NPO/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/NPO/retain/temperature=0.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/NPO/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/NPO/retain/temperature=1.0top_p=0.2",
        "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=1.0top_p=0.2",
#         "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=0.2top_p=0.2",
        # "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=0.0top_p=0.0",
        "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=0.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=1.0top_p=0.2",
        "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/retain/temperature=0.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/BLUR-NPO/retain/temperature=1.0top_p=0.2",
        # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.2top_p=0.2",
        # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.2top_p=1.0",
        # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.8top_p=0.2",
        # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=0.8top_p=1.0", 
        # "/users/2/jruan/pass-k/saves/eval/NPO+ENT/retain/temperature=1.0top_p=1.0",
        "/users/2/jruan/pass-k/saves/eval/RMU/retain/temperature=0.2top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/RMU/retain/temperature=0.8top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/RMU/retain/temperature=1.0top_p=0.8",
        "/users/2/jruan/pass-k/saves/eval/RMU/retain/temperature=1.0top_p=0.2",
        # "/users/2/jruan/pass-k/saves/eval/Original/retain/temperature=0.0top_p=0.0",
        # "/users/2/jruan/pass-k/saves/eval/Retrain/retain/temperature=0.0top_p=0.0",
        # "/users/2/jruan/pass-k/saves/eval/SimNPO/retain/temperature=0.0top_p=0.0",

    ]
    # Run the main processing function
    process_all_generation_jsons(
        root_dirs=my_root_directories,
        eval_task="forget", # Set your eval_task
        tofu=True            # Set your tofu preference
    )

