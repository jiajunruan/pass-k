import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import List

# =====================================================
# ✅ Global OpenAI client (reuse for all calls)
# =====================================================
client = OpenAI()
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # good speed/price
BATCH_SIZE = 20  # how many comparisons per API call
MAX_WORKERS = 10  # number of concurrent batches

# =====================================================
# ⚙️ Batch LLM Judge
# =====================================================
from typing import List
import re

def LLM_judge_batch_multichoice(gen_outputs: List[str], ground_truths: List[str], prompt: str) -> List[int]:
    """
    Evaluates up to BATCH_SIZE candidate answers in one API call.
    Expects the model to return a single bracketed list like: [1,3,2,...]
    Returns a list of integers (one per candidate). Pads/truncates to len(gen_outputs).
    """

    # -------------------------
    # System prompt (rubric + precise output format)
    # -------------------------
    system_prompt = """You are a helpful AI assistant.
You will be given several numbered question–answer pairs.
For each pair, evaluate the candidate's quality according to this rubric:
• 3: Explicitly supports or analyzes the correct answer.
• 2: Provides relevant factual or inferential knowledge, but not directly supporting the correct answer.
• 1: Uninformative, repetitive, or unrelated to the question or correct answer.

IMPORTANT: Return exactly one integer score per item in order. Output must be a single JSON-like bracketed list (square brackets) containing the integers only, e.g.:
[1,3,2]

Do not include any other text, explanation, or punctuation outside the single bracketed list.
"""

    # -------------------------
    # Build the batched user content (numbered items)
    # -------------------------
    batched_text = "\n\n".join([
        f"Q{i+1}:\n[Question]\n{prompt}\n[Reference]\n{gt}\n[Candidate]\n{gen}"
        for i, (gen, gt) in enumerate(zip(gen_outputs, ground_truths))
    ])

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batched_text},
            ],
            temperature=0,
            max_output_tokens=200,
            # cache this system prompt template so identical rubric calls can be cached
            prompt_cache_key="LLM_SCORE_BRACKETED_LIST_V1",
        )

        # -------------------------
        # Robust text extraction (works across SDK shapes)
        # -------------------------
        raw_text = ""
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for c in item.content:
                        # some SDK shapes: c may have .text
                        if hasattr(c, "text"):
                            raw_text += c.text + "\n"
                        else:
                            # fallback: stringify content if it's a dict-like
                            try:
                                raw_text += str(c) + "\n"
                            except:
                                pass
        if not raw_text.strip():
            raw_text = (getattr(response, "output_text", "") or "").strip()

        # -------------------------
        # Parse numbers from a bracketed list if present, otherwise any integers found
        # Examples accepted:
        #   [1,3,2]
        #   1 3 2
        #   1\n3\n2
        # -------------------------
        parsed_nums: List[int] = []

        # Try to find the first bracketed expression
        bracket_match = re.search(r"\[([^\]]+)\]", raw_text)
        if bracket_match:
            inside = bracket_match.group(1)
            # extract integers from inside
            num_strs = re.findall(r"-?\d+", inside)
            parsed_nums = [int(n) for n in num_strs]
        else:
            # fallback: find all integers anywhere (in order)
            num_strs = re.findall(r"-?\d+", raw_text)
            parsed_nums = [int(n) for n in num_strs]

        # -------------------------
        # Pad/truncate to batch size and ensure ints
        # -------------------------
        batch_n = len(gen_outputs)
        # default fill value if model fails: 1 (or change to 0 if you prefer)
        DEFAULT_FILL = 1

        if not parsed_nums:
            # nothing parsed — fallback to default for each item
            results = [DEFAULT_FILL] * batch_n
        else:
            # take first batch_n numbers, pad if needed
            if len(parsed_nums) < batch_n:
                parsed_nums.extend([DEFAULT_FILL] * (batch_n - len(parsed_nums)))
            results = parsed_nums[:batch_n]
        print(results)
        return results

    except Exception as e:
        print(f"[LLM_judge_batch] Error: {e}")
        # On error, return default values
        return [1] * len(gen_outputs)

def LLM_judge_batch(gen_outputs: List[str], ground_truths: List[str], prompt: str) -> List[int]:
    """
    Batched LLM-as-judge. Model must output a bracketed list like: [1,0,1,...].
    YES -> 1, NO -> 0.
    Returns list[int] with size == len(gen_outputs).
    """

    system_prompt = """You are an evaluation model.
For each question, judge whether the candidate answer correctly conveys the essential information in the reference answer.
Output ONLY a single Python-style list of 0/1 integers, where:
- 1 = "YES", the candidate matches the reference.
- 0 = "NO", it does not match.
Example output: [1,0,1,1,0]
No explanations, no extra text.
"""

    # Build batch
    batched_text = "\n\n".join([
        f"Item {i+1}:\nQuestion: {prompt}\nReference: {gt}\nCandidate: {gen}"
        for i, (gen, gt) in enumerate(zip(gen_outputs, ground_truths))
    ])

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batched_text},
            ],
            temperature=0,
            max_output_tokens=100,
            prompt_cache_key="LLM_SCORE_LIST_V2",
        )

        # Extract plain text
        raw_text = getattr(response, "output_text", "").strip()
        if not raw_text:
            # SDK fallback
            raw_text = "".join(
                (c.text if hasattr(c, "text") else str(c))
                for o in getattr(response, "output", [])
                for c in getattr(o, "content", [])
            ).strip()

        # Parse list like [1,0,1]
        m = re.search(r"\[([^\]]+)\]", raw_text)
        numbers = []
        if m:
            numbers = [int(x) for x in re.findall(r"[01]", m.group(1))]

        # If parsing failed, fallback to zeros
        batch_n = len(gen_outputs)
        if not numbers:
            return [0] * batch_n

        # Pad/truncate to batch size
        if len(numbers) < batch_n:
            numbers.extend([0] * (batch_n - len(numbers)))
        print(numbers[:batch_n])
        return numbers[:batch_n]

    except Exception as e:
        print("[LLM_judge_batch] ERROR:", e)
        return [0] * len(gen_outputs)


# =====================================================
# ⚙️ Wrapper with parallel batches
# =====================================================
def LLM_judge(gen_outputs: List[str], ground_truths: List[str], prompt: str) -> List[int]:
    """
    Parallel batched LLM evaluation for large numbers of generations.
    """
    batches = [
        (gen_outputs[i:i+BATCH_SIZE], ground_truths[i:i+BATCH_SIZE])
        for i in range(0, len(gen_outputs), BATCH_SIZE)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(LLM_judge_batch, g, gt, prompt): (g, gt) for g, gt in batches}
        for f in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            results.extend(f.result())

    return results

# =====================================================
# ⚙️ Query Processing
# =====================================================
def process_query_responses(query_data):
    """
    Processes multiple generated responses for a single query,
    adds LLM_judge results.
    """
    ground_truth = query_data.get("ground_truth", "")
    responses = query_data.get("responses", [])
    prompt = query_data.get("prompt", "")

    prefix = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 10 Apr 2025\n\nYou are a helpful assistant.user\n\n"
    if prompt.startswith(prefix):
        prompt = prompt[len(prefix):]

    gen_outputs = [res.get("response", "") for res in responses]
    ground_truths = [ground_truth] * len(gen_outputs)

    judge_results = LLM_judge(gen_outputs, ground_truths, prompt)

    for i, res in enumerate(responses):
        res["LLM_judge"] = judge_results[i] if i < len(judge_results) else 0

    query_data["responses"] = responses
    return query_data

# =====================================================
# ⚙️ File Processing
# =====================================================
def main_processing_script(input_file_path: str, output_file_path: str):
    """
    Loads, evaluates, and writes JSONL file with LLM_judge results.
    """
    processed_results = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    for line in tqdm(lines, desc=f"Processing {os.path.basename(input_file_path)}"):
        try:
            query_data = json.loads(line)
            processed_query_data = process_query_responses(query_data)
            processed_results.append(processed_query_data)
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line: {e}")
        except Exception as e:
            print(f"Error processing query: {e}")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for data in processed_results:
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# =====================================================
# ⚙️ Directory Traversal
# =====================================================
def process_all_generation_jsons(root_dirs: list):
    """
    Traverse root dirs, find generation files, evaluate them.
    """
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"⚠️ Warning: directory '{root_dir}' not found.")
            continue

        print(f"\n📂 Searching in: {root_dir}")
        # for dirpath, _, filenames in os.walk(root_dir):
        #     for filename in filenames:
        #         if filename.endswith(".json") and "generations_n2" in filename:
        #             file_path = os.path.join(dirpath, filename)
        #             print(f"▶️ Processing file: {file_path}")

        #             try:
        #                 main_processing_script(file_path, file_path)
        #                 print(f"✅ Successfully updated: {file_path}")
        #             except Exception as e:
        #                 print(f"❌ Error processing {file_path}: {e}")
        # path = os.path.join(root_dir, "forget", "temperature=0.2top_p=0.2", "generations_n200.json")
        # if os.path.isfile(path):
        #     print(f"▶️ Processing file: {path}")

        #     try:
        #         main_processing_script(path, path)
        #         print(f"✅ Successfully updated: {path}")
        #     except Exception as e:
        #         print(f"❌ Error processing {path}: {e}")
        
        path = os.path.join(root_dir, "forget", "temperature=0.8top_p=1.0", "generations_n200.json")
        if os.path.isfile(path):
            print(f"▶️ Processing file: {path}")

            try:
                main_processing_script(path, path)
                print(f"✅ Successfully updated: {path}")
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")        

        print("-" * 50)
    print("✅ All directories processed.")

# =====================================================
# 🏁 Entry Point
# =====================================================
if __name__ == "__main__":
    my_root_directories = [
        # "/users/2/jruan/pass-k/saves/eval/BLUR-NPO",
        # "/users/2/jruan/pass-k/saves/eval/GradDiff",
        # "/users/2/jruan/pass-k/saves/eval/NPO",
        # "/users/2/jruan/pass-k/saves/eval/NPO+ENT",
        # "/users/2/jruan/pass-k/saves/eval/RMU",
        # "/users/2/jruan/pass-k/saves/eval/Original",
        # "/users/2/jruan/pass-k/saves/eval/Retrain",
        # "/users/2/jruan/pass-k/saves/eval/SimNPO",
        "/users/2/jruan/pass-k/saves/eval/LoUK",
    ]
    import time
    start_time =  time.time()
    process_all_generation_jsons(my_root_directories)
    end_time = time.time()
    print("duration", end_time-start_time)
