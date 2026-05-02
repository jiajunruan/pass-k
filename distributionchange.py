# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import random
# import numpy as np

# # ======================
# # Config
# # ======================
# MODEL_PATH = "HadiUMN/NPO_MUSE-News"

# QA_PAIRS = [
#     {
#         "question": "In what year did Willie Nelson start out as a songwriter?",
#         "answer": "1960s"
#     },
#     {
#         "question": "Who is the photographer of the official Coronation photographs released by Buckingham Palace?",
#         "answer": "Hugo Burnand"
#     },
#     {
#         "question": "How much does it cost for a practical driving test in Northern Ireland?",
#         "answer": "£45.50"
#     },
#     {
#         "question": "How many years did the buyout of Tower extend its life according to the excerpt?",
#         "answer": "13 years"
#     },
#     # 你可以继续加
# ]

# NUM_SAMPLES = 200
# TEMPERATURE = 1.0
# TOP_P = 1.0
# TOP_K_LOG = 10

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.float16

# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)

# # ======================
# # Load model
# # ======================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=DTYPE,
#     device_map="auto"
# )
# model.eval()

# # pad_token 修复
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # ======================
# # Main loop
# # ======================
# for qa_idx, qa in enumerate(QA_PAIRS):
#     QUESTION = qa["question"]
#     GROUNDTRUTH = qa["answer"]

#     print("\n" + "=" * 100)
#     print(f"[QA {qa_idx}] QUESTION: {QUESTION}")
#     print(f"[QA {qa_idx}] GROUNDTRUTH: {GROUNDTRUTH}")
#     print("=" * 100)

#     prompt = QUESTION.strip() + "\nAnswer:"
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True)
#     input_ids = inputs["input_ids"].to(DEVICE)
#     attention_mask = inputs["attention_mask"].to(DEVICE)

#     gt_ids = tokenizer(GROUNDTRUTH, add_special_tokens=False).input_ids

#     hit_count = 0

#     for sample_idx in range(NUM_SAMPLES):
#         with torch.no_grad():
#             output = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 do_sample=True,
#                 temperature=TEMPERATURE,
#                 top_p=TOP_P,
#                 max_new_tokens=32,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 pad_token_id=tokenizer.eos_token_id,
#             )

#         gen_ids = output.sequences[0][input_ids.shape[1]:].tolist()
#         gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

#         # ===== 总是打印 generation =====
#         print("\n" + "-" * 80)
#         print(f"[Sample {sample_idx}] Generated text:")
#         print(repr(gen_text))
#         print("-" * 80)

#         # ===== 是否命中 GT（文本级）=====
#         if GROUNDTRUTH not in gen_text:
#             continue

#         hit_count += 1
#         print(f"[HIT #{hit_count}] Groundtruth substring FOUND")

#         # ===== 找 GT token 在 generation 中的位置 =====
#         for start_pos in range(len(gen_ids)):
#             if gen_ids[start_pos:start_pos + len(gt_ids)] == gt_ids:
#                 print(f"GT token sequence matched at position {start_pos}")
#                 break
#         else:
#             print("GT text matched but token alignment failed.")
#             continue

#         # ===== 分析从 GT 第一个 token 开始的 distribution =====
#         scores = output.scores  # 每一步一个 logits

#         for t in range(start_pos, start_pos + len(gt_ids)):
#             step_logits = scores[t][0]
#             probs = F.softmax(step_logits, dim=-1)

#             gt_token_id = gt_ids[t - start_pos]
#             gt_token_str = tokenizer.decode([gt_token_id])

#             gt_prob = probs[gt_token_id].item()
#             gt_logit = step_logits[gt_token_id].item()

#             topk_probs, topk_ids = torch.topk(probs, TOP_K_LOG)
#             topk_logits = step_logits[topk_ids]

#             print("\n" + "*" * 60)
#             print(f"Step {t} (GT token='{gt_token_str}')")
#             print(f"GT prob={gt_prob:.6f}, logit={gt_logit:.6f}")
#             print(f"Top-{TOP_K_LOG} candidates:")

#             for rank in range(TOP_K_LOG):
#                 tid = topk_ids[rank].item()
#                 tok = tokenizer.decode([tid])
#                 logit = topk_logits[rank].item()
#                 prob = topk_probs[rank].item()
#                 marker = " <-- GT" if tid == gt_token_id else ""
#                 print(
#                     f"[{rank+1:02d}] '{tok}' logit={logit:.4f} prob={prob:.4f}{marker}"
#                 )

#         print("\n" + "=" * 80)
#         print("End of GT-hit analysis")
#         print("=" * 80)

#     print(f"\nTotal GT hits for this QA: {hit_count}/{NUM_SAMPLES}")


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np

# ======================
# Config
# ======================
MODEL_PATH = "HadiUMN/NPO_MUSE-News"

QA_PAIRS = [
    {
        "question": "In what year did Willie Nelson start out as a songwriter?",
        "answer": "1960s"
    },
    {
        "question": "Who is the photographer of the official Coronation photographs released by Buckingham Palace?",
        "answer": "Hugo Burnand"
    },
    {
        "question": "How much does it cost for a practical driving test in Northern Ireland?",
        "answer": "£45.50"
    },
    {
        "question": "How many years did the buyout of Tower extend its life according to the excerpt?",
        "answer": "13 years"
    },
]

NUM_SAMPLES = 500
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ======================
# Load model
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================
# Main loop
# ======================
for qa_idx, qa in enumerate(QA_PAIRS):
    QUESTION = qa["question"]
    GROUNDTRUTH = qa["answer"]

    print("\n" + "=" * 100)
    print(f"[QA {qa_idx}] QUESTION: {QUESTION}")
    print(f"[QA {qa_idx}] GROUNDTRUTH: {GROUNDTRUTH}")
    print("=" * 100)

    prompt = QUESTION.strip() + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    gt_ids = tokenizer(GROUNDTRUTH, add_special_tokens=False).input_ids
    gt_len = len(gt_ids)

    # ===== aggregation buffers =====
    cond_prob_sum = torch.zeros(gt_len, device=DEVICE)
    cond_prob_count = torch.zeros(gt_len, device=DEVICE)

    hit_count = 0

    # ======================
    # Sampling
    # ======================
    for sample_idx in range(NUM_SAMPLES):
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output.sequences[0][input_ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print("\n" + "-" * 80)
        print(f"[Sample {sample_idx}] Generated:")
        print(repr(gen_text))
        print("-" * 80)

        scores = output.scores  # list of [1, vocab] logits

        # ===== exact GT token match =====
        match_pos = None
        for start_pos in range(len(gen_ids) - gt_len + 1):
            if gen_ids[start_pos:start_pos + gt_len] == gt_ids:
                match_pos = start_pos
                break

        if match_pos is None:
            continue

        hit_count += 1
        print(f"[HIT #{hit_count}] Exact GT token match at position {match_pos}")

        # ===== collect conditional probabilities =====
        for j in range(gt_len):
            step = match_pos + j
            step_logits = scores[step][0]
            probs = F.softmax(step_logits, dim=-1)

            gt_token_id = gt_ids[j]
            gt_prob = probs[gt_token_id].item()

            cond_prob_sum[j] += gt_prob
            cond_prob_count[j] += 1

    # ======================
    # Report averages
    # ======================
    print("\n" + "#" * 100)
    print("AVERAGED CONDITIONAL PROBABILITIES (exact GT matches only)")
    print("#" * 100)

    for j in range(gt_len):
        tok = tokenizer.decode([gt_ids[j]])
        if cond_prob_count[j] > 0:
            avg_prob = (cond_prob_sum[j] / cond_prob_count[j]).item()
            print(
                f"Token {j+1}/{gt_len} | '{tok}' | "
                f"Avg p(token | prev + context) = {avg_prob:.6f} "
                f"(N={int(cond_prob_count[j].item())})"
            )
        else:
            print(
                f"Token {j+1}/{gt_len} | '{tok}' | "
                f"No matched samples"
            )

    print(f"\nTotal exact GT hits: {hit_count}/{NUM_SAMPLES}")
