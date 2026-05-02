import subprocess
import os
import time

# 1. Define the specific models and paths
model_configs = {
    "RULE_NPO": "/projects/standard/mhong/shared/jiajunr/open-unlearning/saves/unlearn/NPO_fix/checkpoint-2",
    "RULE_GradDiff": "/projects/standard/mhong/shared/jiajunr/open-unlearning/saves/unlearn/GradDiff_fix/checkpoint-2",
    "NPO": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.5_alpha1_epoch10",
    "GradDiff": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
    "Original": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    "Retrain": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
}

# 2. Configuration for single GPU
gpu_id = 0  # Your only available GPU

print(f"🚀 Starting SEQUENTIAL evaluation for {len(model_configs)} models on GPU {gpu_id}...")
print("Each model will wait for the previous one to finish to prevent OOM.\n")

for idx, (method_label, model_path) in enumerate(model_configs.items()):
    log_file = f"eval_{method_label}.log"
    
    # Construct the command
    command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python src/eval.py "
        f"--config-name=eval.yaml "
        f"experiment=eval/tofu/default "
        f"model=Llama-3.2-1B-Instruct "
        f"model.model_args.pretrained_model_name_or_path={model_path} "
        f"retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json "
        f"task_name={method_label} "
        f"> {log_file} 2>&1"
    )

    print(f"📡 [{idx+1}/{len(model_configs)}] Running: {method_label}...")
    
    start_time = time.time()
    
    # Using subprocess.run with check=True makes the script wait here
    try:
        subprocess.run(command, shell=True, check=True)
        duration = (time.time() - start_time) / 60
        print(f"✅ [DONE] {method_label} (Time taken: {duration:.2f} mins)")
    except subprocess.CalledProcessError as e:
        print(f"❌ [FAIL] {method_label} failed with error. Check {log_file} for details.")
    
    # Optional: Brief sleep to allow VRAM to fully clear between runs
    time.sleep(5)

print("\n🎉 All sequential evaluations are complete. Results are saved in the .log files.")