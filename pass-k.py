# A dictionary mapping method names to their model paths or identifiers.
# model_names = {
#     "Original":"open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
#     "Retrain":"open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
#     "GradDiff":"open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
#     "NPO":"open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.5_alpha1_epoch10",
#     # "NPO+gen3":"/users/2/jruan/open-unlearning/saves/unlearn/SAMPLE_UNLEARN1"
#     # "NPO+ENT":"saves/unlearn/NPO_ENT",
#     "SimNPO":"open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d1_g0.125_ep10",
#     "BLUR-NPO":"HadiUMN/tofu_Llama-3.2-1B-Instruct_forget10_BLURNPO",
#     "RMU":"open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer15_scoeff100_epoch10",
#     # "GradAscent":"saves/unlearn/GA",
#     "NPO+ENT":"/users/2/jruan/open-unlearning/saves/unlearn/NPO+ENT1",
# }
model_names = {"NPO+ENT+TMP":"/users/2/jruan/open-unlearning/saves/unlearn/NPO+ENT1"}
# ### Sequential Execution
# import subprocess
# import os
# import time
# # The command template for launching the evaluation script.
# # The '&' at the end has been removed to ensure sequential execution.
# command_template = (
#     "nohup python src/eval.py "
#     "--config-name=eval.yaml "
#     "experiment=eval/tofu/default "
#     "model=Llama-3.2-1B-Instruct "
#     "model.model_args.pretrained_model_name_or_path={model_path} "
#     "retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json "
#     "task_name={method} "
#     "> {log_file} 2>&1"
# )

# # Iterate over each method and model pair to launch a separate process.
# for method, model_path in model_names.items():
#     # Define the log file name based on the method name.
#     log_file = f"{method}.log"
    
#     # Construct the full command by substituting the placeholders.
#     command = command_template.format(
#         method=method,
#         model_path=model_path,
#         log_file=log_file
#     )
    
#     print(f"⏳ Running evaluation for method: '{method}' (waiting for completion)...")
#     print(f"   Command: {command}")
    
#     # Execute the command in a new shell process.
#     # The script will now wait for this command to complete before starting the next one.
#     try:
#         subprocess.run(command, shell=True, check=True)
#         print(f"✅ Process for '{method}' finished successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Error running process for '{method}': {e}")
#     print("-" * 40)

# print("All evaluation processes have been completed sequentially.")


## Parallel Execution with GPU Distribution
# List to keep track of subprocess.Popen objects and method names
import subprocess
import os
import time

# List of available GPU IDs you want to use
available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # adjust based on your machine

processes = []

for idx, (method, model_path) in enumerate(model_names.items()):
    gpu_id = available_gpus[idx % len(available_gpus)]  # round-robin allocation
    log_file = f"{method}_f2.log"
    
    # Corrected command without nohup and &
    command = (
        f"CUDA_VISIBLE_DEVICES=0 "
        "python src/eval.py "
        "--config-name=eval.yaml "
        "experiment=eval/tofu/default "
        "model=Llama-3.2-1B-Instruct "
        f"model.model_args.pretrained_model_name_or_path={model_path} "
        "retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json "
        f"task_name={method} "
        f"> {log_file} 2>&1"
    )

    print(f"🚀 Launching '{method}' on GPU {gpu_id}.")
    print(f"    Command: {command}")

    # Launch the process
    # Use Popen without shell=True for better security and process control
    # Or keep shell=True if the command relies on shell features like redirection
    process = subprocess.Popen(command, shell=True) 
    processes.append((process, method, gpu_id))
    time.sleep(1)  # slight stagger to avoid overload

print("\n✅ All processes launched in parallel, distributed across GPUs.\n")

# # Wait for completion
# for process, method, gpu_id in processes:
#     retcode = process.wait()
#     if retcode == 0:
#         print(f"✅ '{method}' on GPU {gpu_id} finished successfully.")
#     else:
#         print(f"❌ '{method}' on GPU {gpu_id} exited with return code {retcode}.")

# print("\n🎉 All evaluations completed.")
# while True:
#     time.sleep(60)  # Keep the script running to allow nohup processes to continue
#     print("Script is still running to keep nohup processes alive...")