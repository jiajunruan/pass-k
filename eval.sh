#!/bin/bash

# Define the base command
BASE_CMD="python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default model=Llama-3.2-1B-Instruct retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"

# Define paired models and tasks
# Format: "model_path:task_name"
MODEL_TASK_PAIRS=(
    # "/root/.cache/huggingface/hub/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots/88e31200b97e4c0c04ae0d2f0b591f427046d192:Original"
    # "/root/.cache/huggingface/hub/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_retain90/snapshots/7114300c0049527a71833f5683965c358ad9dcbf:retain"
    # "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10:GradDiff"
    # "/root/autodl-tmp/open-unlearning/saves/unlearn/SAMPLE_UNLEARN:NPO+Entropy"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d1_g0.125_ep10:SimNPO"
    
    # Add more model:task pairs as needed
)

# Loop through all model-task pairs
for pair in "${MODEL_TASK_PAIRS[@]}"; do
    # Split the pair into model and task
    IFS=':' read -r model_path task <<< "$pair"
    
    echo "Running evaluation for model: $model_path with task: $task"
    
    # Construct the full command
    FULL_CMD="$BASE_CMD model.model_args.pretrained_model_name_or_path=$model_path task_name=$task"
    
    # Print the command (for debugging)
    echo "Command: $FULL_CMD"
    
    # Execute the command
    eval $FULL_CMD
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "Error encountered while running: $FULL_CMD"
        # You can choose to exit here or continue with the next iteration
        # exit 1
    fi
    
    echo "--------------------------------------------------"
done

echo "All evaluations completed!"