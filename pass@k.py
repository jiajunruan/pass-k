import os
import json
import math

def calculate_pass_at_k(n, c, k):
    """
    根据公式计算pass@k的值。
    n: 总样本数 (这里是320)
    c: 通过测试的样本数
    k: 考虑的样本数
    """
    if k > n:
        return 0.0
    
    # 组合数 C(n, k)
    total_combinations = math.comb(n, k)
    
    # 组合数 C(n-c, k)
    if n - c < k:
        fail_combinations = 0
    else:
        fail_combinations = math.comb(n - c, k)
        
    return 1.0 - (fail_combinations / total_combinations)

def process_file(input_filename, output_filename):
    # 设定文件名和k值
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    total_entries = 400
    
    # 初始化用于累加pass@k分数的字典
    sum_pass_at_k = {k: 0.0 for k in k_values}
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_entries:
                    break
                
                data = json.loads(line)
                
                # 检查数据结构
                if 'responses' not in data:
                    continue
                
                responses = data['responses']
                n = len(responses)
                
                # 计算通过的response数量 (c)
                c = sum(1 for response in responses if response.get('ES') == 1)
                # for response in responses:
                #     if response.get('ES') == 1:
                #         print(response.get("response"))
                # 计算并累加每个k值下的pass@k
                for k in k_values:
                    pass_at_k_score = calculate_pass_at_k(n, c, k)
                    sum_pass_at_k[k] += pass_at_k_score
                    
        # 计算平均值
        average_pass_at_k = {k: score / total_entries for k, score in sum_pass_at_k.items()}
        
        # 将结果保存到JSON文件
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(average_pass_at_k, outfile, indent=4)
            
        print(f"写入: {output_filename}")
        
    except Exception as e:
        print(f"处理 {input_filename} 时出错: {e}")

if __name__ == "__main__":
    # 你的主目录列表
    folder_list = [
        # "LLMjudgeresult/eval/BLUR-NPO/forget",
        # "LLMjudgeresult/eval/NPO/forget",
        # "LLMjudgeresult/eval/Original/forget",
        # "LLMjudgeresult/eval/Retrain/forget",
        # "LLMjudgeresult/eval/RMU/forget",
        # "LLMjudgeresult/eval/SimNPO/forget",
        # "LLMjudgeresult/eval/NPO+ENT/forget",
        # "LLMjudgeresult/eval/GradDiff/forget",
        "saves/eval/LoUK/forget"
        # ... 其他目录 ...
    ]
    for main_dir in folder_list:
        for subdir in os.listdir(main_dir):
            subdir_path = os.path.join(main_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            file_320 = os.path.join(subdir_path, "generations_n320.json")
            file_200 = os.path.join(subdir_path, "generations_n200.json")
            file_128 = os.path.join(subdir_path, "generations_n128.json")
            if os.path.exists(file_320):
                input_file = file_320
            elif os.path.exists(file_200):
                input_file = file_200
            else:
                input_file = file_128
                
            output_file = os.path.join(subdir_path, "rougeL_summary.json")
            process_file(input_file, output_file)
#     for main_dir in folder_list:
# #         file_path = os.path.join(main_dir, "temperature=0.2top_p=0.2", "generations_n200.json")
# #         output_file = os.path.join(main_dir, "temperature=0.2top_p=0.2", "rougeL_summary.json")
# #         if os.path.exists(file_path):
# #             process_file(file_path, output_file)
#         file_path = os.path.join(main_dir, "temperature=1.0top_p=1.0", "generations_n200.json")
#         output_file = os.path.join(main_dir, "temperature=1.0top_p=1.0", "rougeL_summary.json")
#         if os.path.exists(file_path):
#             process_file(file_path, output_file)        