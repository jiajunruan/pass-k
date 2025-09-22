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

def main(input_filename,output_filename):
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
                    print(f"警告：第 {i+1} 行数据中缺少 'responses' 键，跳过。")
                    continue
                
                responses = data['responses']
                n = len(responses)
                
                # 计算通过的response数量 (c)
                c = sum(1 for response in responses if response.get('ES') == 1)
                
                # 计算并累加每个k值下的pass@k
                for k in k_values:
                    pass_at_k_score = calculate_pass_at_k(n, c, k)
                    sum_pass_at_k[k] += pass_at_k_score
                    
        # 计算平均值
        average_pass_at_k = {k: score / total_entries for k, score in sum_pass_at_k.items()}
        
        # 将结果保存到JSON文件
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(average_pass_at_k, outfile, indent=4)
            
        print(f"Pass@k 平均值计算完成，结果已保存到 {output_filename}")
        
    except FileNotFoundError:
        print(f"错误：文件 '{input_filename}' 未找到。请确保文件在正确的目录下。")
    except json.JSONDecodeError:
        print(f"错误：JSON文件格式不正确。请检查文件 '{input_filename}'。")
    except Exception as e:
        print(f"发生了一个错误: {e}")

if __name__ == "__main__":
    input_filename = 'saves/eval/NPO2/forget/temperature=1.0top_p=1.0/generations_n320.json'
    output_filename = 'saves/eval/NPO2/forget/temperature=1.0top_p=1.0/rougeL_summary.json'
    main(input_filename,output_filename)