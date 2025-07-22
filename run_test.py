import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 本地模型路径
local_model_path = r"/root/demo/model/Qwen2.5-7B-Instruct"

def load_test_data():
    """加载测试数据"""
    test_data = []
    with open('data/test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    return test_data

def get_model_and_tokenizer():
    """加载模型和分词器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'Qwen2.5-7B-Instruct-QLoRA/checkpoint-2500'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # 获取AI token ID - 更可靠的方式
    ai_token_id = tokenizer("AI", add_special_tokens=False).input_ids[0]
    print(f"AI token ID: {ai_token_id}")
    
    return model, tokenizer, device, ai_token_id

def predict(model, tokenizer, device, test_data, ai_token_id):
    """对测试数据进行预测（使用概率阈值）"""
    results = []
    
    # 初始化概率区间计数器
    count_00_05 = 0  # [0.0, 0.5)
    count_05_06 = 0  # [0.5, 0.6)
    count_06_07 = 0  # [0.6, 0.7)
    count_07_08 = 0  # [0.7, 0.8)
    count_08_09 = 0  # [0.8, 0.9)
    count_09_10 = 0  # [0.9, 1.0]
    
    for item in tqdm(test_data, desc="预测中"):
        # 构建输入消息
        message = [
            {'role': 'system', 'content': 'You are a professional AI-generated text detection assistant. Your task is to detect the input text, determine whether it is AI-generated, and output one of the two words "AI-generated" or "manually written". Do not output any other additional information or explanations.'},
            {'role': 'user', 'content': item['text']}
        ]
        
        # 对输入进行编码
        inputs = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt'
        ).to(device)
        
        # 生成预测并获取概率
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1,  # 只生成一个token
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,  # 获取分数
                return_dict_in_generate=True  # 返回详细输出
            )
        
        # 获取第一个新token的概率分布
        first_token_logits = outputs.scores[0][0]  # 第一个生成token的logits
        probs = torch.softmax(first_token_logits, dim=-1)
        
        # 计算AI token的概率
        ai_prob = probs[ai_token_id].item()
        
        # 根据阈值判断（0.6阈值）
        if ai_prob >= 0.8:
            pred_label = 1
        else:
            pred_label = 0
        
        # 统计不同概率区间的样本数量
        if ai_prob < 0.5:
            count_00_05 += 1
        elif 0.5 <= ai_prob < 0.6:
            count_05_06 += 1
        elif 0.6 <= ai_prob < 0.7:
            count_06_07 += 1
        elif 0.7 <= ai_prob < 0.8:
            count_07_08 += 1
        elif 0.8 <= ai_prob < 0.9:
            count_08_09 += 1
        else:  # ai_prob >= 0.9
            count_09_10 += 1
        
        # 打印前5个样本的详细信息
        if len(results) < 5:
            print(f"样本 #{len(results)+1}")
            print(f"Text: {item['text'][:100]}...")
            print(f"AI概率: {ai_prob:.4f}, Pred: {pred_label}")
            print("-" * 50)
        
        results.append(pred_label)
    
    # 打印概率分布统计
    print("\n详细概率分布统计:")
    print(f"[0.0-0.5) 区间样本数: {count_00_05} ({count_00_05/len(test_data)*100:.2f}%)")
    print(f"[0.5-0.6) 区间样本数: {count_05_06} ({count_05_06/len(test_data)*100:.2f}%)")
    print(f"[0.6-0.7) 区间样本数: {count_06_07} ({count_06_07/len(test_data)*100:.2f}%)")
    print(f"[0.7-0.8) 区间样本数: {count_07_08} ({count_07_08/len(test_data)*100:.2f}%)")
    print(f"[0.8-0.9) 区间样本数: {count_08_09} ({count_08_09/len(test_data)*100:.2f}%)")
    print(f"[0.9-1.0] 区间样本数: {count_09_10} ({count_09_10/len(test_data)*100:.2f}%)")
    
    # 打印预测结果统计
    ai_count = sum(results)
    human_count = len(results) - ai_count
    print(f"\n预测结果统计:")
    print(f"AI生成 (1): {ai_count} ({ai_count/len(test_data)*100:.2f}%)")
    print(f"人工写作 (0): {human_count} ({human_count/len(test_data)*100:.2f}%)")
    
    # 打印置信度汇总
    print(f"\n置信度汇总:")
    print(f"高置信度AI (≥0.9): {count_09_10} ({count_09_10/len(test_data)*100:.2f}%)")
    print(f"中等置信度AI (0.7-0.9): {count_07_08 + count_08_09} ({(count_07_08 + count_08_09)/len(test_data)*100:.2f}%)")
    print(f"低置信度AI (0.5-0.7): {count_05_06 + count_06_07} ({(count_05_06 + count_06_07)/len(test_data)*100:.2f}%)")
    print(f"人工写作 (<0.5): {count_00_05} ({count_00_05/len(test_data)*100:.2f}%)")
    
    return results

def save_results(results):
    """保存预测结果"""
    with open('submit.txt', 'w', encoding='utf-8') as f:
        for label in results:
            f.write(str(label) + '\n')
    print(f"预测结果已保存到 submit.txt，共 {len(results)} 条数据")

def main():
    # 加载测试数据
    print("正在加载测试数据...")
    test_data = load_test_data()
    print(f"已加载 {len(test_data)} 条测试数据")
    
    # 加载模型和分词器
    print("正在加载模型和分词器...")
    model, tokenizer, device, ai_token_id = get_model_and_tokenizer()
    
    # 进行预测
    print("开始预测...")
    results = predict(model, tokenizer, device, test_data, ai_token_id)
    
    # 保存结果
    save_results(results)

if __name__ == '__main__':
    main()