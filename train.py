import torch
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import random
import nltk
from nltk.corpus import wordnet
from typing import List
import bitsandbytes as bnb

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # 更小的分块大小减少内存碎片

# 下载NLTK资源
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# 本地模型路径
local_model_path = r"/root/demo/model/Qwen2.5-7B-Instruct"

# ================== 数据增强工具函数 ==================
def get_synonyms(word: str) -> List[str]:
    """获取单词的同义词列表"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            if synonym != word and len(synonym.split()) == 1:
                synonyms.add(synonym)
    return list(synonyms) if synonyms else [word]

def synonym_replacement(text: str, n: int = 2) -> str:
    """同义词替换增强"""
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return " ".join(new_words)

def random_deletion(text: str, p: float = 0.2) -> str:
    """随机删除单词增强"""
    words = text.split()
    if len(words) == 1:
        return text
        
    remaining = [word for word in words if random.random() > p]
    if len(remaining) == 0:
        return random.choice(words)
    return " ".join(remaining)

def random_swap(text: str, n: int = 3) -> str:
    """随机交换单词位置增强"""
    words = text.split()
    n = min(n, len(words) // 2)  # 确保不会交换太多次
    
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return " ".join(words)

def augment_text(text: str, label: int) -> List[str]:
    """应用多种增强策略生成增强样本"""
    augmented_texts = [text]  # 包含原始文本
    
    # 只对人工撰写的文本进行增强
    # if label == 0:  # 0表示人工撰写
        # 同义词替换
        # augmented_texts.append(synonym_replacement(text, n=random.randint(1, 2)))  # 减少增强次数
        
        # 随机删除
        # augmented_texts.append(random_deletion(text, p=random.uniform(0.05, 0.15)))  # 降低删除概率
        
        # 随机交换
        # augmented_texts.append(random_swap(text, n=random.randint(1, 2)))  # 减少交换次数
        
        # 组合增强：替换+删除（降低增强强度）
        # combined = synonym_replacement(text, n=1)
        # combined = random_deletion(combined, p=0.1)
        # augmented_texts.append(combined)
    
    return list(set(augmented_texts))  # 去重

# ================== 数据集处理函数 ==================
def get_dataset(tokenizer):
    comm_data = pickle.load(open(f'data/01-训练集.pkl', 'rb'))
    result_data = []
    
    for data in comm_data:
        # 应用数据增强
        augmented_texts = augment_text(data['text'], data['label'])
        
        for text in augmented_texts:
            message = [
                {
                    'role': 'system', 
                    'content': 'You are a professional AI-generated text detection assistant. '
                               'Your task is to detect the input text, determine whether it is AI-generated, '
                               'and output one of the two words "AI-generated" or "manually written". '
                               'Do not output any other additional information or explanations.'
                },
                {'role': 'user', 'content': text},
                {
                    'role': 'assistant', 
                    'content': 'AI-generated' if data['label'] == 1 else 'manually written'
                }
            ]
            inputs = tokenizer.apply_chat_template(
                message, 
                add_generation_prompt=False, 
                tokenize=True,
                padding=False
            )
            result_data.append(inputs)
    
    print(f"原始样本数: {len(comm_data)}, 增强后总样本数: {len(result_data)}")
    return result_data

def demo():
    # 获取分布式训练中的本地排名
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # 打印GPU信息（仅主进程）
    if local_rank == 0:
        print(f"可用的GPU数量: {torch.cuda.device_count()}")
        print(f"当前使用的GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
        print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f} MB / {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    # 检查模型文件是否存在
    if not os.path.exists(local_model_path) and local_rank == 0:
        raise FileNotFoundError(f"模型文件不存在: {local_model_path}")
    
    try:
        if local_rank == 0:
            print(f"开始加载模型，路径: {local_model_path}")
        
        # 配置量化参数 - 使用8位量化
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  # 改为8位量化
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        # 设备映射
        device_map = {"": f"cuda:{local_rank}"}
        
        # 加载基础模型（使用4位量化）
        base_model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            local_files_only=True,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            use_cache=False,
            low_cpu_mem_usage=True
        )
        
        # 准备模型进行k-bit训练
        base_model = prepare_model_for_kbit_training(base_model)
        
        if local_rank == 0:
            print("基础模型加载成功，开始配置LoRA...")
        
        # 配置LoRA参数 - 更小的配置节省内存
        lora_config = LoraConfig(
            r=16,  # 减少rank值
            lora_alpha=32,  # 减少alpha值
            target_modules=["q_proj", "v_proj"],  # 只针对query和value层
            lora_dropout=0.05,  # 更小的dropout
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 将模型转换为PEFT模型
        model = get_peft_model(base_model, lora_config)
        
        if local_rank == 0:
            model.print_trainable_parameters()
            # 打印内存使用情况
            print(f"模型加载后内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        if local_rank == 0:
            print("开始加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True,
            trust_remote_code=True,
            padding_side='left'
        )
        tokenizer.pad_token = tokenizer.eos_token
        if local_rank == 0:
            print("Tokenizer加载成功")
        
    except Exception as e:
        print(f"[Rank {local_rank}] 加载模型时出错: {str(e)}")
        import traceback
        print(f"[Rank {local_rank}] 详细错误信息: {traceback.format_exc()}")
        return

    # 训练参数设置 - 减少内存消耗
    arguments = TrainingArguments(
        output_dir='Qwen2.5-7B-Instruct-QLoRA',
        per_device_train_batch_size=4,  # 减少批次大小
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # 增加梯度累积步数补偿小批次
        optim='paged_adamw_8bit',  # 使用分页优化器减少内存峰值
        num_train_epochs=3,
        learning_rate=1e-4,  # 降低学习率
        weight_decay=0.001,  # 降低权重衰减
        warmup_ratio=0.05,  # 减少预热比例
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        local_rank=local_rank,
        eval_strategy='no',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=10,  # 减少日志频率
        logging_first_step=True,
        save_total_limit=3,  # 只保留一个检查点
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=True,  # 启用以节省内存
        dataloader_num_workers=2,  # 减少数据加载工作线程
        dataloader_pin_memory=False,  # 禁用内存钉扎
        gradient_checkpointing=True,  # 确保启用
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_timeout=3600,  # 增加超时时间
        ddp_bucket_cap_mb=10,  # 减小通信桶大小
    )

    if local_rank == 0:
        print("开始准备训练数据...")
    train_data = get_dataset(tokenizer)
    if local_rank == 0:
        print(f"训练数据准备完成，样本数量: {len(train_data)}")
    
    # 使用DataCollator处理填充
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=arguments,
        data_collator=data_collator
    )

    if local_rank == 0:
        print("开始训练...")
        print(f"当前内存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
    # 训练前清空缓存
    torch.cuda.empty_cache()
    
    trainer.train()
    
    if local_rank == 0:
        print("训练完成！")
        print(f"峰值内存使用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    demo()