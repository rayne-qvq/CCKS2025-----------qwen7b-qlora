import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import seaborn as sns
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def analyze_text_length(data, title):
    text_sizes = [len(item['text']) for item in data]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(text_sizes, bins=50)
    plt.title(f'{title}文本长度分布')
    plt.xlabel('文本长度')
    plt.ylabel('频次')
    plt.savefig(f'data/analysis/{title}_length_dist.png')
    plt.close()
    
    return {
        '最大长度': max(text_sizes),
        '最小长度': min(text_sizes),
        '平均长度': int(sum(text_sizes) / len(text_sizes)),
        '中位数长度': int(np.median(text_sizes)),
        '标准差': int(np.std(text_sizes))
    }

def analyze_label_distribution(data, title):
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    
    plt.figure(figsize=(8, 6))
    plt.pie(label_counts.values(), labels=['manually written', 'AI-generated'], autopct='%1.1f%%')
    plt.title(f'{title}标签分布')
    plt.savefig(f'data/analysis/{title}_label_dist.png')
    plt.close()
    
    return label_counts

def analyze_duplicates(data):
    texts = [item['text'] for item in data]
    unique_texts = set(texts)
    duplicates = len(texts) - len(unique_texts)
    
    # 找出重复的文本
    text_counter = Counter(texts)
    duplicate_texts = {text: count for text, count in text_counter.items() if count > 1}
    
    return {
        '总样本数': len(texts),
        '唯一文本数': len(unique_texts),
        '重复文本数': duplicates,
        '重复率': round(duplicates / len(texts) * 100, 2),
        '重复文本示例': dict(list(duplicate_texts.items())[:5])  # 只显示前5个重复文本
    }

def analyze_text_features(data):
    # 计算平均词数
    avg_words = np.mean([len(word_tokenize(item['text'])) for item in data])
    
    # 计算标点符号使用情况
    punctuation_pattern = re.compile(r'[^\w\s]')
    avg_punctuation = np.mean([len(punctuation_pattern.findall(item['text'])) for item in data])
    
    return {
        '平均词数': round(avg_words, 2),
        '平均标点符号数': round(avg_punctuation, 2)
    }

def clean_data(data):
    # 1. 去除重复文本
    unique_texts = set()
    cleaned_data = []
    for item in data:
        if item['text'] not in unique_texts:
            unique_texts.add(item['text'])
            cleaned_data.append(item)
    
    # 2. 去除过短和过长的文本
    cleaned_data = [item for item in cleaned_data if 100 < len(item['text']) < 800]
    
    return cleaned_data

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='加载数据'):
            data.append(json.loads(line))
    return data

def demo():
    # 创建分析结果目录
    import os
    os.makedirs('data/analysis', exist_ok=True)
    
    # 加载数据
    print('正在加载数据...')
    data = load_jsonl('data/train.jsonl')
    print(f'原始数据量: {len(data)}')
    
    # 数据分析
    print('\n=== 原始数据分析 ===')
    length_stats = analyze_text_length(data, '原始数据')
    print('文本长度统计:', length_stats)
    
    label_stats = analyze_label_distribution(data, '原始数据')
    print('标签分布:', label_stats)
    
    duplicate_stats = analyze_duplicates(data)
    print('重复文本分析:', duplicate_stats)
    
    feature_stats = analyze_text_features(data)
    print('文本特征分析:', feature_stats)
    
    # 数据清洗
    print('\n正在清洗数据...')
    cleaned_data = clean_data(data)
    print(f'清洗后数据量: {len(cleaned_data)}')
    
    # 清洗后数据分析
    print('\n=== 清洗后数据分析 ===')
    length_stats = analyze_text_length(cleaned_data, '清洗后')
    print('文本长度统计:', length_stats)
    
    label_stats = analyze_label_distribution(cleaned_data, '清洗后')
    print('标签分布:', label_stats)
    
    # 数据分割
    labels = [item['label'] for item in cleaned_data]
    train_data, test_data = train_test_split(cleaned_data, test_size=0.05, stratify=labels, random_state=42)
    
    print('\n=== 训练集分析 ===')
    length_stats = analyze_text_length(train_data, '训练集')
    print('文本长度统计:', length_stats)
    label_stats = analyze_label_distribution(train_data, '训练集')
    print('标签分布:', label_stats)
    
    print('\n=== 测试集分析 ===')
    length_stats = analyze_text_length(test_data, '测试集')
    print('文本长度统计:', length_stats)
    label_stats = analyze_label_distribution(test_data, '测试集')
    print('标签分布:', label_stats)

    # 采样部分数据
    sample_num = 28000
    train_data = train_data[1000:int(len(train_data) - 1000)]
    
    # 采样部分数据
    sample_num = 26000
    train_data = train_data[:8000]
    test_data = test_data[:int(sample_num * 0.2)]
    
    print(f'\n最终训练集数量: {len(train_data)}')
    print(f'最终测试集数量: {len(test_data)}')
    
    # 数据本地存储
    pickle.dump(train_data, open('data/01-训练集.pkl', 'wb'))
    pickle.dump(test_data, open('data/02-测试集.pkl', 'wb'))
    
    print('\n数据分析完成！可视化结果保存在 data/analysis 目录下')

if __name__ == '__main__':
    demo()