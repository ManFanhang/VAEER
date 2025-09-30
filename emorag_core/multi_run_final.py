import json
import os
from pathlib import Path
from multi_emorag import MultiModalKGRetriever
from tqdm import tqdm

def main():
    # 配置路径（请根据实际情况修改）
    input_file = './data/input_data.json'  # 输入JSON文件路径
    output_file = './data/output_rag.json'  # 输出文件路径
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        print("请修改 input_file 变量为正确的路径")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 加载 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化检索器
    retriever = MultiModalKGRetriever(
        kg_path='./senticnet_clean.py',
        model_name='clip-ViT-B-32',
        cache_dir='./cache',
        gpu_id=0  # 设置为None禁用GPU，或设置具体GPU ID
    )
    
    # 结果存储
    processed_data = []

    # 处理数据，使用 tqdm 显示进度条
    for entry in tqdm(data, desc="Processing Entries"):
        result = retriever.retrieve_by_subjects(entry)
        
        # 去掉 similarity 字段
        for category in result:
            for item in result[category]:
                item.pop('similarity', None)
        
        # 合并结果
        merged_entry = {
            'filename': entry['filename'],
            'keywords': entry['keywords'],
            'references': result
        }
        processed_data.append(merged_entry)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成！结果已保存到: {output_file}")
    print(f"共处理了 {len(processed_data)} 条数据")

if __name__ == "__main__":
    main()