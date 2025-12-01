import json
import os
from pathlib import Path
from multi_emorag import MultiModalKGRetriever
from tqdm import tqdm

def main():
    input_file = './data/input_data.json'  
    output_file = './data/output_rag.json'  
    
    if not os.path.exists(input_file):
        print(f"file not exist: {input_file}")
        return
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    retriever = MultiModalKGRetriever(
        kg_path='./senticnet_clean.py',
        model_name='clip-ViT-B-32',
        cache_dir='./cache',
        gpu_id=0  
    )
    
    processed_data = []

    for entry in tqdm(data, desc="Processing Entries"):
        result = retriever.retrieve_by_subjects(entry)
        
        for category in result:
            for item in result[category]:
                item.pop('similarity', None)
        
        merged_entry = {
            'filename': entry['filename'],
            'keywords': entry['keywords'],
            'references': result
        }
        processed_data.append(merged_entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成！结果已保存到: {output_file}")
    print(f"共处理了 {len(processed_data)} 条数据")

if __name__ == "__main__":
    main()
