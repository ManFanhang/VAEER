# EmoRAG - 情感增强检索系统

基于多模态知识图谱的情感感知检索增强生成系统。

## 安装

```bash
pip install sentence-transformers torch numpy scikit-learn tqdm pillow
```

## 使用

### 基本用法
```python
from multi_emorag import MultiModalKGRetriever

retriever = MultiModalKGRetriever(
    kg_path='./senticnet_clean.py',
    model_name='clip-ViT-B-32',
    cache_dir='./cache',
    gpu_id=0  # 设置为None使用CPU
)

result = retriever.retrieve_by_subjects(your_data)
```

### 批量处理
```bash
python multi_run_final.py
```

修改脚本中的输入输出路径：
```python
input_file = './data/input.json'
output_file = './data/output.json'
```

## 功能特性

- 基于 CLIP 的多模态检索
- 集成 SenticNet 的8维情感属性


## 配置参数

- `top_k`: 检索数量（默认5）
- `text_weight`: 文本权重（默认0.6）
- `gpu_id`: GPU设备ID（None为CPU）


