# EmoRAG

An emotion-enhanced generation system based on multimodal knowledge graphs.

## Install

```bash
pip install sentence-transformers torch numpy scikit-learn tqdm pillow
```

## Use

### Basics
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

### Batch processing
```bash
python run_rag.py
```

Define directories：
```python
input_file = './data/input.json'
output_file = './data/output.json'
```





