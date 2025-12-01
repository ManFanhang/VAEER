import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Union
from PIL import Image
import torch
from pathlib import Path
import importlib.util

class MultiModalKGRetriever:
    def __init__(self, 
                 kg_path: str = './senticnet_clean.py',
                 model_name: str = 'clip-ViT-B-32',
                 cache_dir: str = './cache',
                 top_k: int = 5,
                 text_weight: float = 0.6,
                 gpu_id: Optional[int] = None):
        
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if not os.path.isabs(kg_path):
            kg_path = os.path.join(os.path.dirname(__file__), kg_path)
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(os.path.dirname(__file__), cache_dir)
        
        os.makedirs(cache_dir, exist_ok=True)
        
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"KG does not exist: {kg_path}")
        self.senticnet = self._load_knowledge_graph(kg_path)
        
        self.encoder = SentenceTransformer(model_name)
        self.encoder.to(self.device)
        
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "embeddings.pt")
        self.top_k = top_k
        self.text_weight = text_weight
        
        self.word_embeddings = self._compute_embeddings()
    
    def _load_knowledge_graph(self, kg_path: str) -> Dict:
        try:
            spec = importlib.util.spec_from_file_location("senticnet", kg_path)
            if spec is None:
                raise ImportError(f"fail to fetch KG path: {kg_path}")
            
            senticnet_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(senticnet_module)
            
            return senticnet_module.senticnet
            
        except Exception as e:
            raise Exception(f"fail to load KG: {str(e)}")
    
    def _compute_embeddings(self) -> Dict[str, np.ndarray]:
        if os.path.exists(self.cache_path):
            try:
                return torch.load(self.cache_path, map_location=self.device)
            except:
                pass
        
        words = list(self.senticnet.keys())
        embeddings = {}
        batch_size = 512
        
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
            
            for word, emb in zip(batch, batch_embeddings):
                embeddings[word] = emb
        
        torch.save(embeddings, self.cache_path)
        return embeddings

    def retrieve(self, 
                query_text: str, 
                query_image: Optional[Union[str, Image.Image]] = None,
                return_scores: bool = False) -> List[Dict]:
        text_embedding = self.encoder.encode(
            query_text, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            device=self.device
        )
        
        words = list(self.word_embeddings.keys())
        embeddings_matrix = np.stack([self.word_embeddings[word] for word in words])
        
        text_similarities = np.dot(embeddings_matrix, text_embedding)
        
        if query_image is not None:
            if isinstance(query_image, str):
                query_image = Image.open(query_image)
            
            image_embedding = self.encoder.encode(
                query_image, 
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
            
            image_similarities = np.dot(embeddings_matrix, image_embedding)
            
            similarities = (self.text_weight * text_similarities + 
                          (1 - self.text_weight) * image_similarities)
        else:
            similarities = text_similarities
        
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            word = words[idx]
            info = self.senticnet[word]
            result = {
                'word': word,
                'pleasantness': info[0],
                'attention': info[1],
                'sensitivity': info[2], 
                'aptitude': info[3],
                'primary_mood': info[4],
                'secondary_mood': info[5],
                'polarity': info[6],
                'polarity_value': info[7],
                'related_words': info[8:13]
            }
            if return_scores:
                result['similarity'] = float(similarities[idx])
                if query_image is not None:
                    result['text_similarity'] = float(text_similarities[idx])
                    result['image_similarity'] = float(image_similarities[idx])
            results.append(result)
            
        return results

    def retrieve_by_subjects(self, entry: Dict, query_image: Optional[Union[str, Image.Image]] = None) -> Dict[str, List[Dict]]:
        """subjects-based search w.r.t VF, VC, FI_FCR"""
        results = {
            'VF': [],
            'VC': [],
            'FI_FCR': []
        }
        
        # VF
        VF_keywords = entry.get('keywords', {}).get('coarse_ranking', {}).get('subjects', {}).get('VF', [])
        if VF_keywords:
            VF_results = self._retrieve_and_rank(VF_keywords, query_image)
            results['VF'] = VF_results[:2]
        
        # VC
        VC_keywords = entry.get('keywords', {}).get('coarse_ranking', {}).get('subjects', {}).get('VC', [])
        if VC_keywords:
            VC_results = self._retrieve_and_rank(VC_keywords, query_image)
            results['VC'] = VC_results[:2]
        
        # FI_FCR
        FI_FCR_keywords = entry.get('keywords', {}).get('coarse_ranking', {}).get('subjects', {}).get('FI_FCR', [])
        if FI_FCR_keywords:
            FI_FCR_results = self._retrieve_and_rank(FI_FCR_keywords, query_image)
            results['FI_FCR'] = FI_FCR_results[:2]
        
        return results

    def _retrieve_and_rank(self, keywords: List[str], query_image: Optional[Union[str, Image.Image]] = None) -> List[Dict]:
        combined_results = []
        for keyword in keywords:
            query_text = keyword.strip()
            self.text_weight = 0
            self.top_k = 10
            results = self.retrieve(
                query_text=query_text,
                query_image=query_image,
                return_scores=True
            )
            combined_results.extend(results)
        
        combined_results = sorted(combined_results, key=lambda x: x.get('similarity', 0), reverse=True)
        return combined_results
