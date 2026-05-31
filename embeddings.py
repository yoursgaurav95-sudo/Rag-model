"""
Embeddings module for generating vector representations of text
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingModel:
    """Generate embeddings using SentenceTransformer models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded model: {model_name} with dimension: {self.embedding_dim}")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim


class EmbeddingCache:
    """Cache embeddings to avoid recomputing"""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, text: str):
        """Get embedding from cache"""
        return self.cache.get(text)
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache"""
        self.cache[text] = embedding
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
