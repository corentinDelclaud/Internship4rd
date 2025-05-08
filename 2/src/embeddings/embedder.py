from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model."""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def create_embeddings(self, documents):
        """Create embeddings for documents."""
        if not documents:
            return []
            
        # If documents is a list of strings
        if isinstance(documents[0], str):
            return self.model.encode(documents).tolist()
        
        # If documents is a list of dictionaries with 'content' key
        contents = [doc['content'] for doc in documents]
        return self.model.encode(contents).tolist()
    
    def create_query_embedding(self, query):
        """Create embedding for a single query."""
        return self.model.encode(query).tolist()