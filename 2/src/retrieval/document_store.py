import numpy as np
from typing import List, Dict

class DocumentStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents, embeddings):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def get_documents(self):
        return self.documents

    def get_embeddings(self):
        return self.embeddings
    
    def similarity_search(self, query_embedding, top_k=5):
        """Find the most similar documents to the query embedding."""
        if not self.embeddings or len(self.embeddings) == 0:
            return []
        
        # Convert embeddings to numpy array for efficient computation
        embeddings_array = np.array(self.embeddings)
        query_array = np.array(query_embedding)
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings_array, query_array) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_array)
        )
        
        # Get indices of top_k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return documents with their similarity scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
            
        return results