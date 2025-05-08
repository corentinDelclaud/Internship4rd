from typing import List, Dict

class Retriever:
    def __init__(self, document_store):
        """Initialize with a document store."""
        self.document_store = document_store
    
    def retrieve(self, query_embedding, top_k=5):
        """Retrieve the most similar documents to the query."""
        results = self.document_store.similarity_search(query_embedding, top_k)
        
        # Post-process results to include relevant metadata
        for result in results:
            # Keep most relevant section at the top
            if 'sections' in result:
                # We'll keep the full document but highlight sections by relevance
                # in future improvements, we could do section-level retrieval
                pass
                
        return results