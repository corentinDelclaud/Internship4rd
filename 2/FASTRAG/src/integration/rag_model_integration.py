import sys
import os

# Add paths to both projects so imports work correctly
sys.path.append('/home/naeem/Documents/corentin/git/Internship4rd/1')
sys.path.append('/home/naeem/Documents/corentin/git/Internship4rd/2')

from Uncertainty_module.module import CustomModel  # Update the path to the correct module location
from src.embeddings.embedder import Embedder
from src.retrieval.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.utils.document_loader import load_documents
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

class RAGIntegration:
    """
    Integrates RAG retrieval capabilities with CustomModel generation.
    """
    
    def __init__(self, documents_path="data/documents"):
        """
        Initialize the RAG integration with documents and models.
        """
        # Initialize RAG components
        self.documents = load_documents(documents_path)
        self.embedder = Embedder()
        self.document_store = DocumentStore()
        
        # Create embeddings for documents and add to document store
        embeddings = self.embedder.create_embeddings(self.documents)
        self.document_store.add_documents(self.documents, embeddings)
        
        # Initialize retriever
        self.retriever = Retriever(self.document_store)
        
        # Initialize the custom model from Uncertainty module
        self.model = CustomModel()
        
        print(f"RAG Integration initialized with {len(self.documents)} documents")
    
    def _format_prompt_with_context(self, query: str, documents: List[Dict]) -> str:
        """
        Format a prompt that includes retrieved context and the query.
        """
        context = ""
        for doc in documents:
            # Add document sections to context
            if 'sections' in doc:
                for section in doc['sections']:
                    if 'heading' in section and 'content' in section:
                        context += f"## {section['heading']}\n{section['content']}\n\n"
                    elif 'content' in section:
                        context += f"{section['content']}\n\n"
            # If no sections, use the flat content
            elif 'content' in doc:
                context += doc['content'] + "\n\n"
        
        # Create a prompt with the context and query
        prompt = f"""Please answer the following question based on the context provided.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def process_query(self, query: str, top_k: int = 3, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """
        Process a query through the RAG pipeline and generate a response.
        
        Args:
            query: The user's question
            top_k: Number of documents to retrieve
            schema: Optional Pydantic model for structured output
            
        Returns:
            Generated response or structured data according to schema
        """
        # Create query embedding
        query_embedding = self.embedder.create_query_embedding(query)
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query_embedding, top_k=top_k)
        
        # Format prompt with context and query
        prompt = self._format_prompt_with_context(query, retrieved_docs)
        
        # Generate response using the custom model
        response = self.model.generate(prompt, schema)
        
        return response