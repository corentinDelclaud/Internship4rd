from transformers import pipeline
import torch
import json

class Generator:
    def __init__(self, model_name="google/flan-t5-small"):
        """Initialize with a language model."""
        print(f"Loading generation model: {model_name}")
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline('text2text-generation', model=model_name, device=device)
    
    def generate_response(self, retrieved_documents, query):
        """Generate a response based on retrieved documents and query."""
        if not retrieved_documents:
            return {
                "answer": f"No relevant information found for: {query}",
                "sources": []
            }
        
        # Build context from retrieved documents
        context = ""
        sources = []
        
        for doc in retrieved_documents:
            # Add document information to sources
            sources.append({
                "title": doc.get('title', doc.get('id', 'Unknown')),
                "author": doc.get('author', 'Unknown'),
                "date": doc.get('date', 'Unknown'),
                "score": doc.get('score', 0.0)
            })
            
            # Add content to context
            if 'sections' in doc:
                for section in doc['sections']:
                    context += f"{section.get('heading', '')}: {section.get('content', '')}\n\n"
            elif 'content' in doc:
                context += f"{doc['content']}\n\n"
        
        # Create a prompt for the generator
        prompt = f"Answer the following question based on the provided context.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate a response
        result = self.generator(prompt, max_length=150, min_length=30)
        answer = result[0]['generated_text']
        
        # Return a structured response with the answer and sources
        response = {
            "answer": answer,
            "sources": sources
        }
        
        return response