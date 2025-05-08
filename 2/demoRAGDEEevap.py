from src.integration.rag_model_integration import RAGIntegration
from pydantic import BaseModel, Field
from typing import List, Optional

# Define a schema for structured output (optional)
class AnswerWithConfidence(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    confidence: float = Field(description="Confidence score between 0 and 1")
    sources: List[str] = Field(description="Sources that contributed to the answer")

def main():
    # Initialize the RAG integration
    rag = RAGIntegration()
    
    # Example query without schema (free-form text response)
    query = "What is the significance of RAG in AI?"
    response = rag.process_query(query)
    print("\nQuery:", query)
    print("Response:", response[0]["generated_text"])
    
    # Example query with schema (structured output)
    query2 = "How do vector embeddings help in information retrieval?"
    try:
        structured_response = rag.process_query(query2, schema=AnswerWithConfidence)
        print("\nQuery:", query2)
        print("Structured Response:", structured_response.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error with structured output: {e}")
        # Fall back to unstructured response
        response = rag.process_query(query2)
        print("Fallback Response:", response[0]["generated_text"])

if __name__ == "__main__":
    main()