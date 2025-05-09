import json
from src.utils.document_loader import load_documents
from src.embeddings.embedder import Embedder
from src.retrieval.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

def main():
    print("Starting JSON-based RAG application...")
    print("Loading JSON documents...")
    # Load documents from the specified directory
    documents = load_documents('data/documents')
    print(f"Loaded {len(documents)} documents")

    # Initialize the embedder and create embeddings for the documents
    print("Creating embeddings...")
    embedder = Embedder()
    embeddings = embedder.create_embeddings(documents)
    
    # Initialize the document store and add the documents and embeddings
    document_store = DocumentStore()
    document_store.add_documents(documents, embeddings)

    # Initialize the retriever
    retriever = Retriever(document_store)

    # Example query
    query = "What is the significance of RAG in AI?"
    print(f"\nProcessing query: '{query}'")
    
    # Create query embedding
    query_embedding = embedder.create_query_embedding(query)

    # Retrieve relevant documents
    print("Retrieving relevant documents...")
    top_k_documents = retriever.retrieve(query_embedding, top_k=3)
    print(f"Found {len(top_k_documents)} relevant documents")
    
    # Print retrieval results
    print("\nRetrieved documents:")
    for i, doc in enumerate(top_k_documents, 1):
        print(f"{i}. {doc.get('title', doc.get('id', 'Unknown'))} (Score: {doc.get('score', 0.0):.4f})")
    
    # Initialize the generator and generate a response
    print("\nGenerating response...")
    generator = Generator()
    response = generator.generate_response(top_k_documents, query)

    # Print response in a nicely formatted way
    print("\nResponse:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()