# RAG Project

This project implements a simple Retrieval Augmented Generation (RAG) system. It loads documents, creates embeddings, retrieves relevant documents based on a query, and generates responses.

## Project Structure

```
2
├── src
│   ├── main.py               # Entry point of the application
│   ├── embeddings
│   │   ├── __init__.py       # Initializer for the embeddings package
│   │   └── embedder.py       # Contains the Embedder class for creating embeddings
│   ├── generation
│   │   ├── __init__.py       # Initializer for the generation package
│   │   └── generator.py       # Contains the Generator class for generating responses
│   ├── retrieval
│   │   ├── __init__.py       # Initializer for the retrieval package
│   │   ├── document_store.py  # Contains the DocumentStore class for managing documents
│   │   └── retriever.py       # Contains the Retriever class for retrieving documents
│   ├── utils
│   │   ├── __init__.py       # Initializer for the utils package
│   │   └── document_loader.py  # Contains the load_documents function for loading documents
├── data
│   └── documents
│       └── sample_doc.txt    # Sample document for testing
├── requirements.txt           # Lists project dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

This will load the documents, create embeddings, and generate a response based on the example query defined in `main.py`. 

## RAG Implementation

The project utilizes a combination of document retrieval and generation techniques to provide contextually relevant responses to user queries. The core components include:

- **Document Loader**: Loads documents from a specified directory.
- **Embedder**: Creates embeddings for documents and queries.
- **Document Store**: Manages the storage of documents and their embeddings.
- **Retriever**: Retrieves the most relevant documents based on the query embedding.
- **Generator**: Generates a coherent response using the retrieved documents and the original query.