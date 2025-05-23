import json
from graphRAG import graph_rag_pipeline, DOCUMENTS

# List of questions/queries to process
questions = [
    "What are some key renewable energy sources mentioned?",
    "What is the main purpose of environmental policies?",
]

results = []
for query in questions:
    print(f"Processing query: {query}")
    answer = graph_rag_pipeline(DOCUMENTS, query)
    results.append({
        "question": query,
        "context": query,  # Here, context is the same as the query
        "answer": answer
    })

# Write results to a JSON file
with open("graphrag_answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results saved to graphrag_answers.json")