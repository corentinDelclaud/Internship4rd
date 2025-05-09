from knowledge_graph.graph_query import GraphQuery
from knowledge_graph.mock_knowledge_graph import MockKnowledgeGraph
from retrieval.uncertainty_assessment import UncertaintyAssessment


def evaluate_rag_method(method_name, query, actual_output, expected_output, retrieval_context):
    uncertainty_assessment = UncertaintyAssessment()
    metric_scores = uncertainty_assessment.assess_all_metrics(
        query=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )
    print(f"\n=== {method_name} ===")
    print("DeepEval Metric Scores:")
    for metric, score in metric_scores.items():
        print(f"{metric}: {score}")


def main():
    query = "What are the effects of climate change on biodiversity?"

    # --- RAG Method 1: Mock RAG ---
    knowledge_graph = MockKnowledgeGraph()
    graph_query = GraphQuery(knowledge_graph)
    results = graph_query.query_graph(query)
    relevance_criteria = ["effect"]
    filtered_results = graph_query.filter_results(results, relevance_criteria)
    actual_output = "Climate change leads to temperature rise and species loss."
    expected_output = "Climate change causes biodiversity loss and temperature increase."
    retrieval_context = [str(r) for r in filtered_results]
    evaluate_rag_method("Mock RAG", query, actual_output, expected_output, retrieval_context)

    # --- RAG Method 2: FASTRAG (example, replace with real outputs) ---
    # from FASTRAG.src.generation.generator import ...
    # actual_output2, expected_output2, retrieval_context2 = ...
    # evaluate_rag_method("FASTRAG", query, actual_output2, expected_output2, retrieval_context2)

    # --- RAG Method 3: DyPRAG (example, replace with real outputs) ---
    # from DyPRAG.src.generation.generator import ...
    # actual_output3, expected_output3, retrieval_context3 = ...
    # evaluate_rag_method("DyPRAG", query, actual_output3, expected_output3, retrieval_context3)

if __name__ == "__main__":
    main()
