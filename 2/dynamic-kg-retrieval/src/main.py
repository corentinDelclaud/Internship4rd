from knowledge_graph.graph_query import GraphQuery
from knowledge_graph.mock_knowledge_graph import MockKnowledgeGraph
from retrieval.uncertainty_assessment import UncertaintyAssessment


def main():
    # Initialize the knowledge graph with a mock implementation
    knowledge_graph = MockKnowledgeGraph()
    graph_query = GraphQuery(knowledge_graph)
    uncertainty_assessment = UncertaintyAssessment()

    # Example query
    query = "What are the effects of climate change on biodiversity?"

    # Query the knowledge graph
    results = graph_query.query_graph(query)

    # For demonstration, use a simple relevance criteria (e.g., 'effect')
    relevance_criteria = ["effect"]
    filtered_results = graph_query.filter_results(results, relevance_criteria)

    # Prepare dummy outputs for metric testing
    # In a real RAG pipeline, these would be the generated answer, expected answer, and context
    actual_output = "Climate change leads to temperature rise and species loss."
    expected_output = "Climate change causes biodiversity loss and temperature increase."
    retrieval_context = [str(r) for r in filtered_results]

    # Assess all DeepEval metrics
    metric_scores = uncertainty_assessment.assess_all_metrics(
        query=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    # Output the results
    print("Filtered Results:", filtered_results)
    print("\nDeepEval Metric Scores:")
    for metric, score in metric_scores.items():
        print(f"{metric}: {score}")

if __name__ == "__main__":
    main()