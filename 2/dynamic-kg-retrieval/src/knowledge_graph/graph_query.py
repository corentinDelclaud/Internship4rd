class GraphQuery:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph

    def query_graph(self, query):
        # Implement the logic to query the knowledge graph
        results = self.knowledge_graph.execute_query(query)
        return results

    def filter_results(self, results, relevance_criteria):
        # Implement the logic to filter results based on relevance
        filtered_results = [result for result in results if self.is_relevant(result, relevance_criteria)]
        return filtered_results

    def is_relevant(self, result, relevance_criteria):
        # Define the logic to determine if a result meets the relevance criteria
        return all(criterion in result for criterion in relevance_criteria)