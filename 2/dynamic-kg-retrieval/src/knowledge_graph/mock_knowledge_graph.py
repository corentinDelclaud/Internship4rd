class MockKnowledgeGraph:
    def execute_query(self, query):
        # Simulate a knowledge graph query with mock data
        # In a real system, this would query a database or graph engine
        mock_data = [
            {"entity": "climate change", "effect": "temperature rise", "relevance": 0.9},
            {"entity": "biodiversity", "effect": "species loss", "relevance": 0.85},
            {"entity": "climate change", "effect": "sea level rise", "relevance": 0.8},
        ]
        # Return all mock data for demonstration
        return mock_data
