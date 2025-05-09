# Dynamic Knowledge Graph Retrieval

This project implements a dynamic knowledge graph retrieval system that efficiently queries a knowledge graph based on assessed uncertainty, ensuring the retrieval of relevant and accurate information.

## Project Structure

```
dynamic-kg-retrieval
├── src
│   ├── main.py                # Entry point of the application
│   ├── knowledge_graph        # Module for knowledge graph querying
│   │   ├── __init__.py
│   │   ├── graph_query.py     # Contains GraphQuery class for querying the graph
│   │   └── mock_knowledge_graph.py # Mock knowledge graph for testing/demo
│   ├── retrieval              # Module for uncertainty assessment
│   │   ├── __init__.py
│   │   └── uncertainty_assessment.py  # Contains UncertaintyAssessment class
│   └── utils                  # Utility functions
│       └── __init__.py
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd dynamic-kg-retrieval
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python src/main.py
```

### Example

1. Assess the uncertainty of a query using the `UncertaintyAssessment` class.
2. Query the knowledge graph using the `GraphQuery` class, which now requires a knowledge graph instance (e.g., `MockKnowledgeGraph`).
3. Filter the results based on relevance.

#### Sample Code

```python
from knowledge_graph.graph_query import GraphQuery
from knowledge_graph.mock_knowledge_graph import MockKnowledgeGraph
from retrieval.uncertainty_assessment import UncertaintyAssessment

# Initialize components
knowledge_graph = MockKnowledgeGraph()
graph_query = GraphQuery(knowledge_graph)
uncertainty_assessment = UncertaintyAssessment()

query = "What are the effects of climate change on biodiversity?"
uncertainty_score = uncertainty_assessment.assess_uncertainty(query)
prioritized_queries = uncertainty_assessment.prioritize_queries([query])
results = graph_query.query_graph(prioritized_queries[0])
relevance_criteria = ["climate change", "biodiversity"]
filtered_results = graph_query.filter_results(results, relevance_criteria)

print("Uncertainty Score:", uncertainty_score)
print("Filtered Results:", filtered_results)
```

Refer to the source code for detailed usage of the classes and methods.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.