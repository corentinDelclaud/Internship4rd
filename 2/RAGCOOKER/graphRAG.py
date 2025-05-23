import networkx as nx
from cdlib import algorithms
from constants import DOCUMENTS
import ollama
import re
import csv
import concurrent.futures

# 1. Source Documents → Text Chunks
def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
    chunks = []
    for document in documents :
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

# 2. Text Chunks → Element Instances
def extract_elements_from_chunks(chunks):
    elements = []
    print(f"Extracting elements from {len(chunks)} chunks...")
    def process_chunk(chunk):
        response = ollama.chat(
            model="llama2",
            messages=[
                {"role": "system", "content": "Extract entities and relationships from the following text."},
                {"role": "user", "content": chunk}
            ]
        )
        return response['message']['content']
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))
    elements.extend(results)
    print(f"Extracted elements from all chunks.")
    return elements

# 3. Element Instances → Element Summaries
def summarize_elements(elements):
    summaries = []
    print(f"Summarizing {len(elements)} elements...")
    def process_element(element):
        response = ollama.chat(
            model="llama2",
            messages=[
                {"role": "system", "content": "Summarize the following entities and relationships in a structured format. Use '->' to represent relationships, after the 'Relationships:' word."},
                {"role": "user", "content": element}
            ]
        )
        return response['message']['content']
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_element, elements))
    summaries.extend(results)
    print(f"Summarized all elements.")
    return summaries

# 4. Element Summaries → Graph Communities
def build_graph_from_summaries(summaries):
    print(f"Building graph from {len(summaries)} summaries...")
    G = nx.Graph()
    for index, summary in enumerate(summaries):
        print(f"Processing summary {index+1}/{len(summaries)}")
        lines = summary.split("\n")
        entities_section = False
        relationships_section = False
        entities = []
        for line in lines:
            if line.startswith("### Entities:") or line.startswith("**Entities:**"):
                entities_section = True
                relationships_section = False
                continue
            elif line.startswith("### Relationships:") or line.startswith("**Relationships:**"):
                entities_section = False
                relationships_section = True
                continue
            if entities_section and line.strip():
                # Remove leading numbering like '1. ', '12. ', etc.
                entity = re.sub(r'^\d+\.\s*', '', line.strip())
                entity = entity.replace("**", "")
                entities.append(entity)
                G.add_node(entity)
            elif relationships_section and line.strip():
                parts = line.split("->")
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[-1].strip()
                    relation = " -> ".join(parts[1:-1]).strip()
                    G.add_edge(source, target, label=relation)
    print("Graph construction complete.")
    return G

# 5. Graph Communities → Community Summaries
def detect_communities(graph):
    print("Detecting communities in the graph...")
    communities = []
    components = list(nx.connected_components(graph))
    for index, component in enumerate(components):
        print(f"Processing component {index+1}/{len(components)}")
        subgraph = graph.subgraph(component)
        if len(subgraph.nodes) > 1:  # Leiden algorithm requires at least 2 nodes
            try:
                sub_communities = algorithms.leiden(subgraph)
                for community in sub_communities.communities:
                    communities.append(list(community))
            except Exception as e:
                print(f"Error processing community {index}: {e}")
        else:
            communities.append(list(subgraph.nodes))
    print("Communities from detect_communities:", communities)
    return communities

def summarize_communities(communities, graph):
    print(f"Summarizing {len(communities)} communities...")
    community_summaries = []
    for index, community in enumerate(communities):
        print(f"Summarizing community {index+1}/{len(communities)}")
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(
                f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
        description += ", ".join(relationships)

        response = ollama.chat(
            model="llama2",
            messages=[
                {"role": "system", "content": "Summarize the following community of entities and relationships."},
                {"role": "user", "content": description}
            ]
        )
        summary = response['message']['content']
        community_summaries.append(summary.strip())
    print("Community summarization complete.")
    return community_summaries

# 6. Community Summaries → Community Answers → Global Answer
def generate_answers_from_communities(community_summaries, query):
    print(f"Generating answers from {len(community_summaries)} communities...")
    intermediate_answers = []
    for index, summary in enumerate(community_summaries):
        print(f"Generating answer from community {index+1}/{len(community_summaries)}")
        response = ollama.chat(
            model="llama2",
            messages=[
                {"role": "system", "content": "Answer the following query based on the provided summary."},
                {"role": "user", "content": f"Query: {query} Summary: {summary}"}
            ]
        )
        content = response['message']['content']
        intermediate_answers.append(content)
    print("Combining intermediate answers into final answer...")
    final_response = ollama.chat(
        model="llama2",
        messages=[
            {"role": "system", "content": "Combine these answers into a final, concise response."},
            {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
        ]
    )
    final_answer = final_response['message']['content']
    print("Final answer generated.")
    return final_answer

# Putting It All Together
def graph_rag_pipeline(documents, query, chunk_size=600, overlap_size=100):
    print("Starting GraphRAG pipeline...")
    # Step 1: Split documents into chunks
    chunks = split_documents_into_chunks(
        documents, chunk_size, overlap_size)
    print(f"Split documents into {len(chunks)} chunks.")

    # Step 2: Extract elements from chunks
    elements = extract_elements_from_chunks(chunks)

    # Step 3: Summarize elements
    summaries = summarize_elements(elements)

    # Step 4: Build graph and detect communities
    graph = build_graph_from_summaries(summaries)
    print("graph:", graph)
    communities = detect_communities(graph)

    if not communities:
        print("No communities found. Returning empty answer.")
        return "No answer could be generated (no communities found)."
    else:
        print("communities:", communities[0])
    # Step 5: Summarize communities
    community_summaries = summarize_communities(communities, graph)

    # Step 6: Generate answers from community summaries
    final_answer = generate_answers_from_communities(
        community_summaries, query)

    print("GraphRAG pipeline complete.")
    return final_answer