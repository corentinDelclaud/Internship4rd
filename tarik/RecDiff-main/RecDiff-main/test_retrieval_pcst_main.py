import torch
import pandas as pd
import numpy as np
from retrieval import retrieval_via_pcst
from torch_geometric.data.data import Data
import pickle
from DataHander import DataHandler

# Helper to run retrieval_via_pcst with different params on real data

def run_tests_on_main_graphs():
    results = []
    handler = DataHandler()
    handler.LoadData()
    # Use the same logic as in main.py for q_emb, textual_nodes, textual_edges
    ui_graph = handler.ui_graph
    uu_graph = handler.uu_graph
    q_emb = ui_graph.x if hasattr(ui_graph, 'x') else torch.zeros((ui_graph.num_nodes(), 16))
    textual_nodes = getattr(handler, 'textual_nodes', None)
    textual_edges = getattr(handler, 'textual_edges', None)
    if textual_nodes is None or textual_edges is None:
        print("Warning: textual_nodes or textual_edges not found in handler. Using dummy data.")
        num_nodes = ui_graph.num_nodes() if callable(ui_graph.num_nodes) else ui_graph.num_nodes
        num_edges = ui_graph.edge_index.shape[1]
        textual_nodes = pd.DataFrame({"node_id": range(num_nodes), "desc": [f"node_{i}" for i in range(num_nodes)]})
        textual_edges = pd.DataFrame({"src": np.random.randint(0, num_nodes, num_edges),
                                     "edge_attr": [f"edge_{i}" for i in range(num_edges)],
                                     "dst": np.random.randint(0, num_nodes, num_edges)})

    param_grid = [
        {"topk_e": 2, "cost_e": 0.2},
        {"topk_e": 3, "cost_e": 0.5},
        {"topk_e": 5, "cost_e": 0.7},
        {"topk_e": 8, "cost_e": 1.0},
        {"topk_e": 10, "cost_e": 2.0},
    ]

    for params in param_grid:
        data, desc = retrieval_via_pcst(
            ui_graph, q_emb, textual_nodes, textual_edges,
            topk_e=params["topk_e"], cost_e=params["cost_e"]
        )
        results.append({
            "params": params,
            "num_edges": data.edge_index.shape[1],
            "num_nodes": data.num_nodes,
            "desc": desc
        })
        print(f"Test with params {params}: num_edges={data.edge_index.shape[1]}, num_nodes={data.num_nodes}")

    # Save results to a pickle file
    with open("retrieval_pcst_main_test_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to retrieval_pcst_main_test_results.pkl")

if __name__ == "__main__":
    run_tests_on_main_graphs()
