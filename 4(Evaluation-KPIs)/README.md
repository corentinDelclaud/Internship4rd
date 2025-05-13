How to Use

For automatic metrics (EM, F1):

python general_rag_evaluate.py --predictions path/to/preds.json --references path/to/refs.json


For LLM-based comparative evaluation (between two RAGs):

python general_rag_evaluate.py --predictions path/to/preds_a.json --references path/to/refs.json --comparative --other_predictions path/to/preds_b.json --model_name meta-llama/Meta-Llama-3-8B-Instruct


