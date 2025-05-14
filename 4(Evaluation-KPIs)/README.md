How to Use

For automatic metrics (EM, F1):

python generalRAGevaluator.py --predictions example\predictions.json --references example\references.json


For LLM-based comparative evaluation (between two RAGs):

python generalRAGevaluator.py --predictions example\predictions.json --references \example\references.json --comparative --other_predictions example\other_predictions.json --model_name meta-llama/Llama-3.2-1B-Instruct

