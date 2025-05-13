import argparse
import json
import numpy as np
import re
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate_automatic(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    ems, f1s = [], []
    for pred, gt in zip(predictions, ground_truths):
        ems.append(compute_em(pred, gt))
        f1s.append(compute_f1(pred, gt))
    return {
        "EM": np.mean(ems),
        "F1": np.mean(f1s)
    }

def extract_first_json(text):
    # Find all curly-brace blocks
    matches = list(re.finditer(r'\{.*?\}', text, re.DOTALL))
    for match in matches:
        try:
            return json.loads(match.group())
        except Exception:
            continue
    raise ValueError("No valid JSON object found in model output.")

def evaluate_llm_comparative(
    questions: List[str],
    contexts: List[str],
    answers_a: List[str],
    answers_b: List[str],
    model_name: str,
    device: str = "cuda"
) -> List[Dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt_template = """Compare Answer A and Answer B for the following question and context.
Evaluation Criteria:
1. Internalization: Does the answer integrate knowledge, not just repeat context?
2. Fluency: Is the answer well-structured and readable?
3. Relevance: Is the answer on-topic and deep?
4. EM: Does the answer match the ground truth?
5. F1: Does the answer contain relevant information from the context?

Mark the Winner: Identify the superior response. If both are equally strong, mark it as a tie qnd for the Em qnd F1 just note the  and the score.

Question: {question}
Context: {context}
Answer A: {answer_a}
Answer B: {answer_b}

Respond ONLY with a single JSON object in the following format with your respond for the winner model and the reason of your choice, it's possible to have a tie:
{{
  "win model": ,
  "reason":,
    "EM": ,
    "F1": 
}}
"""

    results = []
    for q, c, a, b in zip(questions, contexts, answers_a, answers_b):
        prompt = prompt_template.format(question=q, context=c, answer_a=a, answer_b=b)
        response = generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        try:
            result = extract_first_json(response)
        except Exception as e:
            result = {"win model": "Error", "reason": str(e)}
        results.append({
            "question": q,
            "context": c,
            "answer_a": a,
            "answer_b": b,
            "evaluation": result
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to RAG predictions JSONL or JSON")
    parser.add_argument("--references", type=str, required=True, help="Path to ground truth answers JSONL or JSON")
    parser.add_argument("--comparative", action="store_true", help="If set, run LLM-based comparative evaluation")
    parser.add_argument("--other_predictions", type=str, help="Path to second RAG predictions for comparison")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model for LLM-based eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    # Load predictions and references
    with open(args.predictions) as f:
        preds = json.load(f)
    with open(args.references) as f:
        refs = json.load(f)

    # Automatic metrics
    predictions = [p["prediction"] if isinstance(p, dict) else p for p in preds]
    ground_truths = [r["answer"] if isinstance(r, dict) else r for r in refs]
    auto_metrics = evaluate_automatic(predictions, ground_truths)
    print("Automatic Metrics:", auto_metrics)

    # LLM-based comparative evaluation
    if args.comparative and args.other_predictions:
        with open(args.other_predictions) as f:
            other_preds = json.load(f)
        questions = [p["question"] if isinstance(p, dict) else "" for p in preds]
        contexts = [p.get("context", "") if isinstance(p, dict) else "" for p in preds]
        answers_a = [p["prediction"] if isinstance(p, dict) else p for p in preds]
        answers_b = [p["prediction"] if isinstance(p, dict) else p for p in other_preds]
        results = evaluate_llm_comparative(
            questions, contexts, answers_a, answers_b, args.model_name, args.device
        )
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"LLM-based comparative evaluation saved to {args.output}")

if __name__ == "__main__":
    main()