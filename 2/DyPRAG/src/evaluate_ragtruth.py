import json
import argparse
import os
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def evaluate_truthfulness(dyprag_file: str, rag_file: str, output_file: str, model_name: str, device: str = "cuda") -> Dict[str, str]:
    # Load results from JSON files
    with open(dyprag_file, 'r') as f:
        dyprag_results = json.load(f)
    with open(rag_file, 'r') as f:
        rag_results = json.load(f)

    # Load HuggingFace model and tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    RAGTRUTH_PROMPT_TEMPLATE = """Compare DyPRAG and RAG answers to assess which better internalizes knowledge—integrating its own knowledge with the given context for a natural, informed response.
Evaluation Criteria:
1. Internalization: Does the answer go beyond repetition to integrate knowledge seamlessly?
2. Fluency: Is the response well-structured and readable?
3. Relevance: Does it stay on topic while demonstrating depth?

Mark the Winner: Identify the superior response. If both are equally strong, mark it as a tie.

Question: {question}
Context: {context}
DyPRAG Answer: {dyprag_answer}
RAG Answer: {rag_answer}

Respond in the following format:
{{
  "win model": "DyPRAG or RAG or Tie",
  "reason": "Provide a concise explanation of why the selected answer demonstrates better knowledge integration, referencing the question, context, and specific details from both answers. If one answer has clear advantages in integration, explain them; if there are errors or weaknesses, specify them."
}}"""

    ret = []
    for dyprag_result, rag_result in zip(dyprag_results, rag_results):
        question = dyprag_result['question']
        context = rag_result['passages']
        dyprag_answer = " ".join(dyprag_result['text'].split("\n")).strip()
        dyprag_answer = dyprag_answer.split('assistant')[0]
        rag_answer = " ".join(rag_result['text'].split("\n")).strip()
        rag_answer = rag_answer.split('assistant')[0]
        prompt = RAGTRUTH_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            dyprag_answer=dyprag_answer,
            rag_answer=rag_answer
        )

        try:
            response = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
            # Try to extract the JSON from the response
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            result = json.loads(json_str)
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            result = {"win model": "Error", "reason": str(e)}

        output = {
            "test_id": dyprag_result.get('test_id', None),
            "question": question,
            "context": context,
            "dyprag_answer": dyprag_answer,
            "rag_answer": rag_answer,
            "evaluation": result
        }
        ret.append(output)
        print(f"Winner: {result.get('win model', 'N/A')}")
        print(f"Reason: {result.get('reason', 'N/A')}")
        with open(output_file, "w") as f:
            json.dump(ret, f, indent=2)

def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "evaluation_results.json")
    evaluate_truthfulness(
        dyprag_file=args.dyprag_path,
        rag_file=args.rag_path,
        output_file=output_file,
        model_name=args.model_name,
        device=args.device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dyprag_path", type=str, required=True)
    parser.add_argument("--rag_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str,default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda or cpu)")
    args = parser.parse_args()
    print(args)
    main(args)