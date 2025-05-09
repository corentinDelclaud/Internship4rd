import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../1/')))
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
import json
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRelevancyMetric, ContextualRecallMetric

class CustomModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model_id = "mistralai/Mistral-7B-v0.1"
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Load the model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=1024,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if schema is not None:
            # Create parser required for JSON confinement using lmformatenforcer
            parser = JsonSchemaParser(schema.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )

            # Output and load valid JSON
            output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
            output = output_dict[0]["generated_text"][len(prompt) :]
            json_result = json.loads(output)

            # Return valid JSON object according to the schema DeepEval supplied
            return schema(**json_result)
        return pipeline(prompt)

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "mistralai/Mistral-7B-v0.1"
    
class UncertaintyAssessment:
    def __init__(self):
        self.model = CustomModel()
        self.answer_relevancy = AnswerRelevancyMetric(model=self.model)
        self.faithfulness = FaithfulnessMetric(model=self.model)
        self.contextual_precision = ContextualPrecisionMetric(model=self.model)
        self.contextual_relevancy = ContextualRelevancyMetric(model=self.model)
        self.contextual_recall = ContextualRecallMetric(model=self.model)

    def assess_all_metrics(self, query, actual_output, expected_output=None, retrieval_context=None):
        """
        Evaluate all DeepEval RAG metrics and return their scores in a dictionary.
        """
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        scores = {}
        # Answer Relevancy
        self.answer_relevancy.measure(test_case)
        scores['answer_relevancy'] = self.answer_relevancy.score
        # Faithfulness
        self.faithfulness.measure(test_case)
        scores['faithfulness'] = self.faithfulness.score
        # Contextual Precision
        self.contextual_precision.measure(test_case)
        scores['contextual_precision'] = self.contextual_precision.score
        # Contextual Recall
        self.contextual_recall.measure(test_case)
        scores['contextual_recall'] = self.contextual_recall.score
        # Contextual Relevancy
        self.contextual_relevancy.measure(test_case)
        scores['contextual_relevancy'] = self.contextual_relevancy.score
        return scores

    def prioritize_queries(self, queries, actual_output, expected_output=None, retrieval_context=None):
        """
        Ranks queries based on their answer_relevancy score (or another metric if desired).
        """
        return sorted(
            queries,
            key=lambda q: self.assess_all_metrics(q, actual_output, expected_output, retrieval_context)['answer_relevancy'],
            reverse=True
        )