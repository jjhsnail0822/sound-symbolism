import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import re
import contextlib
import gc
import torch
from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel)
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

class MCQExperiment:
    def __init__(
            self,
            model_path: str,
            data_path: str,
            output_dir: str,
            use_api: bool = False,
            tensor_parallel_size: int = 1,
            max_tokens: int = 32,
            max_model_len: int = 4096,
            temperature: float = 0.0,
            thinking: bool = False,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.use_api = use_api
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens if 'Qwen3' not in model_path else max_model_len # for thinking mode
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.thinking = thinking

        # Load environment variables from .env.local file
        env_path = Path('.env.local')
        load_dotenv(dotenv_path=env_path)

        # Get API key from environment variables
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        pass

    def _cleanup(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()

    def run_mcq_experiment(self):
        # Load MCQ data
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        
        # Initialize VLLM model
        if self.use_api:
            print(f"Using OpenAI API with model {self.model_path}")
        else:
            print(f"Loading model from {self.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = LLM(model=self.model_path, max_model_len=self.max_model_len, tensor_parallel_size=self.tensor_parallel_size)
            sampling_params = SamplingParams(
                temperature=self.temperature, 
                max_tokens=self.max_tokens,
            )
        
        # Run experiment
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")
        if self.thinking:
            print("Thinking mode enabled for Qwen3.")
        all_results = []
        
        # Process in batches
        for query in tqdm(mcq_data):
            if self.use_api:
                # Use OpenAI API
                response = self.client.responses.create(
                    model=self.model_path,
                    input=query['question'],
                    # temperature=self.temperature, # not supported with o4-mini
                )
                model_answer = response.output_text
            else:
                # Use VLLM model
                # response = model.generate(query['question'], sampling_params=sampling_params)
                conversation = [
                    {
                        "role": "user",
                        "content": query['question']
                    }
                ]
                if 'Qwen3' in self.model_path:
                    # For Qwen3, use a different prompt format
                    text = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.thinking,
                    )
                    response = model.generate(text, sampling_params=sampling_params)
                else:
                    response = model.chat(conversation, sampling_params=sampling_params)
                
                # Extract answer from the first element of the response list
                model_answer = response[0].outputs[0].text.strip()
                # remove <think> ... </think> tags
                if self.thinking:
                    model_answer = re.sub(r'<think>.*?</think>', '', model_answer, flags=re.DOTALL)

            # First integer is the answer
            model_answer = re.search(r'\d+', model_answer)
            if model_answer:
                model_answer = model_answer.group(0)
            else:
                model_answer = None
            # Handle cases where the model output is empty or None
            if model_answer is None:
                print(f"Warning: Model output is empty for query: {query['question']}")
                model_answer = "0"
            
            # Check correctness
            try:
                is_correct = int(model_answer) == query['answer']
            except ValueError:
                # Handle cases where the model output is not a number
                print(f"Warning: Model output '{model_answer}' is not a valid integer. Marking as incorrect.")
                is_correct = False

            # print(query['question'])
            print(f"Model answer: {model_answer}, Correct answer: {query['answer']}, Is correct: {is_correct}")
            
            # Store result
            all_results.append({
                "query": query['question'],
                "correct_answer": query['answer'],
                "model_answer": model_answer,
                "is_correct": is_correct
            })
        
        # Calculate accuracy
        correct_count = sum(1 for r in all_results if r["is_correct"])
        total_count = len(all_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"Experiment completed. Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        
        # Save results
        model_name = os.path.basename(self.model_path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        results_filename = f"{self.output_dir}/{self.data_path.split('/')[-1].replace('.json', '')}_{model_name}{'-thinking' if self.thinking else ''}.json"
        
        results_dict = {
            "model": self.model_path,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "thinking": self.thinking,
            "results": all_results,
        }

        if not self.use_api:
            del model
            self._cleanup()

        return results_dict, results_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCQ experiment")
    parser.add_argument("--model", '-m', type=str, required=True, help="Path to the LLM")
    parser.add_argument("--data", '-d', type=str, required=True, help="Path to the MCQ data JSON file")
    parser.add_argument("--gpu", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--output", '-o', type=str, default="results/experiments/understanding", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api", action='store_true', help="Use OpenAI API instead of local model")
    parser.add_argument("--thinking", action='store_true', help="Enable thinking mode for Qwen3")
    
    args = parser.parse_args()

    mcq_experiment = MCQExperiment(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        use_api=args.api,
        tensor_parallel_size=args.gpu,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        thinking=args.thinking,
    )
    results, results_filename = mcq_experiment.run_mcq_experiment()
