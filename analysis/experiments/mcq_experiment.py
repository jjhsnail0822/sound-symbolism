import json
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
from datetime import datetime
import re

def run_mcq_experiment(
    model_path: str, 
    data_path: str, 
    output_dir: str,
    max_tokens: int = 32,
    max_model_len: int = 8192,
    temperature: float = 0.0,
):
    # Load MCQ data
    print(f"Loading MCQ data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        mcq_data = json.load(f)
    print(f"Loaded {len(mcq_data)} questions.")
    
    # Initialize VLLM model
    print(f"Loading model from {model_path}")
    model = LLM(model=model_path, max_model_len=max_model_len)
    sampling_params = SamplingParams(
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    
    # Run experiment
    print(f"Running MCQ experiment on {len(mcq_data)} questions...")
    all_results = []
    
    # Process in batches
    for query in tqdm(mcq_data):
        response = model.generate(query['question'], sampling_params=sampling_params)
        
        # Extract answer from the first element of the response list
        model_answer = response[0].outputs[0].text.strip()

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

        print(query['question'])
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path)
    results_file = f"{output_dir}/mcq_results_{model_name}_{timestamp}.json"
    
    results_dict = {
        "model": model_path,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "results": all_results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCQ experiment")
    parser.add_argument("--model", '-m', type=str, required=True, help="Path to the LLM")
    parser.add_argument("--data", '-d', type=str, required=True, help="Path to the MCQ data JSON file")
    parser.add_argument("--output", '-o', type=str, default="analysis/experiments/understanding", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum model length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    run_mcq_experiment(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature
    )