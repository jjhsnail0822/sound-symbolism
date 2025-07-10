import json
import argparse
from openai import OpenAI
import base64
from tqdm import tqdm
import os
import re
from dotenv import load_dotenv
from pathlib import Path
import random

random.seed(42)

# Load environment variables from .env.local file
env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Helper function to encode audio to base64
def encode_audio_to_base64(file_path):
    """Encodes an audio file to a base64 string."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

class GPTMCQExperiment:
    def __init__(
            self,
            model_name: str,
            data_path: str,
            output_dir: str,
            exp_name: str,
            max_tokens: int,
            temperature: float = 0.0,
            retry_failed_answers: bool = False,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.exp_name = exp_name
        self.retry_failed_answers = retry_failed_answers

        # Initialize OpenAI client
        # Assumes OPENAI_API_KEY is set in the environment variables.
        print(f"Initializing OpenAI client for model {self.model_name}")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def run_mcq_experiment(self):
        # Load MCQ data
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        
        # Run experiment
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")
        
        # Prepare file for saving results
        model_name_for_file = self.model_name.replace('/', '_')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        results_filename = f"{self.output_dir}/{self.data_path.split('/')[-1].replace('.json', '')}_{model_name_for_file}.json"
        print(f"Results will be saved to: {results_filename}")

        # Load existing results if file exists
        existing_results = {}
        if os.path.exists(results_filename):
            print("Found existing results file. Resuming experiment.")
            with open(results_filename, 'r', encoding='utf-8') as f:
                try:
                    saved_data = json.load(f)
                    # Create a dictionary for quick lookup
                    for res in saved_data.get('results', []):
                        # Assuming meta_data is unique for each question
                        key = json.dumps(res['meta_data'], sort_keys=True)
                        existing_results[key] = res
                    print(f"Loaded {len(existing_results)} existing results.")
                except json.JSONDecodeError:
                    print("Warning: Could not decode JSON from results file. Starting from scratch.")
        
        all_results = []

        # # Pick random 50 questions for testing
        # if len(mcq_data) > 50:
        #     print("More than 50 questions found, selecting a random sample of 50.")
        #     mcq_data = random.sample(mcq_data, 50)

        # Process each question
        for query in tqdm(mcq_data):
            query_key = json.dumps(query['meta_data'], sort_keys=True)

            # --- Logic to skip or retry ---
            if query_key in existing_results:
                existing_result = existing_results[query_key]
                # If not retrying, or if retrying but this one was not a failure, skip
                if not self.retry_failed_answers or existing_result.get("model_answer") != "0":
                    all_results.append(existing_result)
                    continue

            # The user content will be a list for multimodal input
            user_content = []

            if 'audio' in self.exp_name.lower(): # audio experiment
                if '<AUDIO>' in query['question']: # word -> meaning
                    question_first_part = query['question'].split("<AUDIO>")[0]
                    question_second_part = query['question'].split("<AUDIO>")[1]
                    word = query['meta_data']['word']
                    language = query['meta_data']['language']
                    if language == 'art':
                        audio_path = f'data/processed/art/tts/{word}.wav'
                    else:
                        audio_path = f'data/processed/nat/tts/{language}/{word}.wav'
                    if not os.path.exists(audio_path):
                        raise FileNotFoundError(f"Audio file not found: {audio_path}")
                    
                    base64_audio = encode_audio_to_base64(audio_path)
                    
                    user_content.append({"type": "text", "text": question_first_part})
                    user_content.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64_audio,
                            "format": "wav"
                        }
                    })
                    user_content.append({"type": "text", "text": question_second_part})

                else: # meaning -> word
                    question_parts = re.split(r'<AUDIO: .*?>', query['question'])
                    user_content.append({"type": "text", "text": question_parts[0]})

                    for i, option in enumerate(query['options_info']):
                        option_audio_path = f'data/processed/nat/tts/{option["language"]}/{option["text"]}.wav'
                        if not os.path.exists(option_audio_path):
                            raise FileNotFoundError(f"Audio file not found: {option_audio_path}")
                        
                        base64_audio = encode_audio_to_base64(option_audio_path)
                        user_content.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": "wav"
                            }
                        })
                        if i + 1 < len(question_parts):
                            user_content.append({"type": "text", "text": question_parts[i + 1]})

            else: # text experiment
                user_content.append({"type": "text", "text": query['question']})

            messages = [
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            try:
                # Generate response
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                model_answer = response.choices[0].message.content.strip()
                # print(f"Prompt: {messages[0]['content'][0]}")
                # print(f"Model Answer: {model_answer}")
                # print(response)
            except Exception as e:
                print(f"An error occurred while calling the OpenAI API: {e}")
                model_answer = "" # Set empty answer on error

            # Extract first integer as answer
            answer_match = re.search(r'\d+', model_answer)
            if answer_match:
                extracted_answer = answer_match.group(0)
            else:
                extracted_answer = None
            
            # Handle cases where the model output is empty or None
            if extracted_answer is None:
                print(f"Warning: Model output is empty or invalid for query: {query['question'][:50]}...")
                extracted_answer = "0" # Default value that will be marked as incorrect
            
            # Check correctness
            try:
                is_correct = int(extracted_answer) == query['meta_data']['answer']
            except ValueError:
                print(f"Warning: Model output '{extracted_answer}' is not a valid integer. Marking as incorrect.")
                is_correct = False

            # Print debug information
            # print(f"Query: {query['question']}")
            # print(f"Model Answer: {model_answer}")
            # print(f"Extracted Answer: {extracted_answer}")
            # print(f"Is Correct: {is_correct}")
            
            # Store result
            result = {
                "meta_data": query['meta_data'],
                "model_answer": extracted_answer,
                "full_response": model_answer,
                "is_correct": is_correct
            }
            all_results.append(result)

            # --- Save results every 10 queries ---
            if len(all_results) % 10 == 0:
                correct_count = sum(1 for r in all_results if r["is_correct"])
                total_count = len(all_results)
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                results_dict = {
                    "model": self.model_name,
                    "accuracy": accuracy,
                    "correct_count": correct_count,
                    "total_count": total_count,
                    "results": all_results,
                }

                with open(results_filename, 'w', encoding='utf-8') as f:
                    json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
        # --- Final save for any remaining results ---
        correct_count = sum(1 for r in all_results if r["is_correct"])
        total_count = len(all_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        results_dict = {
            "model": self.model_name,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": all_results,
        }
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)

        print(f"Experiment completed. Final Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        print(f"Results saved to: {results_filename}")
        
        return results_dict, results_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCQ experiment with OpenAI GPT models")
    # Note: For audio experiments, you may need to use a specific model version like "gpt-4o-audio-preview"
    parser.add_argument("--model", '-m', type=str, default="gpt-4o", help="OpenAI model name (e.g., gpt-4o, gpt-4o-audio-preview)")
    parser.add_argument("--data", '-d', type=str, required=True, help="Path to the MCQ data JSON file")
    parser.add_argument("--output", '-o', type=str, default="results/experiments/semantic_dimension", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--retry-failed", action='store_true', help="Retry questions where the model previously answered '0'")
    
    args = parser.parse_args()

    experiment = GPTMCQExperiment(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        exp_name=args.exp_name,
        retry_failed_answers=args.retry_failed,
    )
    
    results, results_filename = experiment.run_mcq_experiment()