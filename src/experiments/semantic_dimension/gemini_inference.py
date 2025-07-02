import json
import argparse
from google import genai
import base64
from tqdm import tqdm
import os
import re
from dotenv import load_dotenv
from pathlib import Path
import random
from google.genai import types

random.seed(42)

# Load environment variables from .env.local file
env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Helper function to encode audio to base64
def encode_audio_to_base64(file_path):
    """Encodes an audio file to a base64 string."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def read_audio_bytes(file_path):
    """Reads an audio file and returns its content as bytes."""
    with open(file_path, "rb") as audio_file:
        return audio_file.read()

class GeminiMCQExperiment:
    def __init__(
            self,
            model_name: str,
            data_path: str,
            output_dir: str,
            exp_name: str,
            max_tokens: int = 32,
            temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.exp_name = exp_name

        # Initialize Gemini client
        print(f"Initializing Gemini client for model {self.model_name}")
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def run_mcq_experiment(self):
        # Load MCQ data
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        
        # Run experiment
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")
        all_results = []

        # Pick random 50 questions for testing
        if len(mcq_data) > 50:
            print("More than 50 questions found, selecting a random sample of 50.")
            mcq_data = random.sample(mcq_data, 50)

        # Process each question
        for query in tqdm(mcq_data):
            # The user content will be a list of parts for Gemini
            parts = []

            if 'audio' in self.exp_name.lower(): # audio experiment
                if '<AUDIO>' in query['question']: # word -> meaning
                    question_first_part = query['question'].split("<AUDIO>")[0]
                    question_second_part = query['question'].split("<AUDIO>")[1]
                    word = query['meta_data']['word']
                    language = query['meta_data']['language']
                    audio_path = f'data/processed/nat/tts/{language}/{word}.wav'
                    if not os.path.exists(audio_path):
                        raise FileNotFoundError(f"Audio file not found: {audio_path}")
                    
                    audio_bytes = read_audio_bytes(audio_path)
                    
                    parts.append(question_first_part)
                    parts.append(types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type="audio/wav"
                    ))
                    parts.append(question_second_part)

                else: # meaning -> word
                    question_parts = re.split(r'<AUDIO: .*?>', query['question'])
                    parts.append(question_parts[0])

                    for i, option in enumerate(query['options_info']):
                        option_audio_path = f'data/processed/nat/tts/{option["language"]}/{option["text"]}.wav'
                        if not os.path.exists(option_audio_path):
                            raise FileNotFoundError(f"Audio file not found: {option_audio_path}")
                        
                        audio_bytes = read_audio_bytes(option_audio_path)
                        parts.append(types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type="audio/wav"
                        ))
                        if i + 1 < len(question_parts):
                            parts.append(question_parts[i + 1])

            else: # text experiment
                parts.append(query['question'])

            try:
                # Generate response
                generation_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0
                    ),
                    maxOutputTokens=self.max_tokens,
                    temperature=self.temperature,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=generation_config,
                )
                model_answer = response.text.strip()
            except Exception as e:
                print(f"An error occurred while calling the Gemini API: {e}")
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
            print(f"Query: {query['question']}")
            print(f"Model Answer: {model_answer}")
            print(f"Extracted Answer: {extracted_answer}")
            print(f"Is Correct: {is_correct}")
            
            # Store result
            result = {
                "meta_data": query['meta_data'],
                "model_answer": extracted_answer,
                "full_response": model_answer,
                "is_correct": is_correct
            }
            all_results.append(result)
        
        # Calculate accuracy
        correct_count = sum(1 for r in all_results if r["is_correct"])
        total_count = len(all_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"Experiment completed. Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        
        # Save results
        model_name_for_file = self.model_name.replace('/', '_')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        results_filename = f"{self.output_dir}/{self.data_path.split('/')[-1].replace('.json', '')}_{model_name_for_file}.json"
        
        results_dict = {
            "model": self.model_name,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": all_results,
        }

        # Save results to file
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
        print(f"Results saved to: {results_filename}")
        
        return results_dict, results_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCQ experiment with Google Gemini models")
    parser.add_argument("--model", '-m', type=str, default="gemini-1.5-pro-latest", help="Google Gemini model name (e.g., gemini-1.5-pro-latest)")
    parser.add_argument("--data", '-d', type=str, required=True, help="Path to the MCQ data JSON file")
    parser.add_argument("--output", '-o', type=str, default="results/experiments/semantic_dimension", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment")
    
    args = parser.parse_args()

    experiment = GeminiMCQExperiment(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        exp_name=args.exp_name,
    )
    
    results, results_filename = experiment.run_mcq_experiment()