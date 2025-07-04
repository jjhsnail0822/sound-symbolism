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
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path
import math

dimensions = [
    ('good', 'bad'),
    ('beautiful', 'ugly'),
    ('pleasant', 'unpleasant'),
    ('strong', 'weak'),
    ('big', 'small'),
    ('rugged', 'delicate'),
    ('active', 'passive'),
    ('fast', 'slow'),
    ('sharp', 'round'),
    ('realistic', 'fantastical'),
    ('structured', 'disorganized'),
    ('ordinary', 'unique'),
    ('interesting', 'uninteresting'),
    ('simple', 'complex'),
    ('abrupt', 'continuous'),
    ('exciting', 'calming'),
    ('hard', 'soft'),
    ('happy', 'sad'),
    ('harsh', 'mellow'),
    ('heavy', 'light'),
    ('inhibited', 'free'),
    ('masculine', 'feminine'),
    ('solid', 'nonsolid'),
    ('tense', 'relaxed'),
    ('dangerous', 'safe')
]

langs = ['en', 'fr', 'ja', 'ko']
languages = {
    'en': 'English',
    'fr': 'French',
    'ja': 'Japanese',
    'ko': 'Korean'
}
TOP_K = 20 # Number of top logits to return

prompt = """You are a professional linguistic annotator.
Please read a {language} mimetic word and its meaning, and decide which semantic feature best describes the word's meaning.

[WORD]
{word}

[MEANING]
{meaning}

[SEMANTIC DIMENSION]
{dimension1} vs. {dimension2}

[OPTIONS]
1: {dimension1}
2: {dimension2}
3: Neither
Answer with the number only. (1-3)
"""

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
            save_logits: bool = False,
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
        self.save_logits = save_logits

        # Load environment variables from .env.local file
        env_path = Path('.env.local')
        load_dotenv(dotenv_path=env_path)

        if self.use_api:
            if 'gemini' in self.model_path:
                # Configure Google Generative AI
                self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                if self.save_logits:
                    print("Warning: save_logits is not supported for Gemini models. Disabling it.")
                    self.save_logits = False
            else:
                # Get API key from environment variables
                self.client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                )

        self.word_data = {}
        for lang in langs:
            with open(self.data_path.format(language=lang), 'r', encoding='utf-8') as f:
                self.word_data[lang] = json.load(f)

        pass

    def _cleanup(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()

    def run_generation(self):
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
                logprobs=TOP_K,
            )
        
        # Run generation
        print(f"Running generation...")
        if self.thinking:
            print("Thinking mode enabled for Qwen3.")
        
        # Prepare output file path
        model_name = os.path.basename(self.model_path)
        results_filename = f"{self.output_dir}/semantic_dimension_gt_{model_name}"
        if self.thinking:
            results_filename += "_thinking"
        if self.save_logits:
            results_filename += "_logits"
        results_filename += ".json"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load existing results if file exists
        all_results = {}
        if os.path.exists(results_filename):
            try:
                with open(results_filename, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                print(f"Loaded existing results from {results_filename}")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Could not load existing results, starting fresh")
                all_results = {}

        for lang in langs:
            if lang not in all_results:
                all_results[lang] = []

            # Create a set of already processed words for this language
            processed_words = {result['word'] for result in all_results[lang]}
            
            word_count = len(all_results[lang])  # Start from number of already processed words
            
            processed_in_this_run = 0
            for word in tqdm(self.word_data[lang], desc=f"Processing {lang} data"):
                # if processed_in_this_run >= 1:
                #     break
                # Skip if this word is already processed in this language
                if word['word'] in processed_words:
                    continue
                    
                dimension_results = {}
                for dimension in dimensions:
                    if self.use_api:
                        # Use OpenAI API
                        if 'gemini' in self.model_path:
                            content = prompt.format(
                                word=word['word'],
                                meaning=word['meaning'] if isinstance(word['meaning'], str) else word['meaning'][0],
                                language=languages[lang],
                                dimension1=dimension[0],
                                dimension2=dimension[1],
                            )
                            generation_config = types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(
                                    thinking_budget=0
                                ),
                                maxOutputTokens=self.max_tokens,
                                temperature=self.temperature,
                            )
                            try:
                                response = self.client.models.generate_content(
                                    model=self.model_path,
                                    contents=content,
                                    config=generation_config,
                                )
                                model_answer = response.text.strip()
                            except Exception as e:
                                print(f"An error occurred while calling the Gemini API: {e}")
                                model_answer = "" # Set empty answer on error
                            logits = None # Logits not supported by Gemini API
                        else:
                            response = self.client.chat.completions.create(
                                model=self.model_path,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt.format(
                                                    word=word['word'],
                                                    meaning=word['meaning'] if isinstance(word['meaning'], str) else word['meaning'][0],
                                                    language=languages[lang],
                                                    dimension1=dimension[0],
                                                    dimension2=dimension[1],
                                                ),
                                    }
                                ],
                                temperature=self.temperature,
                                logprobs=True if self.save_logits else False,
                                top_logprobs=TOP_K,
                            )
                            model_answer = response.choices[0].message.content

                            if self.save_logits:
                                probabilities_for_123 = {'1': 0.0, '2': 0.0, '3': 0.0}

                                first_token_raw_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                                # print(first_token_raw_logprobs)

                                # Iterate through the logprobs provided by OpenAI for the first token
                                for token in first_token_raw_logprobs:
                                    decoded_token_str = token.token.strip()
                                    # If this decoded token is one of our targets ('1', '2', '3')
                                    if decoded_token_str in probabilities_for_123:
                                        probabilities_for_123[decoded_token_str] = math.exp(token.logprob)

                                logits = probabilities_for_123  # Store the calculated probabilities
                                # print(f"Calculated probabilities for 1,2,3: {logits}")
                    else:
                        # Use VLLM model
                        # response = model.generate(query['question'], sampling_params=sampling_params)
                        conversation = [
                            {
                                "role": "user",
                                "content": prompt.format(
                                    word=word['word'],
                                    meaning=word['meaning'] if isinstance(word['meaning'], str) else word['meaning'][0],
                                    language=languages[lang],
                                    dimension1=dimension[0],
                                    dimension2=dimension[1],
                                )
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
                        # Remove <think> ... </think> tags
                        if self.thinking:
                            model_answer = re.sub(r'<think>.*?</think>', '', model_answer, flags=re.DOTALL)
                        
                        if self.save_logits:
                            probabilities_for_123 = {'1': 0.0, '2': 0.0, '3': 0.0}
                        
                            first_token_raw_logprobs = response[0].outputs[0].logprobs[0] 
                            
                            # Iterate through the logprobs provided by VLLM for the first token
                            for token_id, logprob_obj in first_token_raw_logprobs.items():
                                decoded_token_str = str(logprob_obj.decoded_token).strip()
                                # If this decoded token is one of our targets ('1', '2', '3')
                                if decoded_token_str in probabilities_for_123:
                                    probabilities_for_123[decoded_token_str] = math.exp(logprob_obj.logprob)
                        
                            logits = probabilities_for_123 # Store the calculated probabilities
                            # print(f"Calculated probabilities for 1,2,3: {logits}")

                    # First integer is the answer
                    model_answer = re.search(r'\d+', model_answer)
                    if model_answer:
                        model_answer = model_answer.group(0)
                    else:
                        model_answer = None
                    # Handle cases where the model output is empty or None
                    if model_answer is None:
                        print(f"Warning: Model output is empty for word '{word['word']}' in {lang}. Marking as 0.")
                        model_answer = "0"
                    
                    if model_answer == "1":
                        model_answer = dimension[0]
                    elif model_answer == "2":
                        model_answer = dimension[1]
                    else:
                        model_answer = "Neither"

                    if self.save_logits:
                        dimension_results[f"{dimension[0]}-{dimension[1]}"] = {'answer': model_answer,
                                                                               'logits': logits}
                    else:
                        dimension_results[f"{dimension[0]}-{dimension[1]}"] = model_answer

                    # print(f"Word: {word['word']}, Meaning: {word['meaning']}, Dimension: {dimension[0]}-{dimension[1]}, Answer: {model_answer}")
                
                # Add the completed word result to all_results
                word_result = {
                    "word": word['word'],
                    "meaning": word['meaning'],
                    "en_meaning": word['en_meaning'],
                    "ipa": word['ipa'],
                    "dimensions": dimension_results,
                }
                all_results[lang].append(word_result)
                word_count += 1
                processed_in_this_run += 1
                
                # Save intermediate results every 10 words instead of every word
                if word_count % 10 == 0:
                    try:
                        with open(results_filename, 'w', encoding='utf-8') as f:
                            json.dump(all_results, f, ensure_ascii=False, indent=4)
                        print(f"Saved intermediate results after {word_count} words in {lang}")
                    except Exception as e:
                        print(f"Warning: Could not save intermediate results: {e}")

        # Final save
        try:
            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            print(f"Final results saved to: {results_filename}")
        except Exception as e:
            print(f"Error saving final results: {e}")

        if not self.use_api:
            del model
            self._cleanup()

        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generation")
    parser.add_argument("--model", '-m', type=str, required=True, help="Path to the LLM")
    parser.add_argument("--gpu", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api", action='store_true', help="Use OpenAI API instead of local model")
    parser.add_argument("--thinking", action='store_true', help="Enable thinking mode for Qwen3")
    parser.add_argument("--save-logits", action='store_true', help="Enable logits output")
    
    args = parser.parse_args()

    mcq_experiment = MCQExperiment(
        model_path=args.model,
        data_path='data/processed/nat/{language}.json',
        output_dir='data/processed/nat/semantic_dimension',
        use_api=args.api,
        tensor_parallel_size=args.gpu,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        thinking=args.thinking,
        save_logits=args.save_logits
    )
    mcq_experiment.run_generation()