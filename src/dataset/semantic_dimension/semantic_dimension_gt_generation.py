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

prompt = """You are a professional linguistic annotator.
Please read a {language} mimetic word and its meaning, and decide which semantic feature best describes the word's meaning.

[WORD]
{word}

[MEANING]
{meaning}

[SEMANTIC DIMENSION]
{dimension1} vs {dimension2}

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
            )
        
        # Run generation
        print(f"Running generation...")
        if self.thinking:
            print("Thinking mode enabled for Qwen3.")
        all_results = {}

        for lang in langs:
            all_results[lang] = []
            for word in tqdm(self.word_data[lang], desc=f"Processing {lang} data"):
                dimension_results = {}
                for dimension in dimensions:
                    if self.use_api:
                        # Use OpenAI API
                        response = self.client.responses.create(
                            model=self.model_path,
                            input=prompt.format(
                                word=word['word'],
                                meaning=word['meaning'],
                                language=languages[lang],
                                dimension1=dimension[0],
                                dimension2=dimension[1],
                            ),
                            temperature=self.temperature,
                        )
                        model_answer = response.output_text
                    else:
                        # Use VLLM model
                        # response = model.generate(query['question'], sampling_params=sampling_params)
                        conversation = [
                            {
                                "role": "user",
                                "content": prompt.format(
                                    word=word['word'],
                                    meaning=word['meaning'],
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
                        print(f"Warning: Model output is empty for word '{word['word']}' in {lang}. Marking as 0.")
                        model_answer = "0"
                    
                    if model_answer == "1":
                        model_answer = dimension[0]
                    elif model_answer == "2":
                        model_answer = dimension[1]
                    else:
                        model_answer = "Neither"

                    dimension_results[f"{dimension[0]}-{dimension[1]}"] = model_answer

                    # print(f"Word: {word['word']}, Meaning: {word['meaning']}, Dimension: {dimension[0]}-{dimension[1]}, Answer: {model_answer}")
                
                all_results[lang].append({
                    "word": word['word'],
                    "meaning": word['meaning'],
                    "en_meaning": word['en_meaning'],
                    "ipa": word['ipa'],
                    "dimensions": dimension_results,
                })

        # Save results
        model_name = os.path.basename(self.model_path)
        results_filename = f"{self.output_dir}/semantic_dimension_gt_{model_name}{'-thinking' if self.thinking else ''}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

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
    )
    mcq_experiment.run_generation()