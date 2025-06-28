# python pseudo_word_generationv2.py -m gpt-4o --gpu 1 --api
# python src/experiments/generation/pseudo_word_generationv2.py -m google/gemma-3-1b-it --gpu 4 -l ja
# python src/experiments/generation/pseudo_word_generationv2.py -m Qwen/Qwen3-8B --gpu 4 -l ja --thinking
# python src/experiments/generation/pseudo_word_generationv2.py -m Qwen/Qwen3-4B -l ko --gpu 4
# sbatch -p big_suma_rtx3090 -q big_qos 

import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import re
import contextlib
import gc
import pprint

import pandas as pd
import torch
from typing import List, Dict, Any, Optional
from huggingface_hub import login, model_info
from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel)
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# 환경변수 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env.local')
load_dotenv(dotenv_path=env_path)

os.environ["HF_HOME"] = os.path.join(script_dir, "../models")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(script_dir, "../models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(script_dir, "../models")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(script_dir, "../models")
# breakpoint()

# 모델 경로 매핑 추가
MODEL_PATHS = {
    "google/gemma-3-27b-it": "google/gemma-3-27b-it",
    "google/gemma-3-12b-it": "google/gemma-3-12b-it",
    "google/gemma-3-4b-it": "google/gemma-3-4b-it",
    "google/gemma-3-1b-it": "google/gemma-3-1b-it",
    "Qwen/Qwen3-4B": "Qwen/Qwen3-4b",
    "Qwen/Qwen3-8B": "Qwen/Qwen3-8b",
    "Qwen/Qwen3-14B": "Qwen/Qwen3-14b",
    "Qwen/Qwen3-32B": "Qwen/Qwen3-32b",
}
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def hf_login() -> bool:
    if HUGGINGFACE_TOKEN:
        try:
            login(token=HUGGINGFACE_TOKEN)
            print("✅ Hugging Face에 로그인했습니다.")
            return True
        except Exception as e:
            print(f"⚠️ Hugging Face 로그인 실패: {e}")
    return False

class pseudoWordGeneration:
    def __init__(
            self,
            model_path:str,
            data_path:str,
            prompt_path:str,
            output_dir:str,
            use_api:bool=False,
            tensor_parallel_size:int=1,
            max_tokens:int=512,
            max_model_len:int=4096,
            temperature:float=0.0,
            thinking:bool=False,
            word_nums:int=10,
            language:str="ko",
    ):
        self.model_path = MODEL_PATHS.get(model_path, model_path)  # 매핑된 경로가 있으면 사용
        data_base_path = data_path
        self.output_dir = output_dir
        self.prompt_path = prompt_path
        self.use_api = use_api
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens if 'Qwen3' not in model_path else max_model_len # for thinking mode
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.thinking = thinking
        self.word_nums = word_nums
        self.language = language
        self.data_path = os.path.join(data_base_path, f"{language}.json")
        # Load environment variables from .env.local file
        env_path = Path('.env.local')
        load_dotenv(dotenv_path=env_path)

        # Get API key from environment variables
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def _cleanup(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()

    def run_word_gen(self):
        # Load MCQ data
        print(f"Loading {self.language} data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            word_data = json.load(f)
        print(f"Loaded {len(word_data)} words.")
        
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            prompts:dict = json.load(f)
        
        prompts = prompts["generation"]
        # Initialize VLLM model
        if self.use_api:
            print(f"Using OpenAI API with model {self.model_path}")
        else:
            hf_login()
            print(f"Loading model from {self.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                download_dir="../models"
            )
            sampling_params = SamplingParams(
                temperature=self.temperature, 
                max_tokens=self.max_tokens,
            )
        
        # Run experiment
        print(f"Generating {self.word_nums} words on {len(word_data)} words...")
        if self.thinking:
            print("Thinking mode enabled for Qwen3.")
        all_results = []
        
        prompt_keys = prompts.keys()
        if 'Qwen3' in self.model_path and self.thinking == True:
            self.model_name = self.model_path + "-thinking"
        else:
            self.model_name = self.model_path
            
        # Process in batches
        for key in prompt_keys:
            print(f"Generating with {key} as prompt")
            # for i, word_item in enumerate(tqdm(word_data)):
            for i, word_item in enumerate(word_data):
                prompt:str = prompts[key][self.language]["user_prompt"]
                if self.word_nums > 0 and i >= self.word_nums:
                    break
                num_trials = 0
                
                word = word_item["word"]
                if self.language == "ko":
                    definitions = word_item["definitions"]
                    meaning = definitions[0][:-1]
                else:
                    meaning = word_item["meaning"]
                prompt = prompt.format(meaning=meaning)
                # print(f"prompt: {prompt}")
                # breakpoint()
                while num_trials < 3:
                    try:
                        if self.use_api:
                            # Use OpenAI API
                            response = self.client.responses.create(
                                model=self.model_path,
                                input=prompt,
                                # temperature=self.temperature, # not supported with o4-mini
                            )
                            model_answer = response.output_text
                        else:
                            # Use VLLM model
                            # response = model.generate(query['question'], sampling_params=sampling_params)
                            conversation = [
                                {
                                    "role": "user",
                                    "content": prompt
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
                                
                            elif 'gemma-3' in self.model_path:
                                # Gemma 전용 채팅 형식 적용
                                text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                                print(f"prompt: {prompt}")
                                
                                # Gemma 전용 생성 파라미터 설정
                                sampling_params = SamplingParams(
                                    temperature=self.temperature,
                                    max_tokens=self.max_tokens,
                                    stop=['<end_of_turn>'],  # Gemma 전용 stop 토큰
                                )
                                
                                response = model.generate(text, sampling_params)
                                
                                # 응답 처리
                                if response and response[0].outputs:
                                    model_answer = response[0].outputs[0].text.strip()
                                    # Gemma 응답에서 필요한 부분만 추출
                                    if '`' in model_answer:
                                        model_answer = model_answer.split('`')[1]
                                    else:
                                        # 백틱이 없는 경우 전체 응답 사용
                                        model_answer = model_answer
                            else:
                                response = model.chat(conversation, sampling_params=sampling_params)
                            
                            # Extract answer from the first element of the response list
                            model_answer = response[0].outputs[0].text.strip()
                            # remove <think> ... </think> tags
                            if self.thinking:
                                model_answer = re.sub(r'<think>.*?</think>', '', model_answer, flags=re.DOTALL)
                        # breakpoint()
                        print(f"word: {word}, meaning: {meaning}, model_answer: {model_answer}")
                        # First integer is the answer
                        if "`" in model_answer:
                            model_answer = model_answer.split("`")[1]
                        # breakpoint()
                        
                        # Store result
                        all_results.append({
                            "original_word": word,
                            "meaning": meaning,
                            "generated_word": model_answer,
                            "model": self.model_name,
                            "language": self.language,
                            "trial": key
                        })
                        # print(f"✅ {word} -> {model_answer} 생성 완료")
                        break
                    except Exception as e:
                        num_trials += 1
                        print(f"Error: {e}")
                        breakpoint()
                        continue
                self._cleanup()
        
        if not self.use_api:
            del model
            self._cleanup()
        final_results = self.save_results(all_results)
        return final_results
    
    def save_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_file:os.path = os.path.join(self.output_dir, f"{self.language}_pseudo_words.json")
        csv_file:os.path = os.path.join(self.output_dir, f"{self.language}_pseudo_words.csv")
        
        existing_data = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        
        final_results = self.merge_results(existing_data, all_results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON 형식으로 저장 완료: {output_file}")
        
        df = pd.DataFrame(final_results)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ CSV 형식으로도 저장 완료: {csv_file}")
        
        return final_results
    
    def merge_results(self, existing_data: List[Dict[str, Any]], new_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result_dict = {}
        for item in existing_data:
            key = f"{item['original_word']}_{item['meaning']}_{item['model']}_{item['trial'][5:]}"
            result_dict[key] = item
        
        updated_count = 0
        added_count = 0
        
        for item in new_data:
            key = f"{item['original_word']}_{item['meaning']}_{item['model']}_{item['trial'][5:]}"
            
            if key in result_dict:
                old_word = result_dict[key].get("generated_word", "")
                result_dict[key] = item
                updated_count += 1
                print(f"🔄 단어 업데이트: '{old_word}' → '{item['generated_word']}'")
            else:
                result_dict[key] = item
                added_count += 1
        
        print(f"📊 결과 병합 통계: {updated_count}개 업데이트, {added_count}개 추가")
        
        return list(result_dict.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCQ experiment")
    parser.add_argument("--model", '-m', type=str, required=True, help="Path to the LLM")
    parser.add_argument("--data", '-d', default= "data/processed/nat/",type=str, help="Path to the preprocessed data JSON file")
    parser.add_argument("--prompt", '-p', default= "data/prompts/prompts.json",type=str, help="Path to the prompt JSON file")
    parser.add_argument("--gpu", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--output", '-o', type=str, default="data/processed/art/", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api", action='store_true', help="Use OpenAI API instead of local model")
    parser.add_argument("--thinking", '-t', action='store_true', help="Enable thinking mode for Qwen3")
    parser.add_argument("--word_nums", '-n', type=int, default=10, help="Number of words to generate")
    parser.add_argument("--language", '-l', type=str, default="ko", help="Language of the data")
    
    args = parser.parse_args()

    word_gen = pseudoWordGeneration(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        prompt_path=args.prompt,
        use_api=args.api,
        tensor_parallel_size=args.gpu,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        thinking=args.thinking,
        word_nums=args.word_nums,
        language=args.language,
    )
    results_filename = word_gen.run_word_gen()
