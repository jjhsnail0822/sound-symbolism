#!/usr/bin/env python3

# python src/experiments/generation/logit_conlang_gen.py -l en -m Qwen/Qwen3-32B --gpu 8 --temperature 0.0 -t -n 10 -s 1
# python src/experiments/generation/logit_conlang_gen.py -l en -m Qwen/Qwen3-4B --gpu 4 --temperature 0.0 -n 10 -s 1

import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
import pprint

from dotenv import load_dotenv
from huggingface_hub import login, model_info
import epitran
from vllm import LLM, SamplingParams
from vllm.distributed import (destroy_distributed_environment, destroy_model_parallel)
from transformers import AutoTokenizer
from openai import OpenAI
from tqdm import tqdm
import torch
import gc
import contextlib

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env.local')
load_dotenv(dotenv_path=env_path)

# 환경 변수 로드
load_dotenv('.env.local')
BASE_DIR = os.getenv('BASE_DIR')

os.environ["HF_HOME"] = os.path.join(script_dir, "../models")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(script_dir, "../models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(script_dir, "../models")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(script_dir, "../models")

# 모델 경로 매핑
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

language_code = {"ko": "kor-Hang", "en": "eng-Latn", "fr": "fra-Latn", "ja": "jpn-Hrgn"}

error_handling_prompt = {
    "character_check": {
        "en": "The word must be generated using the specified characters.",
        "fr": "Le mot doit être généré en utilisant les caractères spécifiés.",
        "ja": "この単語は指定された文字を使用して生成する必要があります。",
        "ko": "지정된 문자로 인공어휘를 생성해야 합니다."
    },
    "existing_word_check": {
        "en": "The word is already in the source data.",
        "fr": "Le mot existe déjà dans les données source.",
        "ja": "この単語はすでにソースデータに存在します。",
        "ko": "이 단어는 이미 소스 데이터에 존재합니다."
    },
    "exist_check": {
        "en": "The word is not generated.",
        "fr": "Le mot n'est pas généré.",
        "ja": "この単語は生成されていません。",
        "ko": "단어가 생성되지 않았습니다."
    }
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

class LogitConlangGenerator:
    def __init__(
        self,
        model_name: str,
        data_path:os.path, prompt_path:os.path, output_dir:os.path,
        use_api:bool=False,
        samples:int=10, word_nums:int=10, top_k:int=5,
        tensor_parallel_size:int=4,
        max_tokens:int=512, max_model_len:int=4096,
        temperature:float=0.0, thinking:bool=False,
        language:str="ko"
    ):
        
        self.model_name = MODEL_PATHS.get(model_name, model_name)
        data_base_path = data_path
        self.language = language
        self.output_dir = output_dir
        self.prompt_path = prompt_path
        self.use_api = use_api
        self.temperature = temperature
        self.thinking = thinking
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens if 'Qwen3' not in model_name else max_model_len
        self.max_model_len = max_model_len
        self.top_k = top_k
        self.word_nums = word_nums
        self.samples = samples
        self.data_path = os.path.join(data_base_path, f"{language}.json")
        
        env_path = Path('.env.local')
        load_dotenv(dotenv_path=env_path)
        
        # vLLM 설정
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        
        if self.use_api:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        
        # 결과 저장 경로
        self.logit_output_dir = Path(f"{BASE_DIR}/sound-symbolism/data/processed/art/logits")
        self.logit_output_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_data(self):
        """소스 데이터와 프롬프트 로드"""
        # 소스 데이터 로드
        data_path = Path(f"{BASE_DIR}/sound-symbolism/data/processed/nat/{self.language}.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.source_data = json.load(f)
            self.words = [item["word"] for item in self.source_data]
        
        # 프롬프트 로드
        prompt_path = Path(f"{BASE_DIR}/sound-symbolism/data/prompts/prompts.json")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            self.prompt_templates:dict[str, dict[str, str]] = prompts["generation"]

    def _convert_to_ipa(self, word: str) -> str:
        """Safe IPA conversion"""
        try:
            word = word.strip().lower()
            if not word or not word.isalpha():
                return ""
            
            ipa = self.epi.transliterate(word)
            return ipa
        except (IndexError, ValueError, AttributeError) as e:
            print(f"[WARNING] IPA conversion failed for '{word}': {e}")
            return ""
        except Exception as e:
            print(f"[ERROR] Unexpected error for '{word}': {e}")
            return ""

    def load_model(self):
        """vLLM 모델 로드"""
        print(f"🔄 모델 로드 중: {self.model_name}")
        
        # vLLM 모델 초기화
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            trust_remote_code=True
        )
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.epi:epitran.Epitran = epitran.Epitran(language_code[self.language])
        
        print("✅ 모델 로드 완료")

    def generate_with_logits(self, prompt: str) -> tuple[str, np.ndarray, List[Dict]]:
        """Generate word and get logits"""
        # Set basic generation parameters
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=self.top_k,
            prompt_logprobs=self.top_k,
            stop=['<end_of_turn>', '</s>', '<|endoftext|>']
        )
        
        # Set prompt format for each model
        if 'Qwen3' in self.model_name:
            conversation = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking
            )
        elif 'gemma-3' in self.model_name:
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted_prompt = prompt
        
        # Generate text and collect logits
        outputs = self.model.generate(formatted_prompt, sampling_params)
        # breakpoint()
        generated_text = outputs[0].outputs[0].text.strip().strip("`")
        
        # Collect logit information
        logits_list = []
        token_logprobs_list = []
        
        for output in outputs[0].outputs:
            # Collect entire logit matrix
            if hasattr(output, 'logits'):
                logits_list.append(output.logits)
            
            # Collect top k logit information for each token
            if hasattr(output, 'logprobs') and output.logprobs:
                token_info = {}
                for token_id, logprob_data in output.logprobs[0].items():
                    token_info[token_id] = {
                        'token': logprob_data.decoded_token,
                        'token_id': token_id,
                        'logprob': logprob_data.logprob,
                        'prob': np.exp(logprob_data.logprob),
                        'rank': len(token_info) + 1
                    }
                token_logprobs_list.append(token_info)
        
        if '`' in generated_text:
            word = generated_text.split('`')[1].strip().lower()
        else:
            # If there is no backtick, try to extract word from entire text
            word = generated_text.strip().lower()
            # Additional preprocessing if needed (e.g., removing special characters)
        
        breakpoint()
        ipa = self._convert_to_ipa(word)
        word_info = {'word': word, 'ipa': ipa}

        # Convert logit matrix and normalize
        logits_matrix = np.array(logits_list)
        if logits_matrix.size > 0:
            # Apply softmax to convert to probabilities
            logits_matrix = np.exp(logits_matrix) / np.sum(np.exp(logits_matrix), axis=-1, keepdims=True)
        
        return word_info, logits_matrix, token_logprobs_list
    
    def _error_check(self, word_info:dict, logits:np.ndarray, token_logprobs:List[Dict]) -> None:
        """Check if 
        (1) the characters align with self.language
        (2) the logits and token_logprobs are not empty
        """
        character_check = True
        logit_check = True
        existing_word_check = True
        exist_check = True
        
        if len(word_info['word']) == 0 or logits.size == 0:
            exist_check = False

        if len(token_logprobs) == 0:
            logit_check = False

        if word_info['word'] in self.words:
            existing_word_check = False
            
        # Remove backticks, special characters like '/"\~- using regex
        word_info["word"] = re.sub(r'`|[/\"~\]', '', word_info["word"])
        if self.language == "ko":
            if not all('가' <= char <= '힣' or char.isspace() for char in word_info["word"]):
                character_check = False
        elif self.language == "en":
            if not all(char.isascii() and char.isalpha() or char.isspace() for char in word_info["word"]):
                character_check = False
        elif self.language == "fr":
            if not all((char.isascii() and char.isalpha() or char in "éèêëàâäôöûüç" or char.isspace()) for char in word_info["word"]):
                character_check = False
        elif self.language == "ja":
            if not all('\u3040' <= char <= '\u30ff' or char.isspace() for char in word_info["word"]):
                character_check = False
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        
        return character_check, logit_check, existing_word_check, exist_check
    
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
    
    def run(self, max_samples: Optional[int] = None):
        """전체 실행 프로세스"""
        print("Starting run process...")  # 디버그용 출력 추가
        
        # 데이터와 모델 로드
        self.load_data()
        self.load_model()
        
        results = []
        processed = 0
        prompt_keys = self.prompt_templates.keys()
        
        print(f"Processing {len(self.source_data)} items with prompts: {list(prompt_keys)}")  # 디버그용 출력
        
        for key in prompt_keys:
            blank_prompt:str = self.prompt_templates[key][self.language]["user_prompt"]
            print(f"\nUsing prompt template: {key}")  # 디버그용 출력
            
            for i, item in enumerate(tqdm(self.source_data)):
                if self.word_nums > 0 and i >= self.word_nums:
                    print(f"\nReached word limit: {self.word_nums}")
                    break
                
                if max_samples and processed >= max_samples:
                    print(f"\nReached sample limit: {max_samples}")
                    break
                
                num_trials = 0
                word = item["word"]
                character_check, logit_check, existing_word_check, exist_check = True, True, True, True
                print("trials initialized")
                # breakpoint()
                while num_trials < 3:
                    try:
                        # Extract meaning
                        if self.language == "ko":
                            definitions:list[str] = item["definitions"]
                            # breakpoint()
                            meaning = definitions[0].strip(".")
                        else:
                            meaning = item["meaning"][0].strip(".")
                        
                        # Create prompt and word
                        prompt = blank_prompt.format(meaning=meaning)
                        character_check_prompt = "\n" + error_handling_prompt["character_check"][self.language] if not character_check else ""
                        existing_word_check_prompt = "\n" + error_handling_prompt["existing_word_check"][self.language] if not existing_word_check else ""
                        exist_check_prompt = "\n" + error_handling_prompt["exist_check"][self.language] if not exist_check else ""
                        prompt = prompt + character_check_prompt + existing_word_check_prompt + exist_check_prompt
                        
                        word_info, logits, token_logprobs = self.generate_with_logits(prompt)
                        
                        character_check, logit_check, existing_word_check, exist_check = self._error_check(word_info, logits, token_logprobs)
                        
                        if not (character_check and logit_check and existing_word_check and exist_check):
                            print(f"🔄 Word generation failed: {word_info['word']}")
                            num_trials += 1
                            continue
                        
                        if len(word_info['word']) > 0 and logits.size > 0:
                            results.append({
                                "original_word": item["word"],
                                "meaning": item["meaning"],
                                "generated_word": word_info["word"],
                                "ipa": word_info["ipa"],
                                "logits_matrix": logits,
                                "token_logprobs": token_logprobs,
                                "model": self.model_name,
                                "trial": key
                            })
                            processed += 1
                        print("results appended")
                        break
                    except Exception as e:
                        num_trials += 1
                        print(f"🔄 단어 생성 실패: {e}")
                        breakpoint()
                    finally:
                        gc.collect()
                        torch.cuda.empty_cache()

        if not self.use_api:
            del model
            self._cleanup()
        final_results = self.save_results(results)
        
        # 결과 저장
        if results:
            output_file = self.logit_output_dir / f"{self.language}_logit_{self.model_name.replace('/', '-')}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"\n✅ Saved results to: {output_file}")
            print(f"Total processed items: {len(results)}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate conlang and analyze logits')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], help='Language code')
    parser.add_argument('--model', '-m', required=True, help='Model name')
    parser.add_argument('--gpu', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--output", '-o', type=str, default="data/processed/art/", help="Directory to save results")
    parser.add_argument("--data", '-d', default= "data/processed/nat/",type=str, help="Path to the preprocessed data JSON file")
    parser.add_argument("--prompt", '-p', default= "data/prompts/prompts.json",type=str, help="Path to the prompt JSON file")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api", action='store_true', help="Use OpenAI API instead of local model")
    parser.add_argument("--thinking", '-t', action='store_true', help="Enable thinking mode for Qwen3")
    parser.add_argument("--word_nums", '-n', type=int, default=10, help="Number of words to generate")
    parser.add_argument("--samples", '-s', type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--top_k", '-k', type=int, default=3, help="Number of top k logits to return")
    args = parser.parse_args()
    
    generator = LogitConlangGenerator(
        language=args.language,
        model_name=args.model,
        tensor_parallel_size=args.gpu,
        data_path=args.data,
        prompt_path=args.prompt,
        output_dir=args.output,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        thinking=args.thinking,
        word_nums=args.word_nums,
        samples=args.samples,
        top_k=args.top_k
    )
    
    generator.run(args.samples)

if __name__ == "__main__":
    main()
