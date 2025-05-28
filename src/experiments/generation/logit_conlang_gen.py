#!/usr/bin/env python3

# python src/experiments/generation/logit_conlang_gen.py -l en -m Qwen/Qwen3-4B --gpu 4 -t 0.0 --thinking -n 10 -s 1
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pickle
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import gc
import contextlib

# 환경 변수 로드
load_dotenv('.env.local')
BASE_DIR = os.getenv('BASE_DIR')

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

class LogitConlangGenerator:
    def __init__(self,
                 language: str,
                 model_name: str,
                 trial_num: str,
                 tensor_parallel_size: int = 1,
                 max_tokens: int = 32,
                 max_model_len: int = 4096,
                 temperature: float = 0.0,
                 thinking: bool = False,
                 top_k: int = 5):
        
        self.language = language
        self.model_name = MODEL_PATHS.get(model_name, model_name)
        self.trial_num = trial_num
        self.temperature = temperature
        self.thinking = thinking
        self.top_k = top_k
        
        # vLLM 설정
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        
        # 모델과 토크나이저
        self.model = None
        self.tokenizer = None
        
        # 데이터와 프롬프트
        self.source_data = None
        self.prompt_template = None
        
        # 결과 저장 경로
        self.output_dir = Path(f"{BASE_DIR}/sound-symbolism/data/processed/art/logits")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """소스 데이터와 프롬프트 로드"""
        # 소스 데이터 로드
        data_path = Path(f"{BASE_DIR}/sound-symbolism/dataset/1_preprocess/nat/{self.language}_data.json")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.source_data = json.load(f)
        
        # 프롬프트 로드
        prompt_path = Path(f"{BASE_DIR}/sound-symbolism/analysis/experiments/prompts.json")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            self.prompt_template = prompts["generation"][self.trial_num][self.language]["user_prompt"]

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
        
        print("✅ 모델 로드 완료")

    def generate_with_logits(self, prompt: str) -> tuple[str, np.ndarray, List[Dict]]:
        """단어 생성 및 logit 획득"""
        # 생성 파라미터 설정
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            return_logits=True,
            logprobs=self.top_k,  # top k개의 logit 값 반환
            prompt_logprobs=0
        )
        
        # 채팅 형식 설정
        if 'Qwen3' in self.model_name:
            conversation = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking
            )
        elif 'gemma-3' in self.model_name:
            prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # 텍스트 생성
        outputs = self.model.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # logit 행렬과 토큰별 상위 logit 값 수집
        logits = []
        token_logprobs = []
        
        for output in outputs[0].outputs:
            # 전체 logit 행렬 저장
            if hasattr(output, 'logits'):
                logits.append(output.logits)
            
            # 각 토큰의 상위 k개 logit 값 저장
            if hasattr(output, 'logprobs'):
                token_info = {}
                for token_id, logprob_data in output.logprobs[0].items():
                    token_info[token_id] = {
                        'token': logprob_data.decoded_token,
                        'logprob': logprob_data.logprob,
                        'prob': np.exp(logprob_data.logprob)
                    }
                token_logprobs.append(token_info)
        
        logits_matrix = np.array(logits)
        
        # 생성된 단어 추출
        if '`' in generated_text:
            word = generated_text.split('`')[1]
        else:
            word = generated_text
        
        return word, logits_matrix, token_logprobs

    def run(self, max_samples: Optional[int] = None):
        """전체 실행 프로세스"""
        if not all([self.load_data(), self.load_model()]):
            return
        
        results = []
        processed = 0
        for item in tqdm(self.source_data):
            if max_samples and processed >= max_samples:
                break
            
            # 의미 추출
            if self.language == "ko":
                meaning = item["definitions"][0][:-1]
            else:
                meaning = item["meaning"]
            
            # 프롬프트 생성 및 단어 생성
            prompt = self.prompt_template.format(meaning=meaning)
            word, logits, token_logprobs = self.generate_with_logits(prompt)
            
            if len(word) > 0 and logits.size > 0:
                results.append({
                    'original_word': item.get('word', ''),
                    'meaning': meaning,
                    'generated_word': word,
                    'logits_matrix': logits,  # 전체 logit 행렬
                    'token_logprobs': token_logprobs,  # 토큰별 상위 k개 logit 값
                    'model': self.model_name,
                    'trial': self.trial_num
                })
                processed += 1
            
            # 메모리 정리
            if processed % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # 결과 저장
        if results:
            output_file = self.output_dir / f"logits_{self.language}_{self.trial_num}_{self.model_name.replace('/', '-')}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"✅ 결과 저장 완료: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Conlang 생성 및 Logit 분석')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], help='언어 코드')
    parser.add_argument('--model', '-m', required=True, help='모델 이름')
    parser.add_argument('--gpu', type=int, default=4, help='사용할 GPU 수')
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--api", action='store_true', help="Use OpenAI API instead of local model")
    parser.add_argument("--thinking", '-t', action='store_true', help="Enable thinking mode for Qwen3")
    parser.add_argument("--word_nums", '-n', type=int, default=10, help="Number of words to generate")
    parser.add_argument("--samples", '-s', type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()
    
    generator = LogitConlangGenerator(
        language=args.language,
        model_name=args.model,
        trial_num=args.trial,
        tensor_parallel_size=args.gpu,
        temperature=args.temperature,
        thinking=args.thinking,
        top_k=args.word_nums
    )
    
    generator.run(args.samples)

if __name__ == "__main__":
    main()
