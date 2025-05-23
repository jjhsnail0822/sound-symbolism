#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pickle
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# 환경 변수 로드
load_dotenv('.env.local')
BASE_DIR = os.getenv('BASE_DIR')

class LogitChecker:
    def __init__(self, 
                 language: str,
                 model_type: str,
                 trial_num: str,
                 model_name: str,
                 tensor_parallel_size: int = 1,
                 max_tokens: int = 32,
                 max_model_len: int = 4096):
        """
        Args:
            language: 언어 코드 (en/fr/ko/ja)
            model_type: 모델 타입 (local/openai)
            trial_num: 프롬프트 트라이얼 번호
            model_name: 모델 이름 또는 경로
            tensor_parallel_size: GPU 병렬 처리 크기
            max_tokens: 최대 생성 토큰 수
            max_model_len: 최대 모델 입력 길이
        """
        self.language = language
        self.model_type = model_type
        self.trial_num = trial_num
        self.model_name = model_name
        
        # vLLM 설정
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        
        # 모델과 토크나이저
        self.model = None
        self.tokenizer = None
        
        # 프롬프트와 데이터
        self.prompt_template = None
        self.source_data = None
        
        # 결과 저장 경로
        self.output_dir = Path(f"{BASE_DIR}/sound-symbolism/data/processed/art/logits")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_prompt(self):
        with open(f"{BASE_DIR}/sound-symbolism/analysis/experiments/prompts.json", "r", encoding="utf-8") as f:
            prompts = json.load(f)
        self.prompt_template = prompts["generation"][self.trial_num][self.language]["user_prompt"]

    def load_source_data(self):
        data_file = f"{BASE_DIR}/sound-symbolism/dataset/1_preprocess/nat/{self.language}_data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            self.source_data = json.load(f)

    def load_model(self):
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

    def get_logits_for_generation(self, prompt: str) -> tuple[str, np.ndarray]:
        # 생성 파라미터 설정
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.max_tokens,
            return_logits=True  # logit 값 반환 설정
        )
        
        # 텍스트 생성 및 logit 획득
        outputs = self.model.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # logit 행렬 구성
        logits = []
        for output in outputs[0].outputs:
            if hasattr(output, 'logits'):
                logits.append(output.logits)
        
        logits_matrix = np.array(logits)
        
        # 생성된 단어 추출
        if '`' in generated_text:
            word = generated_text.split('`')[1]
        else:
            word = generated_text
        
        return word, logits_matrix

    def run(self, max_samples: Optional[int] = None):
        # 필요한 데이터 로드
        if not all([
            self.load_prompt(),
            self.load_source_data(),
            self.load_model()
        ]):
            return
        
        # 데이터 준비
        meanings = [(item.get('definition', ''), item.get('word', '')) 
                   for item in self.source_data]
        
        if max_samples:
            meanings = meanings[:max_samples]
        
        results = []
        
        # 각 의미에 대해 단어 생성 및 logit 계산
        for meaning, original_word in tqdm(meanings, desc="단어 생성 및 logit 계산"):
            prompt = self.prompt_template.format(meaning=meaning)
            word, logits = self.get_logits_for_generation(prompt)
            
            if len(word) > 0 and logits.size > 0:
                results.append({
                    'original_word': original_word,
                    'meaning': meaning,
                    'generated_word': word,
                    'logits': logits
                })
        
        # 결과 저장
        if results:
            output_file = self.output_dir / f"logits_{self.language}_{self.trial_num}_{self.model_name.replace('/', '-')}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"✅ Logit 결과 저장 완료: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='단어 생성 logit 분석')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], help='언어 코드')
    parser.add_argument('--model', '-m', required=True, help='모델 이름 또는 경로')
    parser.add_argument('--trial', '-t', required=True, help='프롬프트 트라이얼 번호')
    parser.add_argument('--gpu', type=int, default=1, help='사용할 GPU 수')
    parser.add_argument('--samples', '-n', type=int, help='처리할 샘플 수')
    
    args = parser.parse_args()
    
    checker = LogitChecker(
        language=args.language,
        model_type='local',
        trial_num=args.trial,
        model_name=args.model,
        tensor_parallel_size=args.gpu
    )
    
    checker.run(args.samples)

if __name__ == "__main__":
    main()
