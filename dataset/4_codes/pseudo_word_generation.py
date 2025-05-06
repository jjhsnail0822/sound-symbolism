#!/usr/bin/env python3
# python pseudo_word_generation.py -l ko -m openai --model-name gpt-4o -t trial2 -n 5 -w 2
# python pseudo_word_generation.py -l ko -m local --model-name gemma-3-27b-it -t trial2 -n 100 -w 2
# python pseudo_word_generation.py -l ko -m local --local-model qwen3-4b -t trial10 
# python pseudo_word_generation.py --download-model bloom-560m
# python pseudo_word_generation.py --debug-model gpt2
# python pseudo_word_generation.py -l ko -m openai --model-name gpt-4o --all-trials -n 10
# python pseudo_word_generation.py -l ko -m local --model-name qwen3-14b --all-trials -n 100

import os
import json
import argparse
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any, Optional
from huggingface_hub import login, model_info
import shutil
from openai import OpenAI, AsyncOpenAI
import psutil
import sys

# HuggingFace 모델 로드를 위한 라이브러리
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 환경 변수 로드
load_dotenv('.env.local')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# 사용 가능한 모델 정의 - Hugging Face 모델 ID만 포함
AVAILABLE_MODELS = {
    # Gemma 모델
    "gemma-3-4b-it": {
        "hf_id": "google/gemma-3-4b-it",
        "description": "Google의 Gemma 3 4B Instruction 모델",
        "requires_auth": True
    },
    "gemma-3-12b-it": {
        "hf_id": "google/gemma-3-12b-it", 
        "description": "Google의 Gemma 3 12B Instruction 모델",
        "requires_auth": True
    },
    "gemma-3-27b-it": {
        "hf_id": "google/gemma-3-27b-it",
        "description": "Google의 Gemma 3 27B Instruction 모델",
        "requires_auth": True
    },
    
    # Qwen 2.5 모델
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Qwen의 2.5 3B Instruct 모델",
        "requires_auth": False
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen의 2.5 7B Instruct 모델",
        "requires_auth": False
    },
    "qwen2.5-14b": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen의 2.5 14B Instruct 모델",
        "requires_auth": False
    },
    "qwen2.5-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen의 2.5 32B Instruct 모델",
        "requires_auth": False
    },
    "qwen2.5-72b": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "description": "Qwen의 2.5 72B Instruct 모델",
        "requires_auth": False
    },
    
    # Qwen 3 모델
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "description": "Qwen의 3세대 4B 모델",
        "requires_auth": False
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "description": "Qwen의 3세대 8B 모델",
        "requires_auth": False
    },
    "qwen3-14b": {
        "hf_id": "Qwen/Qwen3-14B",
        "description": "Qwen의 3세대 14B 모델",
        "requires_auth": False
    },
    "qwen3-32b": {
        "hf_id": "Qwen/Qwen3-32B",
        "description": "Qwen의 3세대 32B 모델",
        "requires_auth": False
    },
    
    # Qwen 3 Thinking 모델
    "qwen3-4b-thinking": {
        "hf_id": "Qwen/Qwen3-4B-Thinking",
        "description": "Qwen의 3세대 4B Thinking 모델",
        "requires_auth": False
    },
    "qwen3-8b-thinking": {
        "hf_id": "Qwen/Qwen3-8B-Thinking",
        "description": "Qwen의 3세대 8B Thinking 모델",
        "requires_auth": False
    },
    "qwen3-14b-thinking": {
        "hf_id": "Qwen/Qwen3-14B-Thinking",
        "description": "Qwen의 3세대 14B Thinking 모델",
        "requires_auth": False
    },
    "qwen3-32b-thinking": {
        "hf_id": "Qwen/Qwen3-32B-Thinking",
        "description": "Qwen의 3세대 32B Thinking 모델",
        "requires_auth": False
    },
    
    # OLMo 모델
    "olmo-7b": {
        "hf_id": "allenai/OLMo-2-1124-7B-Instruct",
        "description": "Allen AI의 OLMo 7B Instruct 모델",
        "requires_auth": False
    },
    "olmo-13b": {
        "hf_id": "allenai/OLMo-2-1124-13B-Instruct",
        "description": "Allen AI의 OLMo 13B Instruct 모델",
        "requires_auth": False
    },
    "olmo-32b": {
        "hf_id": "allenai/OLMo-2-0325-32B-Instruct",
        "description": "Allen AI의 OLMo 32B Instruct 모델",
        "requires_auth": False
    }
}

# 사용 가능한 OpenAI 모델 목록
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4-turbo", 
    "gpt-4",
    "gpt-4.1",
    "gpt-3.5-turbo"
]

# Hugging Face 로그인 함수
def hf_login() -> bool:
    if HUGGINGFACE_TOKEN:
        try:
            login(token=HUGGINGFACE_TOKEN)
            print("✅ Hugging Face에 로그인했습니다.")
            return True
        except Exception as e:
            print(f"⚠️ Hugging Face 로그인 실패: {e}")
    return False

# 모델 사용 가능 여부 확인 함수
def check_model_access(model_name: str) -> bool:
    if model_name not in AVAILABLE_MODELS:
        print(f"⚠️ 지원하지 않는 모델입니다: {model_name}")
        return False
    
    model_id = AVAILABLE_MODELS[model_name]["hf_id"]
    print(f"🔍 모델 '{model_id}' 접근성 확인 중...")
    try:
        info = model_info(model_id)
        return True
    except Exception as e:
        print(f"❌ 모델 '{model_id}'에 접근할 수 없습니다: {e}")
        return False

# 모델 디버깅 함수
def debug_model(model_name: str) -> None:
    if model_name not in AVAILABLE_MODELS:
        print(f"❌ 모델 '{model_name}'은(는) AVAILABLE_MODELS에 정의되어 있지 않습니다.")
        return
    
    model_info = AVAILABLE_MODELS[model_name]
    model_id = model_info["hf_id"]
    
    print(f"===== 모델 디버깅 =====")
    print(f"모델 이름: {model_name}")
    print(f"Hugging Face ID: {model_id}")
    print(f"설명: {model_info['description']}")
    
    # 로그인 시도
    hf_login()
    
    # 모델 접근성 확인
    try:
        print("\n🔍 모델 정보 확인 중...")
        info = model_info(model_id)
        print(f"✅ 모델에 접근 가능합니다.")
        print(f"모델 타입: {info.modelId}")
        
        # 파이프라인 생성 테스트
        print("\n🔧 파이프라인 생성 테스트 중...")
        pipe = pipeline("text-generation", model=model_id, max_new_tokens=20)
        test_result = pipe("This is a test:")
        print(f"테스트 결과: {test_result[0]['generated_text']}")
    except Exception as e:
        print(f"❌ 모델에 접근할 수 없습니다: {e}")
    
    print("===== 디버깅 완료 =====")

# 모델 캐시 경로 설정 함수 추가
def setup_model_cache():
    """HuggingFace 모델 캐시 경로 설정"""
    # 사용자 지정 캐시 디렉토리
    custom_cache_dir = "/scratch2/sheepswool/workspace/models"
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(custom_cache_dir):
        try:
            os.makedirs(custom_cache_dir, exist_ok=True)
            print(f"✅ 모델 캐시 디렉토리 생성됨: {custom_cache_dir}")
        except Exception as e:
            print(f"⚠️ 모델 캐시 디렉토리 생성 실패: {e}")
            return False
    
    # 환경 변수 설정
    os.environ["HUGGINGFACE_HUB_CACHE"] = custom_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = custom_cache_dir
    os.environ["HF_HOME"] = custom_cache_dir
    
    # 파일 권한 확인
    try:
        test_file = os.path.join(custom_cache_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("Test write permission")
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"⚠️ 캐시 디렉토리 권한 또는 공간 확인 실패: {e}")
        return False

class PseudoWordGenerator:
    """가상 단어 생성기 클래스"""
    
    def __init__(self, language: str, model: str, trial_num: str, 
                 batch_size: int = 10, output_dir: Optional[str] = None, 
                 local_model: Optional[str] = None):
        """
        가상 단어 생성기 초기화
        
        Args:
            language: 언어 코드 (en/fr/ko/ja)
            model: 모델 타입 (openai, local)
            trial_num: 프롬프트 트라이얼 번호 (trial1, trial2, ...)
            batch_size: 배치 크기
            output_dir: 출력 디렉토리
            local_model: 로컬 모델 이름
        """
        self.language = language.lower()
        self.model_type = model.lower()
        self.trial_num = trial_num
        self.batch_size = batch_size
        self.local_model_name = local_model
        self.model_name = "gpt-4o"  # OpenAI 기본 모델
        self.words_per_meaning = 1
        
        # 로컬 모델 관련 변수
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # OpenAI 클라이언트
        self.client = None
        
        # 출력 디렉토리 설정
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"../../dataset/0_raw/art")
        
        # 프롬프트 템플릿
        self.user_prompt_template = None
        
        # 소스 데이터
        self.source_data = []
    
    def set_trial_num(self, trial_num: str):
        """
        트라이얼 번호 변경
        
        Args:
            trial_num: 새로운 트라이얼 번호
        """
        self.trial_num = trial_num
        # 프롬프트도 다시 로드
        self.user_prompt_template = None
        self.load_prompts()
    
    def load_prompts(self):
        """프롬프트 템플릿 로드"""
        try:
            prompts_path = Path("../../analysis/experiments/prompts.json")

            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
            
            trial_data = prompts_data["generation"][self.trial_num]
            self.user_prompt_template = trial_data[self.language]["user_prompt"]
            print(f"✅ 프롬프트 로드 완료 (트라이얼: {self.trial_num}, 언어: {self.language})")
            return True
            
        except Exception as e:
            traceback.print_exc()
            return False
    
    def load_source_data(self):
        """소스 데이터 로드"""
        # 데이터 파일 경로
        file_path = Path(f"../../dataset/1_preprocess/nat/{self.language}.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.source_data = json.load(f)
        
        print(f"✅ 소스 데이터 로드 완료 ({len(self.source_data)}개 항목)")
        return True
    
    def load_local_model(self):
        """로컬 모델 로드"""
        try:
            if not self.local_model_name:
                print("❌ 로컬 모델이 지정되지 않았습니다.")
                return False
            
            model_config = AVAILABLE_MODELS[self.local_model_name]
            model_id = model_config["hf_id"]
            requires_auth = model_config["requires_auth"]
            
            # 캐시 디렉토리 설정
            cache_dir = "/scratch2/sheepswool/workspace/models"
            
            print(f"🔄 모델 로드 중: {model_id}")
            print(f"📂 캐시 디렉토리: {cache_dir}")
            
            if requires_auth and HUGGINGFACE_TOKEN:
                login(token=HUGGINGFACE_TOKEN)
                print(f"✅ Hugging Face 로그인 완료")
            
            # 4비트 양자화 적용
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=cache_dir
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    load_in_4bit=True,
                    device_map="auto"
                )
                print("✅ 모델 로드 완료 (4비트 양자화)")
                return True
                
            except Exception as e:
                print(f"⚠️ 4비트 양자화 모델 로드 실패: {e}")
                print("🔍 일반 모드로 다시 시도...")
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=cache_dir
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        cache_dir=cache_dir,
                        device_map="auto"
                    )
                    print("✅ 모델 로드 완료 (일반 모드)")
                    return True
                except Exception as e:
                    print(f"❌ 모델 로드 실패: {e}")
                    return False
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            traceback.print_exc()
            return False
    
    def get_local_model_completion(self, prompt: str) -> str:
        """로컬 모델을 사용하여 텍스트 생성"""
        # 파이프라인 사용 (대부분의 모델)
        if self.pipeline:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                num_return_sequences=1
            )
            return outputs[0]["generated_text"][len(prompt):]
        else:
            # 모델과 토크나이저 직접 사용
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                num_return_sequences=1
            )
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    def get_openai_completion(self, prompt: str) -> str:
        if not self.client:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=50
        )
        # answer = response.choices[0].message.content.strip()
        # breakpoint()
        return response.choices[0].message.content.strip()
    
    def generate_multiple_words(self, meaning: str, count: int) -> List[str]:
        words = []
        
        for i in range(count):
            prompt = self.user_prompt_template.format(meaning=meaning)
            
            if self.model_type == "openai":
                result = self.get_openai_completion(prompt)
            else:
                result = self.get_local_model_completion(prompt)
            
            if result:
                word = self.extract_generated_word(result)
                words.append(word)
                print(f"  [단어 {i+1}/{count}] '{word}'")
            
            if self.model_type == "openai" and i < count - 1:
                time.sleep(0.5)
        
        return words
    
    def extract_generated_word(self, text: str) -> str:
        import re
        backtick_pattern = re.compile(r'`([^`]+)`')
        backtick_matches = backtick_pattern.findall(text)
        
        if backtick_matches:
            return backtick_matches[0].strip()
        
        bracket_pattern = re.compile(r'\[([^\]]+)\]')
        bracket_matches = bracket_pattern.findall(text)
        
        if bracket_matches:
            print(f"🔍 {bracket_matches}")
            return bracket_matches[0].strip()
        
        text = text.strip()
        if len(text) > 20:
            return text[:20] + "..."
        
        return text
    
    def run(self, max_words=None):
        # 의미 목록 준비
        self.load_source_data()
        self.load_prompts()
        meanings = [item.get('definitions', '') for item in self.source_data]
        original_words = [item.get('word', '') for item in self.source_data]
        
        # 최대 처리할 의미 수 제한
        if max_words is not None and max_words > 0:
            meanings = meanings[:max_words]
        
        print(f"🎯 총 {len(meanings)}개의 의미에 대해 단어를 생성합니다...")
        print(f"🔍 언어: {self.language}, 모델: {self.model_type}, 트라이얼: {self.trial_num}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            print(f"✅ OpenAI API 클라이언트 초기화 완료 (모델: {self.model_name})")
        else:
            # 로컬 모델 로드
            if not self.load_local_model():
                print("❌ 로컬 모델을 로드할 수 없습니다.")
                return []
        
        all_results = []
        if self.words_per_meaning == 1:
            # 배치 처리 (의미당 단어 1개)
            batch_meanings = [meanings[i:i+self.batch_size] for i in range(0, len(meanings), self.batch_size)]
            
            for batch_idx, batch in enumerate(batch_meanings):
                print(f"🔄 배치 {batch_idx+1}/{len(batch_meanings)} 처리 중...")
                
                # 배치 프롬프트 생성
                prompts = [self.user_prompt_template.format(meaning=meaning[0]) for meaning in batch]
                
                # 배치 처리 (모델 타입에 따라)
                if self.model_type == "openai":
                    results = []
                    for prompt in tqdm(prompts, desc="OpenAI API 호출"):
                        result = self.get_openai_completion(prompt)
                        results.append(result)
                        time.sleep(0.5)  # API 제한 방지를 위한 대기
                else:
                    results = []
                    for prompt in tqdm(prompts, desc="로컬 모델 호출"):
                        result = self.get_local_model_completion(prompt)
                        results.append(result)
                
                # 결과 처리
                for i, (meaning, result) in enumerate(zip(batch, results)):
                    if result:
                        generated_word = self.extract_generated_word(result)
                        
                        item_result = {
                            "original_word": original_words[self.batch_size*batch_idx+i],
                            "meaning": meaning[0],
                            "generated_word": generated_word,
                            "model": self.model_name if self.model_type == "openai" else self.local_model_name,
                            "language": self.language,
                            "trial": self.trial_num
                        }
                        
                        all_results.append(item_result)
                        print(f"  [{(batch_idx*self.batch_size)+i+1}/{len(meanings)}] 의미: '{meaning[:30]}...' → 단어: '{generated_word}'")
                        
        else:
            for i, meaning in enumerate(meanings):
                print(f"🔄 의미 {i+1}/{len(meanings)} 처리 중: '{meaning[:30]}...'")
                
                words = self.generate_multiple_words(meaning[0], self.words_per_meaning)
                
                for j, word in enumerate(words):
                    item_result = {
                        "original_word": original_words[i],
                        "meaning": meaning[0],
                        "generated_word": word,
                        "model": self.model_name if self.model_type == "openai" else self.local_model_name,
                        "language": self.language,
                        "trial": self.trial_num
                    }
                    
                    all_results.append(item_result)
        
        final_results = self.save_results(all_results)
        return final_results
    
    def save_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_file = self.output_dir / f"pseudo_words_{self.language}.json"
        csv_file = self.output_dir / f"pseudo_words_{self.language}.csv"
        
        existing_data = []
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"✅ 기존 파일 로드 완료: {output_file} ({len(existing_data)}개 항목)")
            except Exception as e:
                print(f"⚠️ 기존 파일 로드 실패, 새 파일을 생성합니다: {e}")
        
        final_results = self.merge_results(existing_data, all_results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 결과 저장 완료: {output_file} (총 {len(final_results)}개 항목)")
        
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

def setup_model_cache():
    os.environ["TRANSFORMERS_CACHE"] = "/scratch2/sheepswool/model_cache"
    os.environ["HF_HOME"] = "/scratch2/sheepswool/model_cache"

def setup_requirements():
    try:
        import torch
        import transformers
        print("✅ 기본 패키지 확인 완료 (torch, transformers)")
    except ImportError:
        print("⚠️ 기본 패키지를 설치합니다...")
        import subprocess
        subprocess.check_call(["pip", "install", "torch", "transformers"])
    
    # 양자화 지원 패키지 확인
    try:
        import bitsandbytes
        import accelerate
        print("✅ 양자화 지원 패키지 확인 완료 (bitsandbytes, accelerate)")
    except ImportError:
        print("⚠️ 양자화 지원 패키지를 설치합니다 (대형 모델에 필요)...")
        import subprocess
        subprocess.check_call(["pip", "install", "bitsandbytes", "accelerate"])
    
    print("✅ 모든 필요한 패키지가 설치되어 있습니다.")

# 메인 함수
if __name__ == "__main__":
    # 모델 캐시 경로 설정
    setup_model_cache()
    
    parser = argparse.ArgumentParser(description='가상 단어 생성 도구')
    parser.add_argument('--language', '-l', choices=['en', 'fr', 'ko', 'ja'], 
                        help='언어 코드 (en/fr/ko/ja)')
    parser.add_argument('--model', '-m', default='local', choices=['local', 'openai'],
                        help='사용할 모델 타입 (local/openai)')
    
    # 트라이얼 옵션 그룹
    trial_group = parser.add_mutually_exclusive_group()
    trial_group.add_argument('--trial', '-t', help='prompts.json의 트라이얼 번호 (예: trial1, trial2)')
    trial_group.add_argument('--all-trials', action='store_true', 
                             help='모든 트라이얼 실행 (trial1~trial11)')
    trial_group.add_argument('--trial-range', type=str, 
                             help='트라이얼 범위 지정 (예: 1-5, 7,9,11)')
    
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                        help='배치 크기 (기본값: 10)')
    parser.add_argument('--num-words', '-n', type=int, default=None,
                        help='처리할 의미 수 (기본값: 전체)')
    parser.add_argument('--words-per-meaning', '-w', type=int, default=1,
                        help='의미당 생성할 단어 수 (기본값: 1)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='생성된 단어 저장 디렉토리')
    parser.add_argument('--model-name', type=str, default=None,
                        help='사용할 모델 이름 (OpenAI 또는 로컬 모델)')
    parser.add_argument('--debug-model', type=str, choices=list(AVAILABLE_MODELS.keys()),
                        help='모델 디버깅')
    parser.add_argument('--all-models', action='store_true',
                        help='모든 로컬 모델 순차적으로 실행')
    
    args = parser.parse_args()
    
    # 모델 디버깅
    if args.debug_model:
        debug_model(args.debug_model)
        exit(0)
    
    # 필수 인자 확인
    if not args.language:
        parser.error("--language 인수가 필요합니다.")
    
    if not (args.trial or args.all_trials or args.trial_range):
        parser.error("--trial, --all-trials, 또는 --trial-range 중 하나가 필요합니다.")
    
    # OpenAI 모델 사용 시 --all-models 옵션 무시
    if args.model == 'openai' and args.all_models:
        print("⚠️ OpenAI 모델 사용 시 --all-models 옵션은 무시됩니다.")
        args.all_models = False
    
    # 트라이얼 목록 결정
    trials_to_run = []
    
    if args.all_trials:
        # 모든 트라이얼 실행 (trial1~trial11)
        trials_to_run = [f"trial{i}" for i in range(1, 12)]
    elif args.trial_range:
        # 범위 파싱
        ranges = args.trial_range.split(',')
        for r in ranges:
            if '-' in r:
                start, end = map(int, r.split('-'))
                trials_to_run.extend([f"trial{i}" for i in range(start, end+1)])
            else:
                trials_to_run.append(f"trial{int(r)}")
    else:
        # 단일 트라이얼
        trials_to_run = [args.trial]
    
    # 모델 목록 결정
    models_to_run = []
    
    if args.all_models:
        # 모든 로컬 모델 실행
        models_to_run = list(AVAILABLE_MODELS.keys())
        print(f"🔄 모든 로컬 모델을 순차적으로 실행합니다: {len(models_to_run)}개")
    elif args.model == 'local' and args.model_name:
        # 특정 로컬 모델만 실행
        models_to_run = [args.model_name]
    elif args.model == 'local':
        # 기본 모델 사용
        models_to_run = ["gpt2"]
        print(f"⚠️ 로컬 모델이 지정되지 않아 기본값 {models_to_run[0]}을 사용합니다.")
    
    # 필요한 패키지 확인
    setup_requirements()
    
    # OpenAI 모델 사용
    if args.model == 'openai':
        # 생성기 초기화
        generator = PseudoWordGenerator(
            language=args.language,
            model=args.model,
            trial_num=trials_to_run[0],  # 첫 번째 트라이얼로 초기화
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            local_model=None
        )
        
        # 모델 이름 설정
        if args.model_name:
            generator.model_name = args.model_name
        
        # 의미당 단어 수 설정
        if args.words_per_meaning > 1:
            generator.set_words_per_meaning(args.words_per_meaning)
        
        # 모든 트라이얼 실행
        for trial in trials_to_run:
            print(f"\n{'='*80}")
            print(f"🔍 트라이얼 {trial} 실행 중... (모델: {generator.model_name})")
            print(f"{'='*80}\n")
            
            # 트라이얼 번호 설정
            generator.set_trial_num(trial)
            
            # 실행
            generator.run(args.num_words)
    else:
        # 로컬 모델 사용
        # 모든 로컬 모델 순차적으로 실행
        for model_name in models_to_run:
            print(f"\n{'#'*100}")
            print(f"🔍 모델 {model_name} 로드 중...")
            print(f"{'#'*100}\n")
            
            try:
                # 생성기 초기화
                generator = PseudoWordGenerator(
                    language=args.language,
                    model='local',
                    trial_num=trials_to_run[0],  # 첫 번째 트라이얼로 초기화
                    batch_size=args.batch_size,
                    output_dir=args.output_dir,
                    local_model=model_name
                )
                
                # 의미당 단어 수 설정
                if args.words_per_meaning > 1:
                    generator.set_words_per_meaning(args.words_per_meaning)
                
                # 모델 로드 시도
                success = generator.load_local_model()
                
                if not success:
                    print(f"❌ 모델 {model_name} 로드 실패, 다음 모델로 진행합니다.")
                    continue
                
                # 모든 트라이얼 실행
                for trial in trials_to_run:
                    print(f"\n{'='*80}")
                    print(f"🔍 트라이얼 {trial} 실행 중... (모델: {model_name})")
                    print(f"{'='*80}\n")
                    
                    # 트라이얼 번호 설정
                    generator.set_trial_num(trial)
                    
                    # 실행
                    try:
                        generator.run(args.num_words)
                    except Exception as e:
                        print(f"❌ 트라이얼 {trial} 실행 중 오류 발생: {e}")
                        traceback.print_exc()
                        print("⚠️ 다음 트라이얼로 진행합니다.")
                        continue
            
            except Exception as e:
                print(f"❌ 모델 {model_name} 사용 중 오류 발생: {e}")
                traceback.print_exc()
                print("⚠️ 다음 모델로 진행합니다.")
                continue
            
            finally:
                # 메모리 정리
                if 'generator' in locals() and generator.model:
                    try:
                        del generator.model
                        del generator.tokenizer
                        del generator.pipeline
                        import gc
                        gc.collect()
                        
                        if 'torch' in sys.modules:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        print("✅ 메모리 정리 완료")
                    except:
                        pass
    
    print("\n�� 모든 작업이 완료되었습니다.")
