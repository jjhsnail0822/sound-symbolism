#!/usr/bin/env python3
# python pseudo_word_generation.py -l ko -m local --local-model gpt2 -t trial10 -n 5 -w 2
# python pseudo_word_generation.py -l ko -m local --local-model qwen3-32b -t trial2 -n 100 -w 2
# python pseudo_word_generation.py -l ko -m local --local-model qwen3-4b -t trial10 
# python pseudo_word_generation.py --download-model bloom-560m
# python pseudo_word_generation.py --debug-model gpt2

import os
import json
import argparse
import random
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any, Optional
from huggingface_hub import login, model_info
import shutil
import psutil

# HuggingFace 모델 로드를 위한 라이브러리
from transformers import pipeline, AutoTokenizer

# 환경 변수 로드
load_dotenv('.env.local')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

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

# Hugging Face 로그인 함수
def hf_login() -> bool:
    """Hugging Face에 로그인"""
    if HUGGINGFACE_TOKEN:
        try:
            login(token=HUGGINGFACE_TOKEN)
            print("✅ Hugging Face에 로그인했습니다.")
            return True
        except Exception as e:
            print(f"⚠️ Hugging Face 로그인 실패: {e}")
    else:
        print("⚠️ HUGGINGFACE_TOKEN이 설정되지 않았습니다.")
        print("일부 모델은 로그인이 필요할 수 있습니다.")
    
    return False

# 모델 사용 가능 여부 확인 함수
def check_model_access(model_name: str) -> bool:
    """모델 접근 가능 여부 확인"""
    if model_name not in AVAILABLE_MODELS:
        print(f"⚠️ 지원하지 않는 모델입니다: {model_name}")
        print(f"사용 가능한 모델: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    model_id = AVAILABLE_MODELS[model_name]["hf_id"]
    print(f"🔍 모델 '{model_id}' 접근성 확인 중...")
    
    try:
        info = model_info(model_id)
        print(f"✅ 모델 '{model_id}'에 접근할 수 있습니다.")
        return True
    except Exception as e:
        print(f"❌ 모델 '{model_id}'에 접근할 수 없습니다: {e}")
        print("이 모델은 로그인이 필요하거나 접근이 제한되어 있습니다.")
        return False

# 모델 디버깅 함수
def debug_model(model_name: str) -> None:
    """모델 접근성 디버깅"""
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
        try:
            pipe = pipeline("text-generation", model=model_id, max_new_tokens=20)
            test_result = pipe("This is a test:")
            print(f"✅ 파이프라인 성공적으로 생성되었습니다.")
            print(f"테스트 결과: {test_result[0]['generated_text']}")
        except Exception as e:
            print(f"❌ 파이프라인 생성 실패: {e}")
    except Exception as e:
        print(f"❌ 모델에 접근할 수 없습니다: {e}")
    
    print("===== 디버깅 완료 =====")

def check_disk_space(model_name: str) -> bool:
    """모델 다운로드에 필요한 디스크 공간이 충분한지 확인"""
    # 모델 크기 추정 (단위: GB)
    model_sizes = {
        "3b": 3.0,
        "4b": 4.0,
        "7b": 7.0,
        "8b": 8.0,
        "12b": 12.0,
        "13b": 13.0,
        "14b": 14.0,
        "27b": 27.0,
        "32b": 32.0,
        "72b": 72.0
    }
    
    # 모델 이름에서 크기 추출
    model_size_gb = 2.0  # 기본값
    for size_key, size_value in model_sizes.items():
        if size_key in model_name.lower():
            model_size_gb = size_value
            break
    
    # 캐시 디렉토리 - 환경 변수에서 가져오거나 기본값 사용
    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", "~/.cache/huggingface/hub")
    cache_dir = os.path.expanduser(cache_dir)
    
    # 디스크 사용량 확인
    try:
        # 캐시 디렉토리가 있는 파티션의 남은 공간 확인
        disk_usage = shutil.disk_usage(cache_dir if os.path.exists(cache_dir) else os.path.dirname(cache_dir))
        free_space_gb = disk_usage.free / (1024 ** 3)  # 바이트를 GB로 변환
        
        print(f"🔍 필요한 디스크 공간: {model_size_gb:.1f}GB, 가용 공간: {free_space_gb:.1f}GB")
        print(f"📁 모델 캐시 경로: {cache_dir}")
        
        # 필요한 공간의 1.5배를 확보했는지 확인 (여유 있게)
        required_space = model_size_gb * 1.5
        if free_space_gb < required_space:
            print(f"⚠️ 디스크 공간 부족! 최소 {required_space:.1f}GB가 필요하지만 {free_space_gb:.1f}GB만 사용 가능합니다.")
            print("💡 해결 방법:")
            print("  1. 불필요한 파일을 삭제하여 디스크 공간을 확보하세요.")
            print("  2. 다음 명령어로 HF 캐시를 정리할 수 있습니다: rm -rf ~/.cache/huggingface")
            print("  3. 더 작은 모델(예: 3B/4B 크기 모델)을 사용해보세요.")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️ 디스크 공간 확인 중 오류: {e}")
        return False

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
        print(f"✅ 캐시 디렉토리에 쓰기 권한 확인됨")
        
        # 현재 디스크 공간 확인
        disk_usage = shutil.disk_usage(custom_cache_dir)
        free_space_gb = disk_usage.free / (1024 ** 3)
        print(f"✅ 캐시 디렉토리 가용 공간: {free_space_gb:.1f}GB")
        
        return True
    except Exception as e:
        print(f"⚠️ 캐시 디렉토리 권한 또는 공간 확인 실패: {e}")
        return False

class PseudoWordGenerator:
    def __init__(self, language: str, model: str = "local", trial_num: str = "trial1", 
                 batch_size: int = 10, output_dir: Optional[str] = None, 
                 local_model: Optional[str] = "gpt2"):
        """가상 단어 생성기 초기화"""
        self.language = language.lower()
        if self.language not in ['en', 'ja', 'ko', 'fr']:
            raise ValueError("언어는 'en', 'ja', 'ko', 'fr' 중 하나여야 합니다.")
        
        self.model_type = model.lower()
        if self.model_type != "local":
            self.model_type = "local"  # OpenAI 대신 무조건 local 모델 사용
            print("⚠️ OpenAI API 대신 로컬 모델을 사용합니다.")
        
        self.trial_num = trial_num
        self.batch_size = batch_size
        
        # 출력 디렉토리 설정
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"../0_raw/art/{self.language}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로컬 모델 설정
        self.local_model_name = local_model or "gpt2"
        if self.local_model_name not in AVAILABLE_MODELS:
            print(f"⚠️ 지원하지 않는 모델입니다: {self.local_model_name}")
            print(f"gpt2 모델로 대체합니다.")
            self.local_model_name = "gpt2"
        
        # 모델 파이프라인 (처음에는 None, 필요할 때 로드)
        self.pipeline = None
        
        # 프롬프트 및 소스 데이터 로드
        self.load_prompts()
        self.load_source_data()
        
        # 생성 설정
        self.words_per_meaning = 1  # 기본값: 의미당 1개 단어
    
    def load_prompts(self) -> None:
        """프롬프트 템플릿 로드"""
        # prompts.json 파일 경로
        prompts_file = Path("../../analysis/experiments/prompts.json")
        
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        # generation 키 아래의 프롬프트만 사용
        generation_prompts = prompts_data.get('generation', {})
        
        # 트라이얼 존재 여부 확인
        if self.trial_num not in generation_prompts:
            print(f"❌ generation 프롬프트에서 {self.trial_num}을(를) 찾을 수 없습니다.")
            available_trials = list(generation_prompts.keys())
            if available_trials:
                self.trial_num = available_trials[0]
                print(f"대신 {self.trial_num}을(를) 사용합니다.")
            else:
                raise ValueError("사용 가능한 트라이얼이 없습니다.")
        
        # 트라이얼 정보 로드
        trial_info = generation_prompts[self.trial_num]
        
        # 설명 출력 (있으면)
        if 'explanation' in trial_info:
            print(f"\n===== Trial {self.trial_num} 정보 =====")
            for key, value in trial_info['explanation'].items():
                print(f"  {key}: {value}")
        
        # 언어별 프롬프트 템플릿 로드
        if self.language in trial_info:
            if 'user_prompt' in trial_info[self.language] and trial_info[self.language]['user_prompt']:
                # 프롬프트 최적화
                optimized_prompt = self.optimize_prompt(trial_info)
                if optimized_prompt:
                    self.prompt_template = optimized_prompt
                    print(f"✅ {self.language} 언어용 프롬프트를 최적화하여 로드했습니다.")
                else:
                    self.prompt_template = trial_info[self.language]['user_prompt']
                    print(f"✅ {self.language} 언어용 기본 프롬프트를 로드했습니다.")
            else:
                # 선택된 언어에 프롬프트가 없거나 비어있음
                print(f"⚠️ {self.trial_num}에서 {self.language}에 대한 프롬프트가 비어 있습니다.")
                
                # 한국어 프롬프트가 있으면 대신 사용
                if 'ko' in trial_info and 'user_prompt' in trial_info['ko'] and trial_info['ko']['user_prompt']:
                    self.language = 'ko'  # 언어를 한국어로 변경
                    self.prompt_template = trial_info['ko']['user_prompt']
                    print(f"✅ 대신 한국어(ko) 프롬프트를 사용합니다.")
                else:
                    # 사용 가능한 첫 번째 언어 사용
                    for lang, lang_data in trial_info.items():
                        if lang != 'explanation' and 'user_prompt' in lang_data and lang_data['user_prompt']:
                            self.language = lang
                            self.prompt_template = lang_data['user_prompt']
                            print(f"✅ 대신 {lang} 언어 프롬프트를 사용합니다.")
                            break
                    else:
                        raise ValueError(f"{self.trial_num}에서 사용 가능한 프롬프트가 없습니다.")
        else:
            # 선택된 언어가 없음
            print(f"⚠️ {self.trial_num}에서 {self.language}에 대한 프롬프트를 찾을 수 없습니다.")
            
            # 한국어 프롬프트가 있으면 대신 사용
            if 'ko' in trial_info and 'user_prompt' in trial_info['ko'] and trial_info['ko']['user_prompt']:
                self.language = 'ko'  # 언어를 한국어로 변경
                self.prompt_template = trial_info['ko']['user_prompt']
                print(f"✅ 대신 한국어(ko) 프롬프트를 사용합니다.")
            else:
                # 사용 가능한 첫 번째 언어 사용
                for lang, lang_data in trial_info.items():
                    if lang != 'explanation' and 'user_prompt' in lang_data and lang_data['user_prompt']:
                        self.language = lang
                        self.prompt_template = lang_data['user_prompt']
                        print(f"✅ 대신 {lang} 언어 프롬프트를 사용합니다.")
                        break
                else:
                    raise ValueError(f"{self.trial_num}에서 사용 가능한 프롬프트가 없습니다.")
        
        print(f"\n===== 선택된 프롬프트 =====")
        print(f"트라이얼: {self.trial_num}")
        print(f"언어: {self.language}")
        print(f"프롬프트 템플릿: {self.prompt_template}")

    def optimize_prompt(self, trial_info):
        """프롬프트 최적화: 기존 프롬프트에 추가 지시사항 포함"""
        if 'user_prompt' not in trial_info[self.language]:
            return None
        
        base_prompt = trial_info[self.language]['user_prompt']
        
        # 최적화된 프롬프트 생성
        optimized_prompt = base_prompt
        
        # 기존 프롬프트에 이미 포함되지 않은 경우에만 다음 내용 추가
        addition = "\n\n반드시 다음 규칙을 지켜주세요:\n생성된 단어만 백틱(`) 사이에 입력하고 다른 설명은 하지 마세요."
        
        if "규칙" not in optimized_prompt:
            optimized_prompt += addition
        
        return optimized_prompt
    
    def load_source_data(self) -> None:
        """소스 데이터 로드"""
        # 소스 데이터 파일 경로
        source_file = Path(f"../1_preprocess/nat/{self.language}.json")
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                self.source_data = json.load(f)
            
            print(f"✅ {len(self.source_data)} 소스 단어를 {source_file}에서 로드했습니다.")
        except FileNotFoundError:
            print(f"❌ 소스 데이터 파일을 찾을 수 없습니다: {source_file}")
            self.source_data = []
            print("빈 소스 데이터를 사용합니다.")
    
    def set_words_per_meaning(self, count: int) -> None:
        """의미당 생성할 단어 수 설정"""
        if count < 1:
            print("⚠️ 의미당 단어 수는 최소 1이어야 합니다.")
            count = 1
        
        self.words_per_meaning = count
        print(f"의미당 생성할 단어 수: {self.words_per_meaning}")
    
    def prepare_model(self) -> bool:
        """모델 준비"""
        # 이미 파이프라인이 로드되어 있는 경우
        if self.pipeline is not None:
            return True
        
        # 모델 캐시 경로 설정
        if not setup_model_cache():
            print("⚠️ 모델 캐시 경로 설정 실패, 기본 경로 사용")
        
        # Hugging Face 로그인 시도 (필요한 경우)
        model_info = AVAILABLE_MODELS[self.local_model_name]
        if model_info.get("requires_auth", False):
            if not hf_login():
                print(f"⚠️ 모델 '{self.local_model_name}'은(는) 로그인이 필요할 수 있습니다.")
        
        model_id = model_info["hf_id"]
        
        # 디스크 공간 확인 (이제 새 캐시 디렉토리 사용)
        if not check_disk_space(self.local_model_name):
            print(f"❌ 모델 '{model_id}' 로드 실패: 디스크 공간 부족")
            return False
        
        print(f"🔧 모델 '{model_id}' 로드 중...")
        
        try:
            # 환경 변수 설정 - 로컬 캐시만 사용하도록 설정
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            
            # GPU 사용 가능 여부 확인
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"💻 사용 장치: {device}")
            
            # 모델 크기 추정 (이름에서 숫자 추출)
            model_size = 0
            import re
            size_match = re.search(r'(\d+)[bB]', self.local_model_name)
            if size_match:
                model_size = int(size_match.group(1))
            
            # CUDA 메모리 상태 출력
            if device == "cuda":
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                free_memory_gb = free_memory / (1024**3)
                print(f"🔍 CUDA 가용 메모리: {free_memory_gb:.2f}GB")
                
                if model_size > 0 and free_memory_gb < model_size/2:
                    print(f"⚠️ GPU 메모리가 부족할 수 있습니다. 최소 {model_size/2:.1f}GB 권장 (현재 {free_memory_gb:.2f}GB)")
            
            # 큰 모델 (7B 이상)에 대한 양자화 및 메모리 최적화 설정
            if model_size >= 7:
                print(f"🔍 {model_size}B 이상의 큰 모델을 로드합니다. 메모리 최적화를 적용합니다.")
                
                try:
                    # 캐시 디렉토리 설정 - 환경 변수 또는 임시 디렉토리 사용
                    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", None)
                    
                    # 4비트 양자화 시도
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    # 텍스트 생성 파이프라인 생성 (양자화 적용)
                    from transformers import AutoModelForCausalLM
                    
                    # 모델 로드 시 더 많은 옵션 제공
                    print(f"📥 모델 파일 다운로드 시작...")
                    self.pipeline = pipeline(
                        "text-generation",
                        model=model_id,
                        model_kwargs={
                            "quantization_config": quantization_config, 
                            "device_map": "auto",
                            "cache_dir": cache_dir,
                            "low_cpu_mem_usage": True,
                            "torch_dtype": torch.float16 if device == "cuda" else torch.float32
                        },
                        max_new_tokens=50,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2
                    )
                    print(f"✅ 모델 '{model_id}' 로드 완료! (4비트 양자화 적용)")
                    return True
                except (ImportError, Exception) as e:
                    print(f"⚠️ 4비트 양자화 적용 실패: {e}")
                    # 일반 모드로 계속 시도
            
            # 일반 모드 (작은 모델 또는 양자화 실패 시)
            cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", None)
            self.pipeline = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={
                    "cache_dir": cache_dir,
                    "low_cpu_mem_usage": True
                },
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            print(f"✅ 모델 '{model_id}' 로드 완료!")
            return True
        
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            # 자세한 에러 메시지와 해결 방법 제공
            print("\n==== 문제 해결 방법 ====")
            print("1. 디스크 공간을 확보하세요.")
            print("2. 더 작은 모델(3B/4B)을 사용해보세요.")
            print("3. 다음 명령어로 HF 캐시를 비우세요: rm -rf ~/.cache/huggingface")
            print("4. 다른 경로를 캐시로 사용하려면 다음 명령어를 실행하세요:")
            print("   export HUGGINGFACE_HUB_CACHE=/path/with/more/space")
            print("5. 모델 다운로드를 수동으로 시도하세요:")
            print(f"   python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model_id}')\"")
            print("===========================")
            
            print("❌ 모델 준비 실패")
            return False
    
    def extract_word(self, text: str, meaning: str) -> str:
        """생성된 텍스트에서 단어 추출 및 유효성 검사"""
        # 입력 프롬프트 제거
        prompt = self.prompt_template.format(meaning="")
        if prompt in text and len(prompt) < len(text):
            text = text[len(prompt):].strip()
        
        # 백틱(`) 또는 대괄호([]) 사이의 내용 추출 시도
        extracted_word = None
        
        # 백틱 안에 있는 내용 추출
        if '`' in text:
            parts = text.split('`')
            if len(parts) >= 3:  # `word` 형식
                extracted_word = parts[1].strip()
        
        # 대괄호 안에 있는 내용 추출
        if not extracted_word and '[' in text and ']' in text:
            start_idx = text.find('[')
            end_idx = text.find(']', start_idx)
            if start_idx != -1 and end_idx != -1:
                extracted_word = text[start_idx+1:end_idx].strip()
        
        # 추출 실패 시 첫 번째 줄 사용
        if not extracted_word:
            lines = text.strip().split('\n')
            extracted_word = lines[0].strip() if lines else ""
        
        # 유효성 검사
        if extracted_word:
            # 1. 단어 길이 검사 (너무 길면 무효)
            if len(extracted_word) > 15:
                print(f"⚠️ 생성된 단어가 너무 깁니다: {extracted_word}")
                return None
            
            # 2. 의미와 유사성 검사 (의미가 그대로 포함되면 무효)
            meaning_words = set(meaning.replace(',', ' ').replace('.', ' ').split())
            for word in meaning_words:
                if len(word) > 3 and word in extracted_word:  # 3글자 이상의 의미 단어가 포함
                    print(f"⚠️ 생성된 단어에 의미가 그대로 포함됨: {extracted_word}, 포함단어: {word}")
                    return None
            
            # 3. 특정 무효 패턴 검사
            invalid_patterns = ["생성된 어휘", "생성된어휘", "가상", "음성상징어", 
                              "예시", "단어", "일본어", "한국어", "영어", "프랑스어"]
            for pattern in invalid_patterns:
                if pattern in extracted_word:
                    print(f"⚠️ 생성된 단어에 무효 패턴 포함: {pattern}")
                    return None
            
            # 유효한 단어인 경우 반환
            return extracted_word
        
        return None  # 추출 실패
    
    def generate_words(self, num_words: Optional[int] = None) -> List[Dict[str, Any]]:
        """여러 단어 생성 (배치 처리 없이 한 번에 하나씩)"""
        # 기존 결과 로드
        all_results = self.load_existing_results()
        
        # 생성할 단어 수 결정
        if num_words is None:
            num_words = len(self.source_data)
        else:
            num_words = min(num_words, len(self.source_data))
        
        # 소스 데이터가 없는 경우
        if not self.source_data:
            print("❌ 소스 데이터가 없습니다.")
            return all_results
        
        # 소스 데이터에서 무작위 선택
        selected_data = random.sample(self.source_data, num_words)
        
        # 의미와 원본 단어 추출
        meanings = []
        original_words = []
        
        for item in selected_data:
            # 의미 추출 (리스트인 경우 첫 번째 항목 사용)
            meaning = item.get('meaning', [])
            if isinstance(meaning, list) and meaning:
                meaning = meaning[0]
            elif not meaning:
                meaning = "의미 없음"
            
            meanings.append(meaning)
            original_words.append(item.get('word', ''))
        
        print(f"🔍 {len(meanings)}개 의미에 대해 가상 단어 생성/업데이트 시작...")
        
        # 변경된 항목 수 추적
        added_count = 0
        updated_count = 0
        
        # 각 의미에 대해 단어 생성 (배치 처리 없이 한 번에 하나씩)
        for idx, (meaning, orig_word) in enumerate(tqdm(zip(meanings, original_words), 
                                                  desc="단어 생성 중", total=len(meanings))):
            # 의미당 여러 단어 생성
            for i in range(self.words_per_meaning):
                # 이미 생성된 항목인지 확인
                dup_idx = self.is_duplicate_entry(meaning, orig_word, all_results)
                
                # 기존 항목이 있고 생성된 단어가 비어있지 않으면 스킵
                if dup_idx >= 0 and all_results[dup_idx].get("generated_word"):
                    print(f"🔄 건너뜀: '{meaning}' (이미 '{all_results[dup_idx]['generated_word']}'로 생성됨)")
                    continue
                
                # 단어 생성 (최대 3번 시도)
                generated_word = ""
                for attempt in range(3):
                    word = self.generate_word(meaning)
                    if word:
                        generated_word = word
                        print(f"✅ {attempt+1}번째 시도에서 생성 성공: '{word}'")
                        break
                    print(f"⚠️ {attempt+1}번째 시도 실패, 재시도 중...")
                    time.sleep(0.5)
                
                # 결과 생성
                result = {
                    "original_meaning": meaning,
                    "original_word": orig_word,
                    "generated_word": generated_word,
                    "trial": self.trial_num,
                    "model": AVAILABLE_MODELS[self.local_model_name]["hf_id"],
                    "language": self.language,
                    "words_per_meaning": self.words_per_meaning
                }
                
                # 기존 항목 업데이트 또는 새 항목 추가
                if dup_idx >= 0:
                    # 기존 항목 업데이트
                    if generated_word:  # 생성된 단어가 있는 경우만 업데이트
                        all_results[dup_idx] = result
                        updated_count += 1
                        print(f"🔄 업데이트: '{meaning}' -> '{generated_word}'")
                else:
                    # 새 항목 추가
                    all_results.append(result)
                    added_count += 1
                    if generated_word:
                        print(f"➕ 추가: '{meaning}' -> '{generated_word}'")
                    else:
                        print(f"⚠️ 추가: '{meaning}' -> 생성 실패")
                
                # 변경된 내용 중간 저장 (10개마다)
                if (added_count + updated_count) % 10 == 0 and (added_count + updated_count) > 0:
                    self.save_results(all_results)
        
        # 최종 결과 저장
        self.save_results(all_results)
        
        print(f"✅ 가상 단어 생성 완료 - 추가: {added_count}개, 업데이트: {updated_count}개, 총: {len(all_results)}개")
        
        return all_results
    
    def generate_word(self, meaning: str) -> str:
        """주어진 의미에 대한 가상 단어 생성"""
        if not self.prepare_model():
            return ""
        
        prompt = self.prompt_template.format(meaning=meaning)
        
        try:
            # 채팅 형식 모델인지 확인
            is_chat_model = any(keyword in self.local_model_name.lower() for keyword in 
                              ["instruct", "chat", "it", "thinking"])
            
            # 결과 생성
            if is_chat_model:
                messages = [
                    {"role": "system", "content": "당신은 가상 언어를 만드는 전문가입니다. 주어진 의미에 맞는 짧고 독창적인 음성상징어를 생성해주세요."},
                    {"role": "user", "content": prompt}
                ]
                
                try:
                    result = self.pipeline(messages, max_new_tokens=50)
                    generated_text = result[0]['generated_text']
                except Exception as e:
                    print(f"⚠️ 채팅 모드 에러: {e}, 일반 텍스트 모드로 시도")
                    result = self.pipeline(prompt, max_new_tokens=50)
                    generated_text = result[0]['generated_text']
            else:
                result = self.pipeline(prompt, max_new_tokens=50)
                generated_text = result[0]['generated_text']
            
            # 생성된 텍스트에서 단어 추출
            word = self.extract_word(generated_text, meaning)
            if word:
                print(f"🔤 생성된 단어: {word} (의미: {meaning})")
                return word
            
            return ""
        except Exception as e:
            print(f"❌ 단어 생성 중 오류: {e}")
            return ""
    
    def run(self, num_words: Optional[int] = None) -> List[Dict[str, Any]]:
        """가상 단어 생성 실행"""
        print(f"===== 가상 단어 생성 시작 =====")
        print(f"언어: {self.language}")
        print(f"모델: {AVAILABLE_MODELS[self.local_model_name]['hf_id']}")
        print(f"트라이얼: {self.trial_num}")
        print(f"의미당 단어 수: {self.words_per_meaning}")
        
        # 모델 준비
        if not self.prepare_model():
            print("❌ 모델 준비 실패")
            return []
        
        # 단어 생성
        try:
            results = self.generate_words(num_words)
            print(f"===== 가상 단어 생성 완료 =====")
            return results
        except Exception as e:
            print(f"❌ 단어 생성 중 오류 발생: {e}")
            traceback.print_exc()
            return []

    def load_existing_results(self) -> List[Dict[str, Any]]:
        """기존에 생성된 단어 결과 로드"""
        output_file = self.output_dir / f"pseudo_words_{self.language}_{self.trial_num[5:]}.json"
        
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                print(f"✅ 기존 파일에서 {len(existing_results)}개 결과를 로드했습니다: {output_file}")
                return existing_results
            except Exception as e:
                print(f"⚠️ 기존 파일 로드 중 오류 발생: {e}")
        
        print(f"📝 새 결과 파일을 생성합니다: {output_file}")
        return []

    def is_duplicate_entry(self, meaning: str, orig_word: str, existing_results: List[Dict[str, Any]]) -> int:
        """이미 생성된 단어인지 확인하고 중복 항목의 인덱스 반환"""
        model_id = AVAILABLE_MODELS[self.local_model_name]["hf_id"]
        
        for idx, entry in enumerate(existing_results):
            # 의미, 원본 단어, 모델이 모두 일치하는 경우
            if (entry.get("original_meaning") == meaning and 
                entry.get("original_word") == orig_word and 
                entry.get("model") == model_id and
                entry.get("trial") == self.trial_num):
                return idx
        
        # 중복 없음
        return -1

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """생성된 단어 결과 저장 (JSON 및 CSV)"""
        # 결과가 없는 경우
        if not results:
            print("⚠️ 저장할 결과가 없습니다.")
            return
        
        # JSON 파일 저장
        output_file = self.output_dir / f"pseudo_words_{self.language}_{self.trial_num[5:]}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {len(results)}개 결과를 JSON으로 저장했습니다: {output_file}")
        
        # CSV 파일 저장
        csv_file = self.output_dir / f"pseudo_words_{self.language}_{self.trial_num[5:]}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ 결과를 CSV로도 저장했습니다: {csv_file}")

def setup_requirements():
    """필요한 패키지가 설치되어 있는지 확인하고 설치"""
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
    # 먼저 모델 캐시 경로 설정
    setup_model_cache()
    
    parser = argparse.ArgumentParser(description='가상 단어 생성 도구')
    parser.add_argument('--language', '-l', choices=['en', 'fr', 'ko', 'ja'], 
                        help='언어 코드 (en/fr/ko/ja)')
    parser.add_argument('--model', '-m', default='local', choices=['local'],
                        help='사용할 모델 (local만 지원)')
    parser.add_argument('--trial', '-t', help='prompts.json의 트라이얼 번호 (예: trial1, trial2)')
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                        help='배치 크기 (기본값: 10)')
    parser.add_argument('--num-words', '-n', type=int, default=None,
                        help='처리할 의미 수 (기본값: 전체)')
    parser.add_argument('--words-per-meaning', '-w', type=int, default=1,
                        help='의미당 생성할 단어 수 (기본값: 1)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='생성된 단어 저장 디렉토리')
    parser.add_argument('--local-model', type=str, choices=list(AVAILABLE_MODELS.keys()),
                        help='사용할 로컬 모델')
    parser.add_argument('--debug-model', type=str, choices=list(AVAILABLE_MODELS.keys()),
                        help='모델 디버깅')
    
    args = parser.parse_args()
    
    # 모델 디버깅
    if args.debug_model:
        debug_model(args.debug_model)
        exit(0)
    
    # 필수 인자 확인
    if not args.language or not args.trial:
        parser.error("--language 및 --trial 인수가 필요합니다.")
    
    # 로컬 모델 사용 시 --local-model 필요
    if args.model == 'local' and not args.local_model:
        args.local_model = "gpt2"  # 기본값 설정
        print(f"⚠️ 로컬 모델이 지정되지 않아 기본값 {args.local_model}을 사용합니다.")
    
    # 필요한 패키지 확인
    setup_requirements()
    
    # 생성기 초기화
    generator = PseudoWordGenerator(
        language=args.language,
        model=args.model,
        trial_num=args.trial,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        local_model=args.local_model
    )
    
    # 의미당 단어 수 설정
    if args.words_per_meaning > 1:
        generator.set_words_per_meaning(args.words_per_meaning)
    
    # 실행
    generator.run(args.num_words)
