# Qwen2.5-Omni Attention Heatmap 시각화

이 프로젝트는 Qwen2.5-Omni 모델(7B, 3B)의 attention layer를 heatmap으로 시각화하는 도구입니다.

## 기능

- **단일 Head Attention 시각화**: 특정 layer와 head의 attention weight를 heatmap으로 표시
- **Multi-Head Attention 시각화**: 한 레이어의 여러 head를 동시에 비교
- **Attention 패턴 분석**: 각 head의 평균 attention weight와 최대 attention target 분석
- **모델 비교**: 7B와 3B 모델의 attention 패턴 비교
- **다국어 지원**: 영어, 프랑스어, 일본어, 한국어 등 다양한 언어 텍스트 분석

## 설치

### 1. 의존성 패키지 설치

```bash
pip install -r requirements_attention.txt
```

### 2. 모델 다운로드

코드를 실행하면 자동으로 Qwen2.5-Omni 모델이 다운로드됩니다:
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

## 사용법

### 기본 사용법

```python
from heatmap_check import AttentionHeatmapVisualizer

# 시각화 객체 생성
visualizer = AttentionHeatmapVisualizer("Qwen/Qwen2.5-7B-Instruct")

# 텍스트 입력
text = "Sound symbolism is the study of how sounds relate to meaning."

# Attention weights 추출
attention_weights, tokens = visualizer.get_attention_weights(text)

# 단일 head heatmap 시각화
visualizer.plot_attention_heatmap(
    attention_weights, tokens, 
    layer_idx=0, head_idx=0,
    save_path="attention_heatmap.png"
)

# Multi-head attention 시각화
visualizer.plot_multi_head_attention(
    attention_weights, tokens, layer_idx=0,
    save_path="multihead_attention.png"
)

# Attention 패턴 분석
visualizer.analyze_attention_patterns(attention_weights, tokens, layer_idx=0)
```

### 예시 스크립트 실행

```bash
# 기본 예시 실행
python src/analysis/example_usage.py

# 전체 분석 실행
python src/analysis/heatmap_check.py
```

## 주요 클래스와 메서드

### AttentionHeatmapVisualizer

#### 초기화
```python
visualizer = AttentionHeatmapVisualizer(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="cuda"  # 또는 "cpu"
)
```

#### 주요 메서드

1. **get_attention_weights(text)**
   - 입력 텍스트에 대한 attention weights 추출
   - 반환: (attention_weights, tokens)

2. **plot_attention_heatmap(attention_weights, tokens, layer_idx=0, head_idx=0, save_path=None)**
   - 단일 head의 attention heatmap 시각화
   - 파라미터:
     - `layer_idx`: 시각화할 레이어 인덱스
     - `head_idx`: 시각화할 head 인덱스
     - `save_path`: 저장할 파일 경로

3. **plot_multi_head_attention(attention_weights, tokens, layer_idx=0, num_heads=8, save_path=None)**
   - 여러 head의 attention을 동시에 시각화
   - 파라미터:
     - `num_heads`: 시각화할 head 개수 (최대 8개)

4. **analyze_attention_patterns(attention_weights, tokens, layer_idx=0)**
   - Attention 패턴 분석 및 통계 출력

## 출력 예시

### Heatmap 시각화
- X축: Key tokens
- Y축: Query tokens
- 색상 강도: Attention weight 값
- 파란색 계열의 컬러맵 사용

### 분석 결과
```
=== Attention Pattern Analysis - Layer 0 ===
Model: Qwen/Qwen2.5-7B-Instruct
Sequence length: 12
Number of heads: 32
Number of layers: 32

Mean attention per head:
  Head 0: 0.0312
  Head 1: 0.0298
  ...

Highest attention targets:
  The -> The (weight: 0.1234)
  cat -> sat (weight: 0.0987)
  ...
```

## 파일 구조

```
src/analysis/
├── heatmap_check.py          # 메인 시각화 클래스
├── example_usage.py          # 사용 예시
└── ...

requirements_attention.txt    # 의존성 패키지
README_attention_heatmap.md   # 이 파일
```

## 주의사항

1. **메모리 사용량**: 7B 모델은 약 14GB, 3B 모델은 약 6GB의 GPU 메모리가 필요합니다.
2. **처리 시간**: 긴 텍스트의 경우 attention 계산에 시간이 걸릴 수 있습니다.
3. **모델 크기**: 처음 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다.

## 커스터마이징

### 다른 모델 사용
```python
# 다른 Qwen 모델 사용
visualizer = AttentionHeatmapVisualizer("Qwen/Qwen2.5-14B-Instruct")
```

### 다른 레이어 분석
```python
# 마지막 레이어 분석
last_layer = len(attention_weights) - 1
visualizer.plot_attention_heatmap(attention_weights, tokens, layer_idx=last_layer)
```

### 커스텀 텍스트 분석
```python
texts = [
    "The cat sat on the mat.",
    "Le chat s'est assis sur le tapis.",
    "猫はマットの上に座った。"
]

for text in texts:
    attention_weights, tokens = visualizer.get_attention_weights(text)
    # 분석 로직...
```

## 문제 해결

### CUDA 메모리 부족
- 더 작은 모델(3B) 사용
- 배치 크기 줄이기
- CPU 사용 (`device="cpu"`)

### 모델 로딩 오류
- 인터넷 연결 확인
- Hugging Face 토큰 설정 (필요한 경우)
- 충분한 디스크 공간 확인

## 라이선스

이 코드는 MIT 라이선스 하에 배포됩니다. 