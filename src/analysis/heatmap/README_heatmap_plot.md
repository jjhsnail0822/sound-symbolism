# Heatmap Plot Module

이 모듈은 `semdim_heatmap.py`에서 분리된 시각화 기능을 담당합니다. 저장된 pkl 파일을 불러와서 attention heatmap과 flow plot을 생성합니다.

## 파일 구조

- `heatmap_plot.py`: 메인 시각화 모듈
- `semdim_heatmap.py`: 수정된 원본 파일 (시각화 함수 제거됨)

## 주요 기능

### SemanticDimensionHeatmapPlotter 클래스

- `plot_heatmap()`: 개별 레이어의 attention heatmap 생성
- `plot_average_heatmap()`: 모든 레이어의 평균 attention heatmap 생성
- `plot_flow()`: 레이어 간 attention flow plot 생성
- `plot_from_pkl()`: pkl 파일에서 데이터를 불러와 시각화
- `batch_plot_from_pkl()`: 배치 모드로 모든 pkl 파일 처리

## 사용법

### 1. 개별 파일 시각화

```bash
PYTHONPATH=/path/to/workspace python src/analysis/heatmap/heatmap_plot.py \
    --word-tokens "a" \
    --dimension1 "active" \
    --dimension2 "passive" \
    --lang "en" \
    --data-type "audio" \
    --layer-type "self" \
    --head 0 \
    --layer 0
```

### 2. 배치 모드 (모든 pkl 파일 처리)

```bash
PYTHONPATH=/path/to/workspace python src/analysis/heatmap/heatmap_plot.py \
    --batch-mode \
    --languages en fr ko ja \
    --data-type "audio" \
    --layer-type "self"
```

### 3. Python 코드에서 사용

```python
from heatmap_plot import SemanticDimensionHeatmapPlotter

# 플로터 초기화
plotter = SemanticDimensionHeatmapPlotter(
    output_dir="results/experiments/understanding/attention_heatmap",
    exp_type="semantic_dimension",
    data_type="audio"
)

# 개별 파일 시각화
plotter.plot_from_pkl(
    word_tokens="a",
    dimension1="active", 
    dimension2="passive",
    lang="en"
)

# 배치 처리
plotter.batch_plot_from_pkl(lang="en")
```

## 출력 파일

### Heatmap 파일
- `semdim_{data_type}_{exp_type}_{layer_type}_layer{layer}_head{head}_{dimension1}_{dimension2}_{word_tokens}.png`
- `semdim_avg_heatmap_{data_type}_{layer_type}_{word_tokens}.png`

### Flow 파일
- `semdim_flow_{data_type}_{layer_type}.png`

## 데이터 형식

pkl 파일에는 다음 정보가 포함됩니다:
- `attention_matrix`: 어텐션 매트릭스
- `dimension1`, `dimension2`: 시맨틱 차원
- `answer`: 정답
- `word_tokens`: 단어 토큰
- `option_tokens`: 옵션 토큰
- `tokens`: 전체 토큰 시퀀스 (시각화용)

## 주의사항

1. `PYTHONPATH`를 올바르게 설정해야 합니다.
2. pkl 파일이 지정된 경로에 존재해야 합니다.
3. 토큰 정보가 pkl 파일에 저장되어 있어야 정확한 시각화가 가능합니다.

## 수정 사항

### semdim_heatmap.py 변경사항
- 시각화 함수들 (`plot_heatmap`, `plot_average_heatmap`, `plot_flow`) 제거
- `heatmap_plot` 모듈 import 추가
- `save_matrix` 함수에 `tokens` 파라미터 추가
- `inference_with_hooks`에서 `SemanticDimensionHeatmapPlotter` 사용

### heatmap_plot.py 특징
- 독립적인 시각화 모듈
- pkl 파일에서 데이터 로드
- 배치 처리 지원
- 다양한 시각화 옵션 제공 