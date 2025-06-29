# Semantic Dimension Heatmap Analysis

이 디렉토리는 Qwen2.5-Omni 모델의 semantic dimension attention heatmap 분석을 위한 도구들을 포함합니다.

## 파일 구조

- `semdim_heatmap.py`: 모델 추론을 통해 attention matrix를 추출하고 pickle 파일로 저장
- `heatmap_plot.py`: 저장된 pickle 파일을 읽어서 heatmap과 flow plot을 생성

## 사용법

### 1단계: Attention Matrix 추출 및 저장

```bash
# 기본 실행 (모든 언어, audio 데이터 타입)
python src/analysis/heatmap/semdim_heatmap.py

# 특정 언어만 처리
python src/analysis/heatmap/semdim_heatmap.py --languages en

# 샘플 수 제한
python src/analysis/heatmap/semdim_heatmap.py --max-samples 100 --languages en

# 다른 데이터 타입 사용
python src/analysis/heatmap/semdim_heatmap.py --data-type original --languages en

# 출력 디렉토리 지정
python src/analysis/heatmap/semdim_heatmap.py --output-dir results/my_experiment
```

**주요 옵션:**
- `--model`: 모델 경로 (기본값: Qwen/Qwen2.5-Omni-7B)
- `--data-path`: 데이터 JSON 파일 경로
- `--output-dir`: 결과 저장 디렉토리
- `--data-type`: 데이터 타입 (audio, original, romanized, ipa)
- `--max-samples`: 처리할 최대 샘플 수
- `--languages`: 처리할 언어들 (en, fr, ko, ja)

### 2단계: Heatmap 및 Flow Plot 생성

```bash
# 사용 가능한 pickle 파일 요약 보기
python src/analysis/heatmap/heatmap_plot.py --summary

# 모든 언어의 모든 파일을 자동으로 처리
python src/analysis/heatmap/heatmap_plot.py --auto-process-all

# 특정 언어의 모든 파일을 배치 처리
python src/analysis/heatmap/heatmap_plot.py --batch-mode --languages en

# 특정 단어와 차원에 대한 plot 생성
python src/analysis/heatmap/heatmap_plot.py \
    --word-tokens "apple" \
    --dimension1 "big" \
    --dimension2 "small" \
    --lang en

# 다른 데이터 타입으로 plot 생성
python src/analysis/heatmap/heatmap_plot.py --auto-process-all --data-type original
```

**주요 옵션:**
- `--summary`: 사용 가능한 pickle 파일 요약 표시
- `--auto-process-all`: 모든 언어의 모든 파일을 자동 처리
- `--batch-mode`: 지정된 언어의 모든 파일을 배치 처리
- `--word-tokens`, `--dimension1`, `--dimension2`: 특정 단어와 차원 지정
- `--layer-type`: attention layer 타입 (self, cross, output)
- `--head`, `--layer`: 특정 head와 layer 지정
- `--data-type`: 데이터 타입 (audio, original, romanized, ipa)

## 출력 파일 구조

```
results/experiments/understanding/attention_heatmap/
├── semantic_dimension/
│   ├── audio/
│   │   ├── en/
│   │   │   ├── word_dim1_dim2_self.pkl          # Attention matrix
│   │   │   ├── semdim_heatmap_word_dim1_dim2.png # Individual heatmap
│   │   │   ├── semdim_avg_heatmap_word.png      # Average heatmap
│   │   │   └── semdim_flow_word_dim1_dim2.png   # Flow plot
│   │   ├── fr/
│   │   ├── ko/
│   │   └── ja/
│   ├── original/
│   ├── romanized/
│   └── ipa/
```

## Pickle 파일 구조

각 pickle 파일은 다음 정보를 포함합니다:
- `attention_matrix`: 원본 attention matrix
- `dimension1`, `dimension2`: 비교할 semantic dimension
- `answer`: 정답
- `word_tokens`: 단어 토큰
- `option_tokens`: 옵션 토큰들
- `tokens`: 전체 토큰 시퀀스

## 예시 워크플로우

1. **데이터 준비**: semantic dimension 데이터가 JSON 형태로 준비되어 있어야 합니다.

2. **Attention Matrix 추출**:
   ```bash
   python src/analysis/heatmap/semdim_heatmap.py --max-samples 50 --languages en
   ```

3. **결과 확인**:
   ```bash
   python src/analysis/heatmap/heatmap_plot.py --summary
   ```

4. **Heatmap 생성**:
   ```bash
   python src/analysis/heatmap/heatmap_plot.py --auto-process-all
   ```

## 주의사항

- GPU 메모리가 충분한지 확인하세요
- 대용량 데이터 처리 시 `--max-samples` 옵션을 사용하여 샘플 수를 제한하세요
- pickle 파일이 생성된 후에만 heatmap plot을 생성할 수 있습니다
- 언어별로 다른 폰트가 필요할 수 있습니다 (한국어, 일본어 등) 