#!/usr/bin/env python3
"""
Qwen2.5-Omni inference와 동일한 방식으로 attention heatmap을 추출/시각화하는 스크립트.
- Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor 직접 사용
- 실험 결과 파일에서 입력(텍스트/오디오/멀티모달) 추출
- output_attentions=True로 모델 forward, attention 추출
- matplotlib/seaborn으로 heatmap 저장

사용 예시:
python src/analysis/heatmap/experiment_heatmap_check.py
--result-file results/experiments/understanding/pair_matching/audiolm/all_results_Qwen_Qwen2.5-Omni-7B.json
--model Qwen/Qwen2.5-Omni-7B
--output-dir heatmap_results
--exp-name audio
--max_samples 5
--layer 0
--head 0
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../experiments/word_meaning_matching'))
import argparse
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def plot_attention_heatmap(attention_weights, tokens, layer_idx=0, head_idx=0, save_path=None, title=None):
    if attention_weights is None:
        print("No attention weights provided")
        return
    if layer_idx >= len(attention_weights):
        print(f"Layer index {layer_idx} out of range. Available layers: {len(attention_weights)}")
        return
    layer_attention = attention_weights[layer_idx]  # [batch, heads, seq_len, seq_len]
    if head_idx >= layer_attention.shape[1]:
        print(f"Head index {head_idx} out of range. Available heads: {layer_attention.shape[1]}")
        return
    attention_matrix = layer_attention[0, head_idx].cpu().numpy()  # [seq_len, seq_len]
    fig, ax = plt.subplots(figsize=(12, 10))
    token_labels = [t[:7] + "..." if len(t) > 10 else t for t in tokens]
    sns.heatmap(
        attention_matrix,
        xticklabels=token_labels,
        yticklabels=token_labels,
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    ax.set_title(title or f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}')
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved heatmap: {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni inference와 동일하게 attention heatmap 시각화")
    parser.add_argument('--result-file', '-r', type=str, required=True, help='실험 결과 JSON 파일')
    parser.add_argument('--model', '-m', type=str, required=True, help='모델 이름 (예: Qwen/Qwen2.5-Omni-7B)')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='heatmap 이미지 저장 디렉토리')
    parser.add_argument('--exp-name', type=str, required=True, help='실험 이름 (멀티모달 처리 분기용)')
    parser.add_argument('--max_samples', type=int, default=20, help='최대 시각화 샘플 수')
    parser.add_argument('--layer', type=int, default=0, help='시각화할 attention layer 인덱스')
    parser.add_argument('--head', type=int, default=0, help='시각화할 attention head 인덱스')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 결과 파일 로드
    with open(args.result_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    samples = results_data['results'] if 'results' in results_data else results_data

    # 모델/프로세서 로드
    print(f"Loading Qwen2.5-Omni model: {args.model}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        cache_dir=os.path.join("../","models"),
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, cache_dir=os.path.join("../","models"))

    for idx, sample in enumerate(samples):
        if idx >= args.max_samples:
            break
        # 입력 준비 (멀티모달 분기)
        if 'audio' in args.exp_name.lower():
            # 오디오 실험: word/language/meta_data 등에서 오디오 경로 추출
            if 'meta_data' in sample and 'word' in sample['meta_data'] and 'language' in sample['meta_data']:
                word = sample['meta_data']['word']
                language = sample['meta_data']['language']
                audio_path = f"data/processed/nat/tts/{language}/{word}.wav"
                if not os.path.exists(audio_path):
                    print(f"[WARNING] 오디오 파일 없음: {audio_path}")
                    continue
                # 텍스트 prompt도 필요 (실험별로 다름)
                if 'query' in sample:
                    question = sample['query']
                elif 'meta_data' in sample and 'question' in sample['meta_data']:
                    question = sample['meta_data']['question']
                else:
                    print(f"[WARNING] 입력 텍스트를 찾을 수 없음 (샘플 {idx})")
                    continue
                # MCQ 실험의 멀티모달 conversation 구성 (qwen_omni_inference.py 참고)
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_path},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            else:
                print(f"[WARNING] 오디오 경로를 찾을 수 없음 (샘플 {idx})")
                continue
        else:
            # 텍스트 실험
            if 'query' in sample:
                question = sample['query']
            elif 'meta_data' in sample and 'question' in sample['meta_data']:
                question = sample['meta_data']['question']
            else:
                print(f"[WARNING] 입력 텍스트를 찾을 수 없음 (샘플 {idx})")
                continue
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                },
            ]
        # processor로 입력 준비
        USE_AUDIO_IN_VIDEO = True
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        # === 디버깅용: 모델 inference 전에 prompt 출력 및 breakpoint ===
        print("\n[DEBUG] Prompt to model:")
        print(text_prompt)
        breakpoint()
        
        inputs = processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(model.device).to(model.dtype)
        # 모델 forward (output_attentions=True)
        with torch.no_grad():
            outputs = model(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                output_attentions=True,
                return_dict=True
            )
        attention_weights = outputs.attentions  # tuple of (num_layers, [batch, heads, seq, seq])
        # 토큰 정보
        if 'input_ids' in inputs:
            tokens = processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        else:
            tokens = [str(i) for i in range(attention_weights[0].shape[-1])]
        # 시각화 및 저장
        label = f"sample{idx}_layer{args.layer}_head{args.head}"
        if 'is_correct' in sample:
            label += f"_correct{int(sample['is_correct'])}"
        save_path = os.path.join(args.output_dir, f"{label}.png")
        plot_attention_heatmap(
            attention_weights, tokens,
            layer_idx=args.layer, head_idx=args.head,
            save_path=save_path,
            title=f"Sample {idx} | Layer {args.layer} | Head {args.head}"
        )

if __name__ == "__main__":
    main()

# === heatmap_check.py 활용 예시 (en.json 등에서 직접 입력을 받아 시각화) ===
"""
from src.analysis.heatmap.heatmap_check import AttentionHeatmapVisualizer
import json

# 언어 코드 예시: 'en', 'fr', 'ja' 등
data_path = 'data/processed/nat/en.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

visualizer = AttentionHeatmapVisualizer(model_name="Qwen/Qwen2.5-Omni-7B")

# 예시: 첫 번째 문제의 question 필드로 attention heatmap 시각화
if len(data) > 0 and 'question' in data[0]:
    text = data[0]['question']
    attn_weights, tokens = visualizer.get_attention_weights(text)
    visualizer.plot_attention_heatmap(attn_weights, tokens, layer_idx=0, head_idx=0)
""" 