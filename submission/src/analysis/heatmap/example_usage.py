#!/usr/bin/env python3
"""
Qwen2.5-Omni 모델의 Attention Heatmap 시각화 사용 예시

이 스크립트는 heatmap_check.py의 AttentionHeatmapVisualizer 클래스를 사용하여
Qwen2.5-Omni 모델의 attention layer를 시각화하는 방법을 보여줍니다.
"""
import os

from heatmap_check import AttentionHeatmapVisualizer
import matplotlib.pyplot as plt

def simple_example():
    """간단한 사용 예시"""
    print("=== Qwen2.5-Omni Attention Heatmap 시각화 ===")
    
    # 7B 모델로 시각화 객체 생성
    visualizer = AttentionHeatmapVisualizer("Qwen/Qwen2.5-7B-Instruct")
    
    # 테스트 텍스트
    text = "Sound symbolism is the study of how sounds relate to meaning."
    
    print(f"입력 텍스트: {text}")
    
    # Attention weights 추출
    attention_weights, tokens = visualizer.get_attention_weights(text)
    
    if attention_weights is not None:
        print(f"토큰 수: {len(tokens)}")
        print(f"레이어 수: {len(attention_weights)}")
        print(f"첫 번째 레이어의 head 수: {attention_weights[0].shape[1]}")
        
        # 단일 head heatmap 시각화
        visualizer.plot_attention_heatmap(
            attention_weights, tokens, 
            layer_idx=0, head_idx=0,
            save_path=os.path.join(os.path.dirname(__file__), "example_attention_heatmap.png")
        )
        
        # Multi-head attention 시각화
        visualizer.plot_multi_head_attention(
            attention_weights, tokens, layer_idx=0,
            save_path=os.path.join(os.path.dirname(__file__), "example_multihead_attention.png")
        )
        
        # Attention 패턴 분석
        visualizer.analyze_attention_patterns(attention_weights, tokens, layer_idx=0)

def compare_models():
    """7B와 3B 모델 비교"""
    print("\n=== 모델 비교: 7B vs 3B ===")
    
    models = [
        ("Qwen/Qwen2.5-7B-Instruct", "7B"),
        ("Qwen/Qwen2.5-3B-Instruct", "3B")
    ]
    
    text = "The cat sat on the mat."
    
    for model_name, model_size in models:
        print(f"\n--- {model_size} 모델 분석 ---")
        
        try:
            visualizer = AttentionHeatmapVisualizer(model_name)
            attention_weights, tokens = visualizer.get_attention_weights(text)
            
            if attention_weights is not None:
                # 첫 번째 레이어의 첫 번째 head 시각화
                visualizer.plot_attention_heatmap(
                    attention_weights, tokens, 
                    layer_idx=0, head_idx=0,
                    save_path=os.path.join(os.path.dirname(__file__), f"comparison_{model_size}_attention.png")
                )
                
                # Attention 패턴 분석
                visualizer.analyze_attention_patterns(attention_weights, tokens, layer_idx=0)
                
        except Exception as e:
            print(f"Error with {model_size} model: {e}")

def analyze_different_layers():
    """다른 레이어들의 attention 패턴 분석"""
    print("\n=== 다른 레이어 분석 ===")
    
    visualizer = AttentionHeatmapVisualizer("Qwen/Qwen2.5-7B-Instruct")
    text = "Sound symbolism is the study of how sounds relate to meaning."
    
    attention_weights, tokens = visualizer.get_attention_weights(text)
    
    if attention_weights is not None:
        # 여러 레이어의 attention 시각화
        layers_to_analyze = [0, 5, 10, 15]  # 첫 번째, 중간, 마지막 레이어들
        
        for layer_idx in layers_to_analyze:
            if layer_idx < len(attention_weights):
                print(f"\n--- Layer {layer_idx} 분석 ---")
                
                visualizer.plot_attention_heatmap(
                    attention_weights, tokens, 
                    layer_idx=layer_idx, head_idx=0,
                    save_path=os.path.join(os.path.dirname(__file__), f"layer_{layer_idx}_attention.png")
                )
                
                visualizer.analyze_attention_patterns(attention_weights, tokens, layer_idx)

def custom_text_analysis():
    """사용자 정의 텍스트 분석"""
    print("\n=== 사용자 정의 텍스트 분석 ===")
    
    # 여러 언어의 텍스트
    texts = [
        "The cat sat on the mat.",
        "Le chat s'est assis sur le tapis.",
        "猫はマットの上に座った。",
        "고양이가 매트 위에 앉았다."
    ]
    
    visualizer = AttentionHeatmapVisualizer("Qwen/Qwen2.5-7B-Instruct")
    
    for i, text in enumerate(texts):
        print(f"\n--- 텍스트 {i+1}: {text} ---")
        
        attention_weights, tokens = visualizer.get_attention_weights(text)
        
        if attention_weights is not None:
            visualizer.plot_attention_heatmap(
                attention_weights, tokens, 
                layer_idx=0, head_idx=0,
                save_path=f"custom_text_{i+1}_attention.png"
            )

if __name__ == "__main__":
    # 간단한 예시 실행
    simple_example()
    
    # 모델 비교 (선택적)
    compare_models()
    
    # 다른 레이어 분석 (선택적)
    analyze_different_layers()
    
    # 사용자 정의 텍스트 분석 (선택적)
    custom_text_analysis()
    
    print("\n=== 분석 완료 ===")
    print("생성된 이미지 파일들을 확인해보세요!") 