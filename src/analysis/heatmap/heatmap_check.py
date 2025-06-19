import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor
import warnings
import soundfile as sf

warnings.filterwarnings('ignore')

class AttentionHeatmapVisualizer:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Qwen2.5-Omni 모델의 attention layer를 시각화하는 클래스
        
        Args:
            model_name: 모델 이름 (7B 또는 3B)
            device: 사용할 디바이스
        """
        self.device = device
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # 오디오 입력을 위한 processor (Qwen2.5-Omni 계열은 processor 필요)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            self.processor = None
            print(f"[WARNING] Processor 로드 실패: {e}")
        
        # 모델을 evaluation 모드로 설정
        self.model.eval()
        
        # attention hook을 위한 변수들
        self.attention_maps = {}
        self.hooks = []
        
    def register_hooks(self):
        """모든 attention layer에 hook을 등록"""
        def get_attention_hook(name):
            def hook(module, input, output):
                # attention output을 저장
                if hasattr(output, 'last_hidden_state'):
                    # Multi-head attention의 경우
                    attention_weights = output.attentions[-1] if output.attentions else None
                    if attention_weights is not None:
                        self.attention_maps[name] = attention_weights.detach().cpu()
                else:
                    # 일반적인 attention의 경우
                    self.attention_maps[name] = output.detach().cpu()
            return hook
        
        # 모든 attention layer를 찾아서 hook 등록
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)
                print(f"Registered hook for: {name}")
    
    def remove_hooks(self):
        """등록된 hook들을 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps.clear()
    
    def get_attention_weights(self, text=None, audio_path=None, layer_idx=None, head_idx=None):
        """
        텍스트 또는 오디오 입력에 대한 attention weight를 추출
        Args:
            text: 입력 텍스트 (str)
            audio_path: 입력 오디오 파일 경로 (str)
            layer_idx: 특정 layer 인덱스 (None이면 모든 layer)
            head_idx: 특정 head 인덱스 (None이면 모든 head)
        Returns:
            attention_weights, tokens (또는 audio frame index)
        사용 예시:
            # 텍스트 입력
            attn, tokens = visualizer.get_attention_weights(text="hello world")
            # 오디오 입력
            attn, frames = visualizer.get_attention_weights(audio_path="audio.wav")
        """
        if (text is None and audio_path is None) or (text and audio_path):
            raise ValueError("text 또는 audio_path 중 하나만 입력해야 합니다.")
        self.attention_maps.clear()
        if text is not None:
            # 텍스트 입력
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            attention_weights = outputs.attentions
            if attention_weights is None:
                print("No attention weights found in model output")
                return None
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            return attention_weights, tokens
        else:
            # 오디오 입력
            if self.processor is None:
                raise RuntimeError("이 모델은 오디오 입력을 지원하지 않거나 processor가 필요합니다.")
            # 오디오 파일 로드
            audio, sr = sf.read(audio_path)
            # Qwen2.5-Omni는 16kHz mono를 기대함
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            # processor로 feature 추출
            inputs = self.processor(audios=audio, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            attention_weights = outputs.attentions
            if attention_weights is None:
                print("No attention weights found in model output (audio)")
                return None
            # 오디오 입력의 경우, 토큰 대신 frame index 반환
            frames = list(range(inputs["input_features"].shape[-1])) if "input_features" in inputs else None
            return attention_weights, frames
    
    def plot_attention_heatmap(self, attention_weights, tokens, layer_idx=0, head_idx=0, 
                              figsize=(12, 10), save_path=None):
        """
        Attention heatmap을 그리기
        
        Args:
            attention_weights: attention weight 텐서
            tokens: 토큰 리스트
            layer_idx: 시각화할 layer 인덱스
            head_idx: 시각화할 head 인덱스
            figsize: 그래프 크기
            save_path: 저장할 경로 (None이면 저장하지 않음)
        """
        if attention_weights is None:
            print("No attention weights provided")
            return
        
        # 특정 layer와 head의 attention weight 추출
        if layer_idx >= len(attention_weights):
            print(f"Layer index {layer_idx} out of range. Available layers: {len(attention_weights)}")
            return
        
        layer_attention = attention_weights[layer_idx]  # [batch, heads, seq_len, seq_len]
        
        if head_idx >= layer_attention.shape[1]:
            print(f"Head index {head_idx} out of range. Available heads: {layer_attention.shape[1]}")
            return
        
        attention_matrix = layer_attention[0, head_idx].cpu().numpy()  # [seq_len, seq_len]
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 토큰 라벨 생성 (긴 토큰은 축약)
        token_labels = []
        for token in tokens:
            if len(token) > 10:
                token_labels.append(token[:7] + "...")
            else:
                token_labels.append(token)
        
        # Heatmap 그리기
        sns.heatmap(
            attention_matrix,
            xticklabels=token_labels,
            yticklabels=token_labels,
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}\nModel: {self.model_name}')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        # x축 라벨 회전
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to: {save_path}")
        
        plt.show()
    
    def plot_multi_head_attention(self, attention_weights, tokens, layer_idx=0, 
                                 num_heads=8, figsize=(20, 16), save_path=None):
        """
        여러 head의 attention을 한 번에 시각화
        
        Args:
            attention_weights: attention weight 텐서
            tokens: 토큰 리스트
            layer_idx: 시각화할 layer 인덱스
            num_heads: 시각화할 head 개수
            figsize: 그래프 크기
            save_path: 저장할 경로
        """
        if attention_weights is None:
            print("No attention weights provided")
            return
        
        layer_attention = attention_weights[layer_idx]
        total_heads = layer_attention.shape[1]
        num_heads = min(num_heads, total_heads)
        
        # 서브플롯 생성
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        # 토큰 라벨 생성
        token_labels = []
        for token in tokens:
            if len(token) > 8:
                token_labels.append(token[:5] + "...")
            else:
                token_labels.append(token)
        
        for i in range(num_heads):
            attention_matrix = layer_attention[0, i].cpu().numpy()
            
            sns.heatmap(
                attention_matrix,
                xticklabels=token_labels if i % 4 == 3 else [],
                yticklabels=token_labels if i < 4 else [],
                cmap='Blues',
                ax=axes[i],
                cbar=False
            )
            
            axes[i].set_title(f'Head {i}')
        
        # 전체 제목
        fig.suptitle(f'Multi-Head Attention - Layer {layer_idx}\nModel: {self.model_name}', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-head heatmap to: {save_path}")
        
        plt.show()
    
    def analyze_attention_patterns(self, attention_weights, tokens, layer_idx=0):
        """
        Attention 패턴 분석
        
        Args:
            attention_weights: attention weight 텐서
            tokens: 토큰 리스트
            layer_idx: 분석할 layer 인덱스
        """
        if attention_weights is None:
            print("No attention weights provided")
            return
        
        layer_attention = attention_weights[layer_idx]
        
        print(f"=== Attention Pattern Analysis - Layer {layer_idx} ===")
        print(f"Model: {self.model_name}")
        print(f"Sequence length: {layer_attention.shape[-1]}")
        print(f"Number of heads: {layer_attention.shape[1]}")
        print(f"Number of layers: {len(attention_weights)}")
        
        # 각 head의 평균 attention weight
        mean_attention = layer_attention.mean(dim=(0, 2, 3))  # [heads]
        print(f"\nMean attention per head:")
        for i, mean_attn in enumerate(mean_attention):
            print(f"  Head {i}: {mean_attn:.4f}")
        
        # 가장 높은 attention을 받는 토큰 쌍 찾기
        max_attention = layer_attention.max(dim=1)[0]  # [batch, seq_len, seq_len]
        max_indices = max_attention[0].argmax(dim=1)  # [seq_len]
        
        print(f"\nHighest attention targets:")
        for i, target_idx in enumerate(max_indices):
            if i < len(tokens) and target_idx < len(tokens):
                print(f"  {tokens[i]} -> {tokens[target_idx]} (weight: {max_attention[0, i, target_idx]:.4f})")

def main():
    """메인 실행 함수"""
    # 테스트할 텍스트들
    test_texts = [
        "The cat sat on the mat.",
        "Le chat s'est assis sur le tapis.",
        "猫はマットの上に座った。",
        "Sound symbolism is the study of how sounds relate to meaning."
    ]
    
    # 모델들
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct"
    ]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Analyzing model: {model_name}")
        print(f"{'='*60}")
        
        try:
            # 시각화 객체 생성
            visualizer = AttentionHeatmapVisualizer(model_name)
            
            for i, text in enumerate(test_texts):
                print(f"\n--- Test {i+1}: {text[:50]}... ---")
                
                # Attention weights 추출
                attention_weights, tokens = visualizer.get_attention_weights(text)
                
                if attention_weights is not None:
                    # 단일 head heatmap
                    visualizer.plot_attention_heatmap(
                        attention_weights, tokens, 
                        layer_idx=0, head_idx=0,
                        save_path=f"attention_heatmap_{model_name.split('/')[-1]}_test{i+1}.png"
                    )
                    
                    # Multi-head heatmap (첫 번째 테스트만)
                    if i == 0:
                        visualizer.plot_multi_head_attention(
                            attention_weights, tokens, layer_idx=0,
                            save_path=f"multihead_attention_{model_name.split('/')[-1]}_test{i+1}.png"
                        )
                    
                    # Attention 패턴 분석
                    visualizer.analyze_attention_patterns(attention_weights, tokens, layer_idx=0)
                
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue

if __name__ == "__main__":
    main()
