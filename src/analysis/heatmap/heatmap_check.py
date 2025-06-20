import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import librosa
import json

PROMPT_JSON_PATH = os.path.join(os.path.dirname(__file__), '../../data/prompts/prompts.json')

class QwenOmniAttentionVisualizer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def prepare_multimodal_input(self, text, audio_path):
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        conversation = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": text}
            ]}
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=prompt,
            audio=audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs, prompt

    def get_attention_and_tokens(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, return_dict=True)
        attentions = outputs.attentions
        input_ids = inputs.get('input_ids', None)
        if input_ids is not None:
            tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        else:
            tokens = [str(i) for i in range(attentions[-1].shape[-1])]
        return attentions, tokens

    def find_option_token_indices(self, tokens, option_type="number", num_options=4):
        if option_type == "number":
            option_prefixes = [f"{i+1}." for i in range(num_options)]
        elif option_type == "alpha":
            option_prefixes = [f"{chr(97+i)}." for i in range(num_options)]
        else:
            raise ValueError("option_type must be 'number' or 'alpha'")
        indices = []
        for prefix in option_prefixes:
            found = False
            for idx, tok in enumerate(tokens):
                if prefix in tok:
                    indices.append(idx)
                    found = True
                    break
            if not found:
                indices.append(None)
        return indices

    def find_audio_token_indices(self, tokens):
        audio_token_indices = [i for i, t in enumerate(tokens) if "audio" in t or t.startswith("<|extra_id_")]
        return audio_token_indices

    def plot_attention_heatmap(self, attention_matrix, tokens, option_indices=None, correct_option_idx=None, audio_token_indices=None, highlight_audio=True, highlight_option=True, save_path=None, title=None):
        seq_len = len(tokens)
        fig, ax = plt.subplots(figsize=(max(10, seq_len//2), max(8, seq_len//2)))
        token_labels = [t if len(t) < 12 else t[:9]+"..." for t in tokens]
        sns.heatmap(attention_matrix, xticklabels=token_labels, yticklabels=token_labels, cmap='Blues', ax=ax, cbar_kws={'label': 'Attention Weight'})
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        if title:
            ax.set_title(title)
        if highlight_audio and audio_token_indices:
            for idx in audio_token_indices:
                ax.add_patch(plt.Rectangle((idx, -0.5), 1, seq_len, fill=False, edgecolor='red', lw=2, linestyle='--'))
                ax.add_patch(plt.Rectangle((-0.5, idx), seq_len, 1, fill=False, edgecolor='red', lw=2, linestyle='--'))
        if highlight_option and option_indices and correct_option_idx is not None:
            opt_idx = option_indices[correct_option_idx]
            if opt_idx is not None:
                ax.add_patch(plt.Rectangle((opt_idx, -0.5), 1, seq_len, fill=False, edgecolor='green', lw=2))
                ax.add_patch(plt.Rectangle((-0.5, opt_idx), seq_len, 1, fill=False, edgecolor='green', lw=2))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved heatmap: {save_path}")
        plt.show()

    def visualize_mcq_attention(self, text, audio_path, correct_option_idx=0, option_type="number", num_options=4, layer=None, head=0, save_path=None):
        inputs, prompt = self.prepare_multimodal_input(text, audio_path)
        print("[DEBUG] Model prompt:")
        print(prompt)
        attentions, tokens = self.get_attention_and_tokens(inputs)
        if layer is None:
            layer = len(attentions) - 1
        attn = attentions[layer][0, head].cpu().numpy()
        option_indices = self.find_option_token_indices(tokens, option_type=option_type, num_options=num_options)
        audio_token_indices = self.find_audio_token_indices(tokens)
        print("[INFO] Audio token indices:", audio_token_indices)
        print("[INFO] Option token indices:", option_indices)
        self.plot_attention_heatmap(
            attn, tokens,
            option_indices=option_indices,
            correct_option_idx=correct_option_idx,
            audio_token_indices=audio_token_indices,
            highlight_audio=True,
            highlight_option=True,
            save_path=save_path,
            title=f"Attention Heatmap (Layer {layer}, Head {head})"
        )
        for idx in audio_token_indices:
            print(f"Audio token at index {idx}: context window: {tokens[max(0, idx-3):idx+4]}")

def load_prompt_template(language="en"):
    with open(PROMPT_JSON_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["word_meaning_matching"]["word_to_meaning_audio"]["user_prompt"]

def build_mcq_prompt(word, options, language="en"):
    template = load_prompt_template(language)
    options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    prompt = template.format(word=word, options=options_str, MAX_OPTION=len(options))
    return prompt

def get_audio_path(word, language):
    return f"data/processed/nat/tts/{language}/{word}.wav"

def extract_sample_info(sample):
    """
    Extracts word, options, language, correct answer index from a sample dict.
    """
    # Try to get word, language, options, answer
    word = sample.get("word")
    language = sample.get("language")
    if not word and "meta_data" in sample:
        word = sample["meta_data"].get("word")
    if not language and "meta_data" in sample:
        language = sample["meta_data"].get("language")
    # Options: try 'options_info' (list of dicts with 'text'), else 'options' (list of str)
    if "options_info" in sample:
        options = [opt["text"] if isinstance(opt, dict) and "text" in opt else str(opt) for opt in sample["options_info"]]
    elif "options" in sample:
        options = [str(opt) for opt in sample["options"]]
    else:
        options = []
    # Correct answer: try 'answer', else 'meta_data.answer'
    correct = sample.get("answer")
    if correct is None and "meta_data" in sample:
        correct = sample["meta_data"].get("answer")
    # Option type: number (default), or alpha if specified
    option_type = "number"
    return word, options, language, correct, option_type

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Qwen2.5-Omni MCQ attention heatmap from MCQ JSON file (audio+text, word_to_meaning_audio prompt)")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model name (7B or 3B)")
    parser.add_argument('--json', type=str, required=True, help="Path to MCQ JSON file (e.g. masked_meaning_to_word_mcq_no_dialogue-en.json)")
    parser.add_argument('--sample', type=int, default=0, help="Index of the sample to visualize")
    parser.add_argument('--layer', type=int, default=None, help="Layer index (default: last)")
    parser.add_argument('--head', type=int, default=0, help="Head index")
    parser.add_argument('--save', type=str, default=None, help="Path to save heatmap image")
    args = parser.parse_args()

    # Load MCQ data
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of samples.")
    if args.sample < 0 or args.sample >= len(data):
        raise IndexError(f"Sample index {args.sample} out of range (0-{len(data)-1})")
    sample = data[args.sample]
    word, options, language, correct, option_type = extract_sample_info(sample)
    if not word or not language or not options:
        raise ValueError(f"Sample missing required fields: word={word}, language={language}, options={options}")
    audio_path = get_audio_path(word, language)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    prompt = build_mcq_prompt(word, options, language=language)
    print(f"[INFO] Visualizing sample {args.sample} from {args.json}")
    print(f"[INFO] Word: {word}, Language: {language}, Options: {options}, Correct: {correct}")
    visualizer = QwenOmniAttentionVisualizer(args.model)
    visualizer.visualize_mcq_attention(
        text=prompt,
        audio_path=audio_path,
        correct_option_idx=correct if correct is not None else 0,
        option_type=option_type,
        num_options=len(options),
        layer=args.layer,
        head=args.head,
        save_path=args.save
    )
