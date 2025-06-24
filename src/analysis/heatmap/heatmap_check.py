import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import os
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
import librosa
import json
import re

# python src/analysis/heatmap/heatmap_check.py --model Qwen/Qwen2.5-Omni-7B --data data/prompts/understanding/pair_matching/audiolm/masked_meaning_to_word_mcq_no_dialogue-en.json --output src/analysis/heatmap/plots --head 0
PROMPT_JSON_PATH = os.path.join(os.path.dirname(__file__), '../../../data/prompts/prompts.json')

class QwenOmniAttentionVisualizer:
    def __init__(
            self,
            model_path: str,
            data_path: str,
            output_dir: str,
            exp_name: str,
            max_tokens: int = 32,
            temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.exp_name = exp_name

        # Load Qwen Omni model
        print(f"Loading Qwen Omni model from {self.model_path}")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2",
            attn_implementation="eager",
        )
        self.model.disable_talker()
        
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

    def run_mcq_experiment(self, visualize_sample_indices=None, layer_gap=3, head=0, use_hooks=False, hook_type='simple'):
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")
        
        # Debug model structure if using hooks
        if use_hooks:
            print("Debugging model structure for hook placement...")
            self.debug_model_structure()
        
        all_results = []
        if visualize_sample_indices is None:
            visualize_sample_indices = []
        for idx, query in enumerate(tqdm(mcq_data)):
            word = query['meta_data']['word']
            language = query['meta_data']['language']
            if '<AUDIO>' in query['question']:
                question_first_part = query['question'].split("<AUDIO>")[0]
                question_second_part = query['question'].split("<AUDIO>")[1]
                audio_path = f'data/processed/nat/tts/{language}/{word}.wav'
                prompt = f"{question_first_part}<AUDIO>{question_second_part}"
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
                            {"type": "text", "text": question_first_part},
                            {"type": "audio", "audio": audio_path},
                            {"type": "text", "text": question_second_part},
                        ],
                    },
                ]
            else: # meaning -> word
                question_parts = re.split(r'<AUDIO: .*?>', query['question'])
                option_audio_paths = []
                for option in query['options_info']:
                    # breakpoint()
                    option_audio_paths.append(f'data/processed/nat/tts/{option["language"]}/{option["text"]}.wav')
                    if not os.path.exists(option_audio_paths[-1]):
                        raise FileNotFoundError(f"Audio file not found: {option_audio_paths[-1]}")
                content = [{"type": "text", "text": question_parts[0]}]
                for i in range(len(option_audio_paths)):
                    content.append({"type": "audio", "audio": option_audio_paths[i]})
                    if i + 1 < len(question_parts):
                        content.append({"type": "text", "text": question_parts[i + 1]})
                prompt = query['question']
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ]
            USE_AUDIO_IN_VIDEO = True
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            with torch.no_grad():
                text_ids = self.model.generate(
                    **inputs, 
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    max_new_tokens=self.max_tokens,
                )
            full_text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print('Full text:', full_text)
            if "assistant\n" in full_text:
                model_answer = full_text.split("assistant\n")[-1].strip()
            else:
                raise ValueError(f"Unexpected format in model output: {full_text}")
            answer_match = re.search(r'\d+', model_answer)
            if answer_match:
                extracted_answer = answer_match.group(0)
            else:
                extracted_answer = None
            if extracted_answer is None:
                print(f"Warning: Model output is empty for query: {query['question'][:50]}...")
                extracted_answer = "0"
            try:
                is_correct = int(extracted_answer) == query['answer']
            except ValueError:
                print(f"Warning: Model output '{extracted_answer}' is not a valid integer. Marking as incorrect.")
                is_correct = False
            result = {
                "query": query['question'],
                "correct_answer": query['answer'],
                "model_answer": extracted_answer,
                "full_response": model_answer,
                "is_correct": is_correct
            }
            all_results.append(result)

            # === Attention Visualization & Score Plot ===
            if idx in visualize_sample_indices:
                # Use the existing conversation for attention extraction
                USE_AUDIO_IN_VIDEO = True
                att_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                att_audios, att_images, att_videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                att_inputs = self.processor(
                    text=att_text,
                    audio=att_audios,
                    images=att_images,
                    videos=att_videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO
                )
                att_inputs = att_inputs.to(self.model.device).to(self.model.dtype)
                
                options = [opt["text"] if isinstance(opt, dict) and "text" in opt else str(opt) for opt in query.get("options_info", query.get("options", []))]
                option_type = "number"
                num_options = len(options)
                
                # Choose attention extraction method
                if use_hooks:
                    print(f"[INFO] Using {hook_type} hook-based attention extraction for sample {idx}")
                    if hook_type == 'simple':
                        attentions, tokens = self.get_attention_with_simple_hooks(att_inputs)
                    elif hook_type == 'qwen':
                        attentions, tokens = self.get_attention_with_qwen_hooks(att_inputs)
                    elif hook_type == 'forward':
                        attentions, tokens = self.get_attention_with_forward_hooks(att_inputs)
                    else:
                        attentions, tokens = self.get_attention_and_tokens(att_inputs)
                else:
                    attentions, tokens = self.get_attention_and_tokens(att_inputs)
                
                option_indices = self.find_option_token_indices(tokens, option_type=option_type, num_options=num_options)
                audio_token_indices = self.find_audio_token_indices(tokens)
                print(f"[INFO] Sample {idx} - Audio token indices: {audio_token_indices}")
                print(f"[INFO] Sample {idx} - Option token indices: {option_indices}")
                num_layers = len(attentions)
                layers_to_check = [0] + [i for i in range(layer_gap, num_layers-1, layer_gap)] + [num_layers-1]
                for lidx in sorted(set(layers_to_check)):
                    attn = attentions[lidx][0, head].cpu().float().numpy()
                    title = f"Attention Heatmap (Sample {idx}, Layer {lidx}, Head {head})"
                    spath = os.path.join(self.output_dir, f"sample{idx}_layer{lidx}_head{head}_attn.png")
                    self.plot_attention_heatmap(
                        attn, tokens,
                        option_indices=option_indices,
                        correct_option_idx=query['answer'] if query['answer'] is not None else 0,
                        audio_token_indices=audio_token_indices,
                        highlight_audio=True,
                        highlight_option=True,
                        save_path=spath,
                        title=title
                    )
                attn_stats = self.analyze_attention_across_layers(attentions, tokens, audio_token_indices, option_indices, query['answer'] if query['answer'] is not None else 0, layer_gap=layer_gap)
                print(f"[ATTN SCORE SUMMARY] Sample {idx}")
                for stat in attn_stats:
                    print(f"Layer {stat['layer']}: audio->option sum={stat['audio_to_option_sum']:.4f}, option->audio sum={stat['option_to_audio_sum']:.4f}")
                # 꺾은선 그래프
                layers = [stat['layer'] for stat in attn_stats]
                audio2opt = [stat['audio_to_option_sum'] for stat in attn_stats]
                opt2audio = [stat['option_to_audio_sum'] for stat in attn_stats]
                plt.figure(figsize=(8,5))
                plt.plot(layers, audio2opt, marker='o', label='Audio→Option attention sum')
                plt.plot(layers, opt2audio, marker='s', label='Option→Audio attention sum')
                plt.xlabel('Layer')
                plt.ylabel('Attention Score Sum')
                plt.title(f'Attention Score Sum Across Layers (Sample {idx})')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plot_path = os.path.join(self.output_dir, f"sample{idx}_attn_score_flow.png")
                plt.savefig(plot_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"[INFO] Saved attention score plot: {plot_path}")
        correct_count = sum(1 for r in all_results if r["is_correct"])
        total_count = len(all_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Experiment completed. Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        model_name = os.path.basename(self.model_path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        results_filename = f"{self.output_dir}/{self.data_path.split('/')[-1].replace('.json', '')}_{model_name}.json"
        results_dict = {
            "model": self.model_path,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": all_results,
        }
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        print(f"Results saved to: {results_filename}")
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return results_dict, results_filename

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

    def prepare_multimodal_input(self, text):
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
                    {"type": "text", "text": text},
                ],
            },
        ]
        USE_AUDIO_IN_VIDEO = True
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(
            text=prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        return inputs, prompt

    def get_attention_and_tokens(self, inputs):
        """Extract attention from Qwen2.5-Omni model by accessing internal thinker model"""
        with torch.no_grad():
            thinker_model = self.model.thinker.model # Newly added. Test and remove if not needed.
            
            # Call the thinker model's forward method
            outputs = thinker_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True,
                return_dict=True
            )
        
        attentions = outputs.attentions
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # if 'input_ids' in inputs:
        #     tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # else:
        #     tokens = [str(i) for i in range(attentions[0].shape[-1])]
        return attentions, tokens

    def get_attention_with_hooks(self, inputs):
        """Extract attention using PyTorch hooks - more elegant approach"""
        attention_outputs = []
        
        def attention_hook(module, input, output):
            # For multi-head attention, output is typically (batch, seq_len, seq_len, num_heads)
            # or (batch, num_heads, seq_len, seq_len)
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_outputs.extend(output.attentions)
            elif isinstance(output, tuple) and len(output) > 1 and hasattr(output[1], 'attentions'):
                attention_outputs.extend(output[1].attentions)
        
        # Register hooks on attention layers
        hooks = []
        thinker_model = self.model.thinker.model
        
        # Find all attention layers in the model
        for name, module in thinker_model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = thinker_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True,
                    return_dict=True
                )
            
            # If hooks didn't capture attention, fall back to outputs
            if not attention_outputs and hasattr(outputs, 'attentions'):
                attention_outputs = outputs.attentions
            
            tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            return attention_outputs, tokens
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

    def get_attention_with_forward_hooks(self, inputs):
        """Alternative approach using forward hooks on specific attention modules"""
        attention_maps = []
        
        def attention_forward_hook(module, input, output):
            # For Qwen2.5-Omni, attention output might be in a specific format
            if isinstance(output, tuple):
                # Some models return (hidden_states, attention_weights)
                if len(output) > 1 and output[1] is not None:
                    attention_maps.append(output[1])
            elif hasattr(output, 'attentions'):
                attention_maps.extend(output.attentions)
        
        hooks = []
        thinker_model = self.model.thinker.model
        
        # Register hooks on transformer blocks
        for name, module in thinker_model.named_modules():
            if 'block' in name or 'layer' in name:
                # Try to find attention submodules
                for subname, submodule in module.named_modules():
                    if 'attn' in subname.lower() or 'attention' in subname.lower():
                        hook = submodule.register_forward_hook(attention_forward_hook)
                        hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = thinker_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True,
                    return_dict=True
                )
            
            # Fallback to direct output if hooks didn't work
            if not attention_maps and hasattr(outputs, 'attentions'):
                attention_maps = outputs.attentions
            
            tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            return attention_maps, tokens
            
        finally:
            for hook in hooks:
                hook.remove()

    def get_attention_with_qwen_hooks(self, inputs):
        """Specialized hook method for Qwen2.5-Omni architecture"""
        attention_scores = []
        
        def attention_score_hook(module, input, output):
            """Hook to capture attention scores from Qwen attention modules"""
            # For Qwen models, attention scores are typically computed in the attention module
            # and stored in the module's state or passed through output
            if hasattr(module, '_attention_scores'):
                attention_scores.append(module._attention_scores)
            elif isinstance(output, tuple) and len(output) > 1:
                # Check if attention weights are in the output tuple
                if isinstance(output[1], torch.Tensor) and len(output[1].shape) >= 3:
                    attention_scores.append(output[1])
        
        def attention_compute_hook(module, input, output):
            """Hook to capture attention during computation"""
            # This hook captures the attention computation process
            if hasattr(module, 'attn_weights'):
                attention_scores.append(module.attn_weights)
        
        hooks = []
        thinker_model = self.model.thinker.model
        
        # Find and hook attention modules in Qwen architecture
        for name, module in thinker_model.named_modules():
            # Look for specific attention module patterns in Qwen
            if any(pattern in name.lower() for pattern in ['attn', 'attention', 'self_attn']):
                # Register hook for attention score capture
                hook1 = module.register_forward_hook(attention_score_hook)
                hook2 = module.register_forward_hook(attention_compute_hook)
                hooks.extend([hook1, hook2])
                
                # Also try to hook the compute_attention method if it exists
                if hasattr(module, 'compute_attention'):
                    original_compute = module.compute_attention
                    
                    def compute_hook(*args, **kwargs):
                        result = original_compute(*args, **kwargs)
                        if isinstance(result, tuple) and len(result) > 1:
                            attention_scores.append(result[1])
                        return result
                    
                    module.compute_attention = compute_hook
                    hooks.append(lambda: setattr(module, 'compute_attention', original_compute))
        
        try:
            with torch.no_grad():
                outputs = thinker_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True,
                    return_dict=True
                )
            
            # If hooks didn't capture anything, use the direct output
            if not attention_scores and hasattr(outputs, 'attentions'):
                attention_scores = outputs.attentions
            
            tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            return attention_scores, tokens
            
        finally:
            # Clean up hooks
            for hook in hooks:
                if callable(hook):
                    hook()
                else:
                    hook.remove()

    def debug_model_structure(self):
        """Debug method to understand the model structure for hook placement"""
        thinker_model = self.model.thinker.model
        print("=== Qwen2.5-Omni Thinker Model Structure ===")
        
        attention_modules = []
        for name, module in thinker_model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                attention_modules.append((name, type(module).__name__))
                print(f"Attention module: {name} ({type(module).__name__})")
        
        print(f"\nTotal attention modules found: {len(attention_modules)}")
        return attention_modules

    def find_option_token_indices(self, tokens, option_type="number", num_options=4):
        indices = []
        for i in range(num_options):
            if option_type == "number":
                pattern = f"{i+1}."
            else:
                pattern = chr(ord('A') + i) + "."
            found = None
            for idx, t in enumerate(tokens):
                if pattern in t:
                    found = idx
                    break
            indices.append(found)
        return indices

    def find_audio_token_indices(self, tokens):
        indices = [i for i, t in enumerate(tokens) if "<|audio|>" in t or "<audio>" in t]
        return indices

    def analyze_attention_across_layers(self, attentions, tokens, audio_token_indices, option_indices, correct_option_idx, layer_gap=3):
        num_layers = len(attentions)
        results = []
        layers_to_check = [0] + [i for i in range(layer_gap, num_layers-1, layer_gap)] + [num_layers-1]
        for layer in sorted(set(layers_to_check)):
            attn = attentions[layer][0, 0].cpu().float().numpy()
            audio_sum = 0.0
            option_sum = 0.0
            for aidx in audio_token_indices:
                oidx = option_indices[correct_option_idx] if correct_option_idx is not None else None
                if oidx is not None:
                    audio_sum += attn[aidx, oidx]
                    option_sum += attn[oidx, aidx]
            results.append({
                "layer": layer,
                "audio_to_option_sum": audio_sum,
                "option_to_audio_sum": option_sum
            })
        return results

    def get_attention_with_simple_hooks(self, inputs):
        """Simplified hook method that directly captures attention from Qwen2.5-Omni"""
        captured_attentions = []
        
        def simple_attention_hook(module, input, output):
            """Simple hook that captures attention weights"""
            # For most transformer models, attention weights are in the output
            if isinstance(output, tuple) and len(output) > 1:
                # Check if the second element looks like attention weights
                potential_attn = output[1]
                if isinstance(potential_attn, torch.Tensor) and len(potential_attn.shape) >= 3:
                    captured_attentions.append(potential_attn)
            elif hasattr(output, 'attentions'):
                captured_attentions.extend(output.attentions)
        
        hooks = []
        thinker_model = self.model.thinker.model
        
        # Hook into transformer layers
        for name, module in thinker_model.named_modules():
            # Look for transformer blocks or layers
            if any(pattern in name.lower() for pattern in ['block', 'layer', 'transformer']):
                # Check if this module has attention submodules
                has_attention = any('attn' in subname.lower() for subname, _ in module.named_modules())
                if has_attention:
                    hook = module.register_forward_hook(simple_attention_hook)
                    hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = thinker_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True,
                    return_dict=True
                )
            
            # If hooks didn't work, use the direct output
            if not captured_attentions and hasattr(outputs, 'attentions'):
                captured_attentions = outputs.attentions
            
            tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            return captured_attentions, tokens
            
        finally:
            for hook in hooks:
                hook.remove()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Qwen2.5-Omni MCQ attention heatmap from MCQ JSON file (audio+text, word_to_meaning_audio prompt)")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model name (7B or 3B)")
    parser.add_argument('--data', type=str, default="data/processed/nat/mcq/masked_meaning_to_word_mcq_no_dialogue-en.json", help="Path to MCQ JSON file (e.g. masked_meaning_to_word_mcq_no_dialogue-en.json)")
    parser.add_argument('--output', type=str, default="src/analysis/heatmap/heatmap_results", help="Path to save heatmap image")
    parser.add_argument('--max-tokens', type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('--sample', type=int, default=0, help="Index of the sample to visualize (for attention plot)")
    parser.add_argument('--layer-gap', type=int, default=3, help="Layer gap for attention score plot")
    parser.add_argument('--head', type=int, default=0, help="Head index")
    parser.add_argument('--exp_name', type=str, default="word_to_meaning_audio", help="Experiment name")
    parser.add_argument('--use-hooks', action='store_true', help="Use new hook-based attention extraction methods")
    parser.add_argument('--hook-type', type=str, default='simple', choices=['simple', 'qwen', 'forward'], 
                       help="Type of hook to use: simple, qwen, or forward")
    args = parser.parse_args()

    visualizer = QwenOmniAttentionVisualizer(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output, 
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        exp_name=args.exp_name,
    )
    # Only visualize the sample specified by --sample
    visualizer.run_mcq_experiment(visualize_sample_indices=[args.sample], layer_gap=args.layer_gap, head=args.head, use_hooks=args.use_hooks, hook_type=args.hook_type)
    