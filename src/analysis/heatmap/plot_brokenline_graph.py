import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from typing import Dict, List, Tuple, Any

# multi_threading_attention_computation.py final_word_layer_stats.pkl feature_to_score.json, ipa_dict, dim_pairs
# f"np_{data_type}_{lang}_sampling_every_{sampling_rate}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}_processed_words_{processed_words}_word_stats.pkl"
# Load required data structures

# plot_brokenline_graph.py 파일에서 코드를 만들어주면 좋겠어. multi_threading_attention_computation.py 파일에서는 plot_sampled_word_heatmap function의 제목 설정 및 파일명 등, final_word_layer_stats variable의 데이터구조와 feature_to_score variable의 데이터, dim_pairs, 그리고 ipa_dict variable 등을 참고해서 코드를 구현해주면 좋겠어.
# 1. X축은 layer의 번호를 가리키게 할 것이야
# 2. Y축은 attention score 점수를 가리키게 할 것이야.
# 3. 내가 원하는 것은 주어진 데이터에 대하여, 하나의 semantic dimension pair에서 feature_to_score의 semantic dimension이 가리키는 ipa를 ipa_dict에서 가져와줘. feature_to_score는 key로 semdim pair를 가지는데, 그의 key-value는 양수가 semdim pair의 [1] index와 관련이 있음을, 음수가 semdim pair의 [0] index와 관련이 있음을 알려줘. 0인 경우에는 관계가 없다는 뜻이야. 예를 들어 만약 Abrupt-Continuous pairs에서 sonorants key의 value가 0.85라면, sonorants는 continuous와 관련이 있으므로 ipa_dict에서 sonorants를 찾아내면 돼.
# 4. 3단계에서 찾아온 ipa 중에서, 만약 final_word_stats가 이들 중 하나 이상의 ipa를 포함한다면 한 layer에 대하여 존재하는 그 모든 관련 ipa들을 찾아서 평균점수를 계산해주면 돼. 평균을 계산할 때 관련이 있다고 판단된 feature들끼리 ipa가 묶이면 돼(ex - abrupt feature가 voiced_fricatives와 voiceless_stops와 관련 있으면 이들 ipa가 모두 묶임). 이 상태에서 layer에 따른 ipa 음소군의 attention score 변화를 확인할 수 있도록 그래프를 그리면 돼.
# 5. legend는 포함되어야 해.
# 6. 꺾은선 그래프는 색깔이 아닌 점선의 모양과 꺽이는 지점의 모양으로 구분해줘. 이때 dim_pairs의 feature들은 각각 구분이 되도록 해줘야 해.
# 7. Y축의 범위는 0-1이 되게 해줘.
# 8. Y축 0.5를 기준으로 연한 점선을 그려줘.
# 9. 각 점의 상단에는 그 값을 보여줘. 소숫점 셋째짜리까지 보여주면 되는데 모두 0-1 사이의 값을 가지므로 1의 자리 숫자의 0을 생략하고 ".025"와 같이 작성해줘.
def load_data_structures():
    """Load the required data structures from the main computation file"""
    # Load feature_to_score
    feature_to_score = json.load(open("data/processed/art/resources/feature_to_score.json", "r"))

    # make keys be lowercase
    feature_to_score = {k.lower(): v for k, v in feature_to_score.items()}
    
    # Load ipa_dict
    ipa_dict = {
        "voiced_stops": ['b','d','ɟ','ɡ','ɖ','ɢ'],
        "voiceless_stops": ['p','t','c','k','ʈ','q'],
        "voiced_fricatives": ['v','ð','z','ʒ','ʐ','ʝ','ɣ','ʁ','ʕ','ɦ','β'],
        "voiceless_fricatives": ['f','s','ʃ','θ','ç','x','χ','ħ','h','ɸ','ʂ','ɬ'],
        "sonorants": ['m','ɱ','n','ɳ','ɲ','ŋ','ɴ','l','ɭ','ʎ','ʟ','ʀ','r','ɾ','ɽ','ɹ','ɺ','ɻ','j','ɰ'],
        "others": ['ʔ','w','ʡ','ʢ'],
        "front": ['i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ʘ', 'ɛ', 'œ', 'æ', 'a', 'ɶ'],
        "back": ['ɨ', 'ʉ', 'ɯ', 'u', 'ʊ', 'ɤ', 'o', 'ɜ', 'ɝ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ'],
    }
    
    # Define dim_pairs
    dim_pairs = [
        ("abrupt", "continuous"), ("continuous", "abrupt"), ("active", "passive"), ("passive", "active"),
        ("beautiful", "ugly"), ("ugly", "beautiful"), ("big", "small"), ("small", "big"),
        ("dangerous", "safe"), ("safe", "dangerous"), ("exciting", "calming"), ("calming", "exciting"),
        ("fast", "slow"), ("slow", "fast"), ("good", "bad"), ("bad", "good"),
        ("happy", "sad"), ("sad", "happy"), ("hard", "soft"), ("soft", "hard"),
        ("harsh", "mellow"), ("mellow", "harsh"), ("heavy", "light"), ("light", "heavy"),
        ("inhibited", "free"), ("free", "inhibited"), ("interesting", "uninteresting"), ("uninteresting", "interesting"),
        ("masculine", "feminine"), ("feminine", "masculine"), ("orginary", "unique"), ("unique", "orginary"),
        ("pleasant", "unpleasant"), ("unpleasant", "pleasant"), ("realistic", "fantastical"), ("fantastical", "realistic"),
        ("rugged", "delicate"), ("delicate", "rugged"), ("sharp", "round"), ("round", "sharp"),
        ("simple", "complex"), ("complex", "simple"), ("solid", "nonsolid"), ("nonsolid", "solid"),
        ("strong", "weak"), ("weak", "strong"), ("structured", "disorganized"), ("disorganized", "structured"),
        ("tense", "relaxed"), ("relaxed", "tense"),
    ]

    # dim_pairs = [
    #     # ("tense", "relaxed"), ("relaxed", "tense"),
    #     # ("sharp", "round"), ("round", "sharp"),
    #     # ("masculine", "feminine"), ("feminine", "masculine"),
    #     # ("exciting", "calming"), ("calming", "exciting"),
    # ]

    return feature_to_score, ipa_dict, dim_pairs

def get_related_features_for_dim_pair(dim_pair: Tuple[str, str], feature_to_score: Dict) -> Dict[str, List[str]]:
    """
    Get features related to each dimension in a pair
    Returns: {dim1: [features], dim2: [features]}
    """
    dim1, dim2 = dim_pair
    pair_key = f"{dim1}-{dim2}"
    
    related_features = {dim1: [], dim2: []}
    
    if pair_key in feature_to_score:
        for feature, score in feature_to_score[pair_key].items():
            if score > 0:  # Related to dim2 (second dimension)
                related_features[dim2].append(feature)
            elif score < 0:  # Related to dim1 (first dimension)
                related_features[dim1].append(feature)
            # score == 0 means no relationship, so we ignore

    return related_features

def get_ipa_symbols_for_features(features: List[str], ipa_dict: Dict) -> List[str]:
    """Get all IPA symbols for the given features"""
    ipa_symbols = []
    for feature in features:
        if feature in ipa_dict:
            ipa_symbols.extend(ipa_dict[feature])
    return list(set(ipa_symbols))  # Remove duplicates

def calculate_layer_average_score(layer_stats: Dict, ipa_symbols: List[str], dimension: str) -> float:
    """Calculate average attention score for given IPA symbols in a layer"""
    scores = []
    for ipa in ipa_symbols:
        if ipa in layer_stats and dimension in layer_stats[ipa]:
            scores.append(layer_stats[ipa][dimension])
    
    if scores:
        return np.mean(scores)
    return 0.0

def plot_broken_line_graph(final_word_layer_stats: Dict, 
                          feature_to_score: Dict, 
                          ipa_dict: Dict, 
                          dim_pairs: List[Tuple[str, str]],
                          data_type: str,
                          lang: str,
                          compute_rule: str,
                          layer_start: int,
                          layer_end: int,
                          check_model_response: bool,
                          sampling_rate: int,
                          save_path: str = None):
    """
    Plot broken line graph showing attention scores across layers for semantic dimension pairs
    """
    if save_path is None:
        save_path = 'results/plots/attention/broken_line_graphs/'
    os.makedirs(save_path, exist_ok=True)
    
    # Get unique dimension pairs (remove duplicates)
    unique_pairs = []
    seen_pairs = set()
    for dim1, dim2 in dim_pairs:
        if (dim1, dim2) not in seen_pairs and (dim2, dim1) not in seen_pairs:
            unique_pairs.append((dim1, dim2))
            seen_pairs.add((dim1, dim2))
    
    # Define line styles and markers for different features
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Track used styles and markers
    used_styles = set()
    used_markers = set()
    
    # Plot for each dimension pair
    for pair_idx, (dim1, dim2) in enumerate(unique_pairs):
        # Get related features for this pair
        related_features = get_related_features_for_dim_pair((dim1, dim2), feature_to_score)
        
        # Plot for each dimension
        for dim_idx, dimension in enumerate([dim1, dim2]):
            features = related_features[dimension]
            if not features:
                continue
            
            # Get IPA symbols for these features
            ipa_symbols = get_ipa_symbols_for_features(features, ipa_dict)
            if not ipa_symbols:
                continue
            
            # Check if any of these IPA symbols exist in our data
            has_data = False
            for layer in final_word_layer_stats.keys():
                for ipa in ipa_symbols:
                    if ipa in final_word_layer_stats[layer] and dimension in final_word_layer_stats[layer][ipa]:
                        has_data = True
                        break
                if has_data:
                    break
            
            if not has_data:
                continue

            # Calculate scores for each layer
            layers = sorted([int(l) for l in final_word_layer_stats.keys()])
            scores = []
            
            for layer in layers:
                if layer in final_word_layer_stats:
                    score = calculate_layer_average_score(final_word_layer_stats[layer], ipa_symbols, dimension)
                    scores.append(score)
                else:
                    scores.append(0.0)
            
            # Choose line style and marker
            style_idx = len(used_styles) % len(line_styles)
            marker_idx = len(used_markers) % len(markers)
            
            line_style = line_styles[style_idx]
            marker = markers[marker_idx]
            
            used_styles.add(line_style)
            used_markers.add(marker)
            
            # Create label
            feature_names = ', '.join(features)
            label = f"{dimension} ({feature_names})"
            
            # Plot the line
            line = ax.plot(layers, scores, linestyle=line_style, marker=marker, 
                          linewidth=2, markersize=6, label=label)
            
            # # Add value annotations
            # for i, (layer, score) in enumerate(zip(layers, scores)):
            #     if score > 0:  # Only annotate if score is not zero
            #         # Format score as .XXX (remove leading 0)
            #         score_str = f"{score:.3f}".lstrip('0')
            #         ax.annotate(score_str, (layer, score), 
            #                    xytext=(0, 10), textcoords='offset points',
            #                    ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attention Score', fontsize=12)
    ax.set_title(f'Layer-wise Attention Scores for Semantic Dimensions\n'
                f'Data: {data_type} | Lang: {lang} | Rule: {compute_rule} | '
                f'L{layer_start}-L{layer_end} | Check: {check_model_response} | '
                f'Sampling: {sampling_rate}', fontsize=14, pad=20)
    
    # Set y-axis range to 0-1
    ax.set_ylim(0, 1)
    
    # Add horizontal line at y=0.5
    ax.axhline(y=0.5, color='lightgray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    file_name = f"broken_line_{data_type}_{lang}_{compute_rule}_L{layer_start}_L{layer_end}_check{check_model_response}_sampling{sampling_rate}.png"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Broken line graph saved to {file_path}")
    plt.close()

def load_and_plot_from_file(file_path: str, 
                           data_type: str,
                           lang: str,
                           compute_rule: str,
                           layer_start: int,
                           layer_end: int,
                           check_model_response: bool,
                           sampling_rate: int):
    """
    Load final_word_layer_stats from file and create broken line graph
    """
    # Load data structures
    feature_to_score, ipa_dict, dim_pairs = load_data_structures()
    
    # Load final_word_layer_stats
    layer_file_path = file_path.replace('.pkl', '_layer_stats.pkl')
    if os.path.exists(layer_file_path):
        with open(layer_file_path, 'rb') as f:
            final_word_layer_stats = pkl.load(f)
        
        # Create the plot
        plot_broken_line_graph(
            final_word_layer_stats=final_word_layer_stats,
            feature_to_score=feature_to_score,
            ipa_dict=ipa_dict,
            dim_pairs=dim_pairs,
            data_type=data_type,
            lang=lang,
            compute_rule=compute_rule,
            layer_start=layer_start,
            layer_end=layer_end,
            check_model_response=check_model_response,
            sampling_rate=sampling_rate
        )
    else:
        print(f"Layer stats file not found: {layer_file_path}")

# Example usage
if __name__ == "__main__":
    # Example parameters
    file_path = "src/analysis/heatmap/results/np_audio_Constructed_fraction_check_model_response_True_0_27_sampling_every_1_processed_words_2665.pkl"
    data_type = "audio"
    lang = "Constructed"
    compute_rule = "fraction"
    layer_start = 0
    layer_end = 27
    check_model_response = True
    sampling_rate = 1
    
    # Load and plot
    load_and_plot_from_file(
        file_path=file_path,
        data_type=data_type,
        lang=lang,
        compute_rule=compute_rule,
        layer_start=layer_start,
        layer_end=layer_end,
        check_model_response=check_model_response,
        sampling_rate=sampling_rate
    )
