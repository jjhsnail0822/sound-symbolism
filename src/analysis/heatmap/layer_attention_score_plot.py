import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from typing import Dict, List, Tuple, Any

# multi_threading_attention_computation.py final_word_layer_stats.pkl feature_to_score.json, ipa_dict, dim_pairs
# f"np_{data_type}_{lang}_sampling_every_{sampling_rate}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}_processed_words_{processed_words}_word_stats.pkl"
# Load required data structures

def load_data_structures():
    # Load feature_to_score
    ipa_feature_to_score = json.load(open("data/processed/art/resources/feature_to_score.json", "r"))

    # make keys be lowercase
    ipa_feature_to_score = {k.lower(): v for k, v in ipa_feature_to_score.items()}

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
        ("abrupt", "continuous"),
        ("passive", "active"),
        ("beautiful", "ugly"),
        ("big", "small"),
        ("dangerous", "safe"),
        ("exciting", "calming"),
        ("fast", "slow"),
        ("good", "bad"),
        ("happy", "sad"),
        ("hard", "soft"),
        ("harsh", "mellow"),
        ("heavy", "light"),
        ("inhibited", "free"),
        ("interesting", "uninteresting"),
        ("masculine", "feminine"),
        ("orginary", "unique"),
        ("pleasant", "unpleasant"),
        ("realistic", "fantastical"),
        ("delicate", "rugged"),
        ("sharp", "round"),
        ("simple", "complex"),
        ("solid", "nonsolid"),
        ("strong", "weak"),
        ("tense", "relaxed"),
        ("structured", "disorganized"),
    ]

    # dim_pairs = [
    #     ("tense", "relaxed"),
    #     ("sharp", "round"),
    #     ("masculine", "feminine"),
    #     ("exciting", "calming"),
    # ]

    # dim_pairs = [
    #     ("tense", "relaxed"),
    #     ("masculine", "feminine"),
    #     ("realistic", "fantastical"),
    #     ("fast", "slow"),
    # ]

    return ipa_feature_to_score, ipa_dict, dim_pairs

def get_related_ipa_features_for_dim_pair(dim_pair: Tuple[str, str], ipa_feature_to_score: Dict) -> Dict[str, List[str]]:
    dim1, dim2 = dim_pair
    pair_key = f"{dim1}-{dim2}"
    
    related_ipa_features = {dim1: [], dim2: []}
    
    if pair_key in ipa_feature_to_score:
        for feature, score in ipa_feature_to_score[pair_key].items():
            if score > 0:  # Related to dim2 (second dimension)
                related_ipa_features[dim2].append(feature)
            elif score < 0:  # Related to dim1 (first dimension)
                related_ipa_features[dim1].append(feature)
            # score == 0 means no relationship, so we ignore

    print(f"Related IPA features for {dim_pair}: {related_ipa_features}")
    return related_ipa_features

def get_ipa_symbols_for_ipa_features(ipa_features: List[str], ipa_dict: Dict[str, List[str]]) -> List[str]:
    ipa_symbols = []
    for ipa_feature in ipa_features:
        if ipa_feature in ipa_dict:
            ipa_symbols.extend(ipa_dict[ipa_feature])
        else:
            print(f"IPA feature '{ipa_feature}' not found in IPA dictionary.")
    return ipa_symbols

def calculate_layer_average_score(layer_stats: Dict, 
                                  ipa_symbols_1: List[str], 
                                  ipa_symbols_2: List[str], 
                                  dim_feature1: str, 
                                  dim_feature2: str) -> float:
    scores = []
    
    for ipa in ipa_symbols_1:
        if ipa in layer_stats and dim_feature1 in layer_stats[ipa]:
            scores.append(layer_stats[ipa][dim_feature1])

    for ipa in ipa_symbols_2:
        if ipa in layer_stats and dim_feature2 in layer_stats[ipa]:
            scores.append(layer_stats[ipa][dim_feature2])

    if scores:
        return np.mean(scores)
    return 0.0


def plot_broken_line_graph(stats_data: Dict[str, Dict], 
                          feature_to_score: Dict, 
                          ipa_dict: Dict, 
                          dim_pairs: List[Tuple[str, str]],
                          lang: str,
                          compute_rule: str,
                          layer_start: int,
                          layer_end: int,
                          check_model_response: bool,
                          sampling_rate: int,
                          save_path: str = None,
                          average_all_dims: bool = False):
    if save_path is None:
        save_path = 'results/plots'
    os.makedirs(save_path, exist_ok=True)

    # Define line styles and markers for different features
    line_styles = ['-', '-.', ':', '--']
    markers = ['o', 's', '^', 'v']
    colors = ['blue', 'red', 'green', 'purple']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use layers from the first dataset as a reference
    first_key = list(stats_data.keys())[0]
    final_word_layer_stats_ref = stats_data[first_key]
    layers = sorted([int(l) for l in final_word_layer_stats_ref.keys()])

    if average_all_dims:
        for i, (data_type, final_word_layer_stats) in enumerate(stats_data.items()):
            all_dims_scores_by_layer = {layer: [] for layer in layers}

            for dim_feature1, dim_feature2 in dim_pairs:
                related_ipa_features = get_related_ipa_features_for_dim_pair((dim_feature1, dim_feature2), feature_to_score)
                ipa_features_1 = related_ipa_features[dim_feature1]
                ipa_features_2 = related_ipa_features[dim_feature2]
                if not ipa_features_1 and not ipa_features_2:
                    continue

                ipa_symbols_1 = get_ipa_symbols_for_ipa_features(ipa_features_1, ipa_dict)
                ipa_symbols_2 = get_ipa_symbols_for_ipa_features(ipa_features_2, ipa_dict)
                if not ipa_symbols_1 and not ipa_symbols_2:
                    continue

                for layer in layers:
                    if layer in final_word_layer_stats:
                        score = calculate_layer_average_score(
                            final_word_layer_stats[layer], 
                            ipa_symbols_1, 
                            ipa_symbols_2, 
                            dim_feature1, 
                            dim_feature2
                        )
                        if score > 0: # Only consider valid scores
                            all_dims_scores_by_layer[layer].append(score)

            avg_scores = np.array([np.mean(all_dims_scores_by_layer[l]) if all_dims_scores_by_layer[l] else 0 for l in layers])
            std_scores = np.array([np.std(all_dims_scores_by_layer[l]) if all_dims_scores_by_layer[l] else 0 for l in layers])
            
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            label = f'Average Attention Fraction ({ "IPA" if data_type == "ipa" else data_type.capitalize()})'

            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)

            ax.plot(layers, avg_scores, color=color, linestyle=linestyle, marker=marker, label=label, zorder=3)
            
            # Calculate and plot trendline
            z = np.polyfit(layers, avg_scores, 1)
            p = np.poly1d(z)
            ax.plot(layers, p(layers), color=color, linestyle='--', alpha=0.7, zorder=2)

            # ax.fill_between(layers, avg_scores - std_scores, avg_scores + std_scores, color=color, alpha=0.2, zorder=1)
        
        ax.legend(loc="best", fontsize=20)

    else:
        # Track used styles and markers
        used_styles = set()
        used_markers = set()
        
        # Plot for each dimension pair
        for pair_idx, (dim_feature1, dim_feature2) in enumerate(dim_pairs):
            # Get related features for this pair
            related_ipa_features = get_related_ipa_features_for_dim_pair((dim_feature1, dim_feature2), feature_to_score)

            ipa_features_1 = related_ipa_features[dim_feature1]
            ipa_features_2 = related_ipa_features[dim_feature2]
            if not ipa_features_1 and not ipa_features_2:
                continue
            
            # Get IPA symbols for these features
            ipa_symbols_1 = get_ipa_symbols_for_ipa_features(ipa_features_1, ipa_dict)
            ipa_symbols_2 = get_ipa_symbols_for_ipa_features(ipa_features_2, ipa_dict)
            if not ipa_symbols_1 and not ipa_symbols_2:
                continue
            
            # Check if any of these IPA symbols exist in our data
            has_data = False
            for layer in final_word_layer_stats.keys():
                for ipa in ipa_symbols_1:
                    if ipa in final_word_layer_stats[layer] and dim_feature1 in final_word_layer_stats[layer][ipa]:
                        has_data = True
                        break
                if has_data:
                    break
            
            for layer in final_word_layer_stats.keys():
                for ipa in ipa_symbols_2:
                    if ipa in final_word_layer_stats[layer] and dim_feature2 in final_word_layer_stats[layer][ipa]:
                        has_data = True
                        break
                if has_data:
                    break

            if not has_data:
                continue

            # Calculate scores for each layer
            scores = []
            
            for layer in layers:
                if layer in final_word_layer_stats:
                    score = calculate_layer_average_score(
                        final_word_layer_stats[layer], 
                        ipa_symbols_1, 
                        ipa_symbols_2, 
                        dim_feature1, 
                        dim_feature2
                    )
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
            feature_names = ', '.join(ipa_features_1 + ipa_features_2)
            label = f"{dim_feature1}-{dim_feature2} ({feature_names})"
            
            # Plot the line
            line = ax.plot(layers, scores, linestyle=line_style, marker=marker, 
                            linewidth=2, markersize=6, label=label)
            
            # Calculate and plot trendline
            if any(s > 0 for s in scores):
                z = np.polyfit(layers, scores, 1)
                p = np.poly1d(z)
                ax.plot(layers, p(layers), color=line[0].get_color(), linestyle='--', alpha=0.7, zorder=2)

            # # Add value annotations
            # for i, (layer, score) in enumerate(zip(layers, scores)):
            #     if score > 0:  # Only annotate if score is not zero
            #         # Format score as .XXX (remove leading 0)
            #         score_str = f"{score:.3f}".lstrip('0')
            #         ax.annotate(score_str, (layer, score), 
            #                    xytext=(0, 10), textcoords='offset points',
            #                    ha='center', va='bottom', fontsize=8)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Customize the plot
    # ax.set_xlabel('Layer', fontsize=12)
    # ax.set_ylabel('Attention Score', fontsize=12)
    title_suffix = "Average" if average_all_dims else "Semantic Dimensions"
    data_types_str = ", ".join(stats_data.keys())
    # ax.set_title(f'Layer-wise Attention Scores for {title_suffix}\n'
    #             f'Data: {data_types_str} | Lang: {lang} | Rule: {compute_rule} | '
    #             f'L{layer_start}-L{layer_end} | Check: {check_model_response} | '
    #             f'Sampling: {sampling_rate}', fontsize=14, pad=20)
    
    # Set y-axis range to 0-1
    ax.set_ylim(0.45, 0.55)
    
    # Add horizontal line at y=0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1, zorder=2)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    file_name = f"layer_attention_score_plot"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(file_path + '.pdf', dpi=300, bbox_inches='tight')
    print(f"Layer attention score plot saved to {file_path}")

    plt.close()

def load_and_plot_from_files(file_paths: Dict[str, str], 
                           lang: str,
                           compute_rule: str,
                           layer_start: int,
                           layer_end: int,
                           check_model_response: bool,
                           sampling_rate: int,
                           average_all_dims: bool = False):
    """
    Load final_word_layer_stats from multiple files and create a single broken line graph.
    """
    # Load data structures
    feature_to_score, ipa_dict, dim_pairs = load_data_structures()
    
    stats_data = {}
    for data_type, file_path in file_paths.items():
        layer_file_path = file_path.replace('.pkl', '_layer_stats.pkl')
        if os.path.exists(layer_file_path):
            with open(layer_file_path, 'rb') as f:
                stats_data[data_type] = pkl.load(f)
            print(f"Loaded layer stats for '{data_type}' from {layer_file_path}")
        else:
            print(f"Layer stats file not found for '{data_type}': {layer_file_path}")

    if not stats_data:
        print("No data loaded, skipping plot generation.")
        return

    # Create the plot
    plot_broken_line_graph(
        stats_data=stats_data,
        feature_to_score=feature_to_score,
        ipa_dict=ipa_dict,
        dim_pairs=dim_pairs,
        lang=lang,
        compute_rule=compute_rule,
        layer_start=layer_start,
        layer_end=layer_end,
        check_model_response=check_model_response,
        sampling_rate=sampling_rate,
        average_all_dims=average_all_dims
    )

# Example usage
if __name__ == "__main__":
    # Example parameters
    file_paths = {
        "ipa": "src/analysis/heatmap/results/np_ipa_Constructed_fraction_check_model_response_True_0_27_sampling_every_1_processed_words_2679.pkl",
        "audio": "src/analysis/heatmap/results/np_audio_Constructed_fraction_check_model_response_True_0_27_sampling_every_1_processed_words_2665.pkl"
    }
    lang = "Constructed"
    compute_rule = "fraction"
    layer_start = 0
    layer_end = 27
    check_model_response = True
    sampling_rate = 1
    
    # Load and plot from multiple files
    load_and_plot_from_files(
        file_paths=file_paths,
        lang=lang,
        compute_rule=compute_rule,
        layer_start=layer_start,
        layer_end=layer_end,
        check_model_response=check_model_response,
        sampling_rate=sampling_rate,
        average_all_dims=True  # Set to True to get the average plot
    )