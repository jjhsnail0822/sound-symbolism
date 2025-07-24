import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap

constructed_dims = [
    "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
    "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
    "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
    "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
    "passive-active", "big-small", "heavy-light", "hard-soft", 
    "solid-nonsolid", "inhibited-free"
]

human_eval_result = {
        "big-small": {"macro_f1_score": 0.8458},
        "tense-relaxed": {"macro_f1_score": 0.8194},
        "ordinary-unique": {"macro_f1_score": 0.7948},
        "sharp-round": {"macro_f1_score": 0.7405},
        "simple-complex": {"macro_f1_score": 0.7352},
        "harsh-mellow": {"macro_f1_score": 0.7306},
        "beautiful-ugly": {"macro_f1_score": 0.6944},
        "abrupt-continuous": {"macro_f1_score": 0.6659},
        "heavy-light": {"macro_f1_score": 0.6638},
        "masculine-feminine": {"macro_f1_score": 0.6538},
        "happy-sad": {"macro_f1_score": 0.6478},
        "inhibited-free": {"macro_f1_score": 0.6458},
        "dangerous-safe": {"macro_f1_score": 0.6247},
        "fast-slow": {"macro_f1_score": 0.5875},
        "exciting-calming": {"macro_f1_score": 0.5717},
        "hard-soft": {"macro_f1_score": 0.5678},
        "solid-nonsolid": {"macro_f1_score": 0.5652},
        "strong-weak": {"macro_f1_score": 0.5637},
        "realistic-fantastical": {"macro_f1_score": 0.4330},
}

def get_color(val):
    green = np.array(to_rgb("#5c940d"))
    red = np.array(to_rgb("#e03131"))
    white = np.array([1, 1, 1])
    if val >= 0.5:
        alpha = min(1.0, (val - 0.5) * 2)
        return tuple((1 - alpha) * white + alpha * green)
    else:
        alpha = min(1.0, (0.5 - val) * 2)
        return tuple((1 - alpha) * white + alpha * red)

def get_model_colors(model_names):
    color_map = {}
    
    # Define base colors for model families
    family_colors = {
        'gpt-4o': plt.cm.get_cmap('Blues'),
        'Qwen': plt.cm.get_cmap('Reds'),
        'gemini': plt.cm.get_cmap('Greens'),
    }
    # Group models by family
    model_families = {}
    other_models = []
    for name in model_names:
        found_family = False
        for family_name in family_colors:
            if name.startswith(family_name):
                if family_name not in model_families:
                    model_families[family_name] = []
                model_families[family_name].append(name)
                found_family = True
                break
        if not found_family:
            other_models.append(name)
            
    # Assign colors to families
    for family_name, members in model_families.items():
        cmap = family_colors[family_name]
        for i, member_name in enumerate(sorted(members)):
            # Use a range from 0.4 to 0.9 to get distinguishable colors
            color_map[member_name] = cmap(0.4 + (i / (len(members) -1 if len(members)>1 else 1)) * 0.5)

    # Assign colors to other models from a different colormap
    other_cmap = plt.cm.get_cmap('plasma')
    for i, model_name in enumerate(other_models):
        color_map[model_name] = other_cmap(i / len(other_models) if len(other_models) > 1 else 0.5)
        
    return color_map
    
def plot_final_rq1_bar_comparison(human_eval, constructed, natural, constructed_model_scores, natural_model_scores, metric_label='macro_f1_score', input_type=''):
    dims = [d for d in constructed_dims if d in human_eval]
    # print(f"[DEBUG] dims (from human_eval): {dims}")
    def get_scores(dims, dct, metric_label):
        scores = []
        for dim in dims:
            score_dict = dct.get(dim, None)
            if score_dict is None:
                scores.append(np.nan)
            else:
                v = score_dict.get(metric_label, np.nan)
                scores.append(v)
        return scores

    group_data = {
        'Natural': get_scores(dims, natural, metric_label),
        'Constructed': get_scores(dims, constructed, metric_label),
        'Human (Audio only)': get_scores(dims, human_eval, metric_label),
    }
    model_scores_map = {
        'Natural': natural_model_scores,
        'Constructed': constructed_model_scores,
    }
    two_stage_font_size = 16
    one_stage_font_size = 20
    # print(f"[DEBUG] group_data: {group_data}")
    group_names = list(group_data.keys())
    n_groups = len(group_names)
    fig, axes = plt.subplots(1, n_groups, figsize=(20, 8), sharey=False)
    # fig, axes = plt.subplots(1, n_groups, figsize=(12, 8), sharey=False)
    if n_groups == 1:
        axes = [axes]

    all_model_names = sorted(list(model_scores_map.get('Natural', {}).keys()))
    model_name_to_color = get_model_colors(all_model_names)

    for ax, group in zip(axes, group_names):
        scores = group_data[group]
        filtered = [(d, s) for d, s in zip(dims, scores) if not np.isnan(s)]
        if not filtered:
            ax.set_title(group, fontsize=two_stage_font_size, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        dims_filtered, scores_filtered = zip(*filtered)
        sort_idx = np.argsort([-s for s in scores_filtered])
        dims_sorted = [dims_filtered[i] for i in sort_idx]
        scores_sorted = [scores_filtered[i] for i in sort_idx]
        colors = [get_color(v) for v in scores_sorted]
        n = len(dims_sorted)
        bars = ax.barh(range(n), scores_sorted, color=colors, edgecolor='black')

        # Plot model average points, connecting points for the same dimension
        if group in model_scores_map:
            model_scores = model_scores_map[group]
            dim_to_y = {dim: i for i, dim in enumerate(dims_sorted)}

            # Iterate through each dimension to draw a line connecting model scores
            for i, dim in enumerate(dims_sorted):
                y_coord = i
                
                # Collect points for this dimension from all models
                points_for_dim = []
                for model_name in all_model_names:
                    model_data = model_scores.get(model_name, {})
                    score_dict = model_data.get(dim, {})
                    score = score_dict.get(metric_label, np.nan)
                    if not np.isnan(score):
                        points_for_dim.append({'x': score, 'model': model_name})
                
                if not points_for_dim:
                    continue

                # Sort points by x-value to draw a connecting line
                points_for_dim.sort(key=lambda p: p['x'])
                
                x_vals = [p['x'] for p in points_for_dim]
                
                # Plot the connecting line
                if len(x_vals) > 1:
                    ax.plot(x_vals, [y_coord] * len(x_vals), color='gray', alpha=0.7, zorder=2, linewidth=1)

            # Plot the individual model points on top, ensuring legend is created correctly
            # We do this in a separate loop to ensure lines are drawn under all points
            # and legend handles are created only once per model.
            plotted_labels = set()
            for model_name in all_model_names:
                points_to_plot = []
                for dim, y_coord in dim_to_y.items():
                    model_data = model_scores.get(model_name, {})
                    score_dict = model_data.get(dim, {})
                    score = score_dict.get(metric_label, np.nan)
                    if not np.isnan(score):
                        points_to_plot.append((score, y_coord))
                
                if points_to_plot:
                    x_vals, y_vals = zip(*points_to_plot)
                    label = model_name if model_name not in plotted_labels else None
                    ax.scatter(x_vals, y_vals, color=model_name_to_color[model_name], zorder=3, label=label, s=50, alpha=0.8, edgecolors='black')
                    if label:
                        plotted_labels.add(model_name)


        ax.set_xlim(0, 1)
        ax.set_xlabel("Macro-F1 Score", fontsize=one_stage_font_size)
        ax.set_yticks(range(n))
        ax.set_yticklabels(dims_sorted, fontsize=two_stage_font_size)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        ax.set_title(group, fontsize=two_stage_font_size, pad=10)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=len(handles), fontsize=one_stage_font_size-4)

    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space for legend
    save_path = f"./results/plots/rq1/final_rq1_bar_comparison_{input_type}_{metric_label}.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    save_path = f"./results/plots/rq1/final_rq1_bar_comparison_{input_type}_{metric_label}.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()

def aggregate_score_and_count(language_list, data, metric, final_semdim_score, final_semdim_count, final_input_type_list):
    for language in language_list:
        models_data = data[language]
        
        # Group gpt-4o models
        grouped_models_data = {}
        for model_name, model_data in models_data.items():
            key = 'gpt-4o' if model_name.startswith('gpt-4o') else model_name
            if key not in grouped_models_data:
                grouped_models_data[key] = []
            grouped_models_data[key].append(model_data)

        for model, model_data_list in grouped_models_data.items():
            for model_data in model_data_list:
                word_types = model_data.keys()
                for word_type in word_types:
                    if word_type not in ["constructed", "natural"]:
                        continue
                    input_types = model_data[word_type].keys()
                    for input_type in input_types:
                        if input_type not in ["ipa", "audio", "ipa_and_audio", "original", "original_and_audio"]:
                            continue
                        if input_type not in final_input_type_list:
                            final_input_type_list.append(input_type)
                        semantic_dimensions = model_data[word_type][input_type]["dimensions"].keys()
                        for semdim in semantic_dimensions:
                            if semdim not in constructed_dims:
                                continue
                            if semdim not in final_semdim_score[input_type][word_type]:
                                final_semdim_score[input_type][word_type][semdim] = []
                            final_semdim_score[input_type][word_type][semdim].append(model_data[word_type][input_type]["dimensions"][semdim][metric])
                            if semdim not in final_semdim_count[input_type][word_type]:
                                final_semdim_count[input_type][word_type][semdim] = []
                            final_semdim_count[input_type][word_type][semdim].append(model_data[word_type][input_type]["dimensions"][semdim]["count"])
    return final_semdim_score, final_semdim_count, final_input_type_list

def compute_mean_average_score(final_semdim_score, final_semdim_count, final_semdim_score_weighted, eval_metric:str="macro_f1_score"):
    semdim_wordtype_scores = {
        "constructed": {},
        "natural": {},
    }
    for input_type in final_semdim_score.keys():
        for word_type in final_semdim_score[input_type].keys():
            if word_type not in ["constructed", "natural"]:
                continue
            for semdim in final_semdim_score[input_type][word_type].keys():
                metrics = final_semdim_score[input_type][word_type][semdim]
                if semdim not in semdim_wordtype_scores[word_type]:
                    semdim_wordtype_scores[word_type][semdim] = []
                semdim_wordtype_scores[word_type][semdim].extend(metrics)
    for word_type in ["constructed", "natural"]:
        for semdim, values in semdim_wordtype_scores[word_type].items():
            if len(values) > 0:
                mean_val = float(np.mean(values))
            else:
                mean_val = float('nan')
            if semdim not in final_semdim_score_weighted["whole"][word_type]:
                final_semdim_score_weighted["whole"][word_type][semdim] = {}
            final_semdim_score_weighted["whole"][word_type][semdim][eval_metric] = mean_val
            print(f"[DEBUG] word_type={word_type}, semdim={semdim}, mean={mean_val}")
    return final_semdim_score_weighted

def aggregate_per_model(json_path, metric='macro_f1_score'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_scores = {}
    models_data = data.get('all', {})

    # Group gpt-4o and qwen models
    grouped_models_data = {}
    for model_name, model_data in models_data.items():
        key = model_name
        if model_name.startswith('gpt-4o'):
            key = 'gpt-4o'
        elif 'Qwen' in model_name:
            # Keep specific qwen models separate but handle them as a family later
            pass

        if key not in grouped_models_data:
            grouped_models_data[key] = []
        grouped_models_data[key].append((model_name, model_data))

    final_model_scores = {}
    for group_key, model_list in grouped_models_data.items():
        
        # If it's a grouped model like gpt-4o, calculate a single average
        if group_key == 'gpt-4o':
            semdim_scores_for_model_constructed = {}
            semdim_scores_for_model_natural = {}
            
            for _, model_data in model_list:
                for word_type, word_type_data in model_data.items():
                    if word_type not in ["constructed", "natural"]: continue
                    
                    semdim_target = semdim_scores_for_model_constructed if word_type == "constructed" else semdim_scores_for_model_natural
                    
                    for input_type, input_type_data in word_type_data.items():
                        if input_type not in ["ipa", "audio", "ipa_and_audio", "original", "original_and_audio"]: continue
                        
                        for semdim, semdim_data in input_type_data.get("dimensions", {}).items():
                            if semdim not in constructed_dims: continue
                            if semdim not in semdim_target: semdim_target[semdim] = []
                            
                            score = semdim_data.get(metric)
                            if score is not None: semdim_target[semdim].append(score)

            final_model_scores[group_key] = {"constructed": {}, "natural": {}}
            for semdim, scores in semdim_scores_for_model_constructed.items():
                if scores: final_model_scores[group_key]["constructed"][semdim] = {metric: np.mean(scores)}
            for semdim, scores in semdim_scores_for_model_natural.items():
                if scores: final_model_scores[group_key]["natural"][semdim] = {metric: np.mean(scores)}

        # For other models (including individual qwen models), process them one by one
        else:
            for original_model_name, model_data in model_list:
                final_model_scores[original_model_name] = {"constructed": {}, "natural": {}}
                semdim_scores_for_model_constructed = {}
                semdim_scores_for_model_natural = {}

                for word_type, word_type_data in model_data.items():
                    if word_type not in ["constructed", "natural"]: continue
                    semdim_target = semdim_scores_for_model_constructed if word_type == "constructed" else semdim_scores_for_model_natural
                    for input_type, input_type_data in word_type_data.items():
                        if input_type not in ["ipa", "audio", "ipa_and_audio", "original", "original_and_audio"]: continue
                        for semdim, semdim_data in input_type_data.get("dimensions", {}).items():
                            if semdim not in constructed_dims: continue
                            if semdim not in semdim_target: semdim_target[semdim] = []
                            score = semdim_data.get(metric)
                            if score is not None: semdim_target[semdim].append(score)
                
                for semdim, scores in semdim_scores_for_model_constructed.items():
                    if scores: final_model_scores[original_model_name]["constructed"][semdim] = {metric: np.mean(scores)}
                for semdim, scores in semdim_scores_for_model_natural.items():
                    if scores: final_model_scores[original_model_name]["natural"][semdim] = {metric: np.mean(scores)}

    return final_model_scores

def aggregate_macro_f1_by_inputtype(json_path, constructed_dims=constructed_dims, metric='macro_f1_score'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Always use {lang: {model: ...}} structure
    language_list = ['all']
    final_input_type_list = []
    final_semdim_score = {
        "ipa": {"constructed": {}, "natural": {}},
        "audio": {"constructed": {}, "natural": {}},
        "ipa_and_audio": {"constructed": {}, "natural": {}},
        "original": {"constructed": {}, "natural": {}},
        "original_and_audio": {"constructed": {}, "natural": {}},
    }
    final_semdim_count = {
        "ipa": {"constructed": {}, "natural": {}},
        "audio": {"constructed": {}, "natural": {}},
        "ipa_and_audio": {"constructed": {}, "natural": {}},
        "original": {"constructed": {}, "natural": {}},
        "original_and_audio": {"constructed": {}, "natural": {}},
    }
    final_semdim_score_weighted = {
        "whole": {"constructed": {}, "natural": {}},
    }

    final_semdim_score, final_semdim_count, final_input_type_list = aggregate_score_and_count(language_list, data, metric, final_semdim_score, final_semdim_count, final_input_type_list)
    
    final_semdim_score_weighted = compute_mean_average_score(final_semdim_score, final_semdim_count, final_semdim_score_weighted)

    return final_semdim_score_weighted

if __name__ == "__main__":
    json_path = './results/statistics/semdim_stat.json'
    metric = 'macro_f1_score'
    results = aggregate_macro_f1_by_inputtype(json_path, constructed_dims=constructed_dims, metric=metric)
    
    model_results = aggregate_per_model(json_path, metric=metric)
    natural_model_scores = {model: data['natural'] for model, data in model_results.items() if data['natural']}
    constructed_model_scores = {model: data['constructed'] for model, data in model_results.items() if data['constructed']}

    constructed_scores = results["whole"]["constructed"]
    natural_scores = results["whole"]["natural"]
    
    print(f"\nPlotting for input_type: whole")
    plot_final_rq1_bar_comparison(
        human_eval_result, 
        constructed_scores, 
        natural_scores, 
        constructed_model_scores,
        natural_model_scores,
        metric_label='macro_f1_score', 
        input_type='whole'
    )

