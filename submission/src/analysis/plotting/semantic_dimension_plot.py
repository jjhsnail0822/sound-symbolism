import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from collections import defaultdict
from scipy.stats import pearsonr
from matplotlib.patches import Patch

natural_languages = ["en", "fr", "ja", "ko"]
constructed_languages = ["art"]

constructed_dims = [
    "beautiful-ugly", "delicate-rugged", "tense-relaxed", "simple-complex", 
    "happy-sad", "sharp-round", "fast-slow", "masculine-feminine", 
    "strong-weak", "exciting-calming", "harsh-mellow", "ordinary-unique", 
    "dangerous-safe", "realistic-fantastical", "abrupt-continuous", 
    "passive-active", "big-small", "heavy-light", "hard-soft", 
    "solid-nonsolid", "inhibited-free"
]

human_eval_json_path = 'results/experiments/semantic_dimension/human_eval/ranked_dimension_f1_scores.json'
with open(human_eval_json_path, 'r', encoding='utf-8') as f:
    human_eval_data = json.load(f)

human_eval_result = {
    item['dimension']: {'macro_f1_score': item['average_f1_score']}
    for item in human_eval_data['ranked_dimensions']
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
        'gpt-4o': plt.colormaps['Blues'],
        'Qwen': plt.colormaps['Reds'],
        'gemini': plt.colormaps['Greens'],
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
    other_cmap = plt.colormaps['plasma']
    for i, model_name in enumerate(other_models):
        color_map[model_name] = other_cmap(i / len(other_models) if len(other_models) > 1 else 0.5)
        
    return color_map

def aggregate_score_per_model(json_path, metric='macro_f1_score', target_input_types=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores_aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model_name, model_data in data.items():
        for word_group, group_data in model_data.items():
            if word_group not in ['natural', 'constructed']:
                continue
            for input_type, type_data in group_data.items():
                if target_input_types and input_type not in target_input_types:
                    continue
                for dimension, dim_data in type_data.items():
                    if dimension not in constructed_dims:
                        continue
                    score = dim_data.get('all', {}).get(metric)
                    if score is not None:
                        scores_aggregated[model_name][word_group][dimension].append(score)

    # Calculate mean scores
    final_model_scores = defaultdict(lambda: defaultdict(dict))
    for model_name, word_groups in scores_aggregated.items():
        for word_group, dimensions in word_groups.items():
            for dimension, scores in dimensions.items():
                if scores:
                    final_model_scores[model_name][word_group][dimension] = np.mean(scores)

    return final_model_scores

def aggregate_score_by_input_type(json_path, metric='macro_f1_score'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores_aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model_name, model_data in data.items():
        for word_group, group_data in model_data.items():
            if word_group not in ['natural', 'constructed']:
                continue
            for input_type, type_data in group_data.items():
                for dimension, dim_data in type_data.items():
                    if dimension not in constructed_dims:
                        continue
                    score = dim_data.get('all', {}).get(metric)
                    if score is not None:
                        # Aggregate scores by input_type, keeping word_group separate
                        scores_aggregated[input_type][word_group][model_name].append(score)

    # Calculate mean scores
    final_input_type_scores = defaultdict(lambda: defaultdict(dict))
    for input_type, word_groups in scores_aggregated.items():
        for word_group, models in word_groups.items():
            for model_name, scores in models.items():
                if scores:
                    # Average score across all dimensions for a given model
                    final_input_type_scores[input_type][word_group][model_name] = np.mean(scores)

    return final_input_type_scores

def aggregate_scores_for_correlation(json_path, metric='macro_f1_score', target_input_types=None):
    """
    Aggregates scores for correlation analysis, creating combined groups for 
    each word group and input type (e.g., 'Natural-Original').
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Structure: scores[model_name][group_key][dimension] = score
    # group_key will be like 'Natural-Original', 'Constructed-IPA'
    final_scores = defaultdict(lambda: defaultdict(dict))

    for model_name, model_data in data.items():
        # Intermediate storage: aggregated_scores[group_key][dimension] = list_of_scores
        aggregated_scores = defaultdict(lambda: defaultdict(list))

        for word_group, group_data in model_data.items():
            if word_group not in ['natural', 'constructed']:
                continue
            for input_type, type_data in group_data.items():
                if target_input_types and input_type not in target_input_types:
                    continue
                
                # Create a combined key with desired capitalization, e.g., "Natural-Original"
                word_group_label_map = {
                    'natural': 'Nat.',
                    'constructed': 'Con.'
                }
                input_type_label_map = {
                    'original': 'Original',
                    'ipa': 'IPA',
                    'audio': 'Audio',
                    'original_and_audio': 'Original+Audio',
                    'ipa_and_audio': 'IPA+Audio'
                }
                word_group_label = word_group_label_map.get(word_group, word_group.capitalize())
                input_type_label = input_type_label_map.get(input_type, input_type)
                group_key = f"{word_group_label}-{input_type_label}"

                for dimension, dim_data in type_data.items():
                    if dimension not in constructed_dims:
                        continue
                    score = dim_data.get('all', {}).get(metric)
                    if score is not None:
                        aggregated_scores[group_key][dimension].append(score)

        # Calculate mean for the new aggregated groups
        for group_key, dimensions in aggregated_scores.items():
            for dim, scores in dimensions.items():
                if scores:
                    final_scores[model_name][group_key][dim] = np.mean(scores)
                
    return final_scores

def plot_bar_chart(llm_data, human_eval, save_path_prefix, metric='macro_f1_score'):
    # Separate scores by word group and prepare for plotting
    natural_scores = defaultdict(list)
    constructed_scores = defaultdict(list)
    natural_model_scores = defaultdict(lambda: defaultdict(dict))
    constructed_model_scores = defaultdict(lambda: defaultdict(dict))

    all_model_names = sorted(llm_data.keys())

    for model_name, word_group_data in llm_data.items():
        if 'natural' in word_group_data:
            for dim, score in word_group_data['natural'].items():
                natural_scores[dim].append(score)
                natural_model_scores[model_name][dim] = {metric: score}
        if 'constructed' in word_group_data:
            for dim, score in word_group_data['constructed'].items():
                constructed_scores[dim].append(score)
                constructed_model_scores[model_name][dim] = {metric: score}

    # Calculate the average score across all models for the main bars
    avg_natural_scores = {dim: {metric: np.mean(scores)} for dim, scores in natural_scores.items()}
    avg_constructed_scores = {dim: {metric: np.mean(scores)} for dim, scores in constructed_scores.items()}

    dims = [d for d in constructed_dims if d in human_eval]
    
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
        'Natural': get_scores(dims, avg_natural_scores, metric),
        'Constructed': get_scores(dims, avg_constructed_scores, metric),
        'Human (Audio only)': get_scores(dims, human_eval, metric),
    }
    model_scores_map = {
        'Natural': natural_model_scores,
        'Constructed': constructed_model_scores,
    }
    
    two_stage_font_size = 16
    one_stage_font_size = 20
    group_names = list(group_data.keys())
    n_groups = len(group_names)
    fig, axes = plt.subplots(1, n_groups, figsize=(16, 6), sharey=False)
    if n_groups == 1:
        axes = [axes]

    model_name_to_color = get_model_colors(all_model_names)
    model_markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
    model_marker_map = {name: model_markers[i % len(model_markers)] for i, name in enumerate(all_model_names)}

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
        ax.barh(range(n), scores_sorted, color=colors, edgecolor='black')

        if group in model_scores_map:
            model_scores = model_scores_map[group]
            dim_to_y = {dim: i for i, dim in enumerate(dims_sorted)}

            # Iterate through each dimension to draw a line connecting model scores
            for i, dim in enumerate(dims_sorted):
                y_coord = i
                points_for_dim = []
                for model_name in all_model_names:
                    score = model_scores.get(model_name, {}).get(dim, {}).get(metric, np.nan)
                    if not np.isnan(score):
                        points_for_dim.append(score)
                
                if len(points_for_dim) > 1:
                    ax.plot(sorted(points_for_dim), [y_coord] * len(points_for_dim), color='gray', alpha=0.7, zorder=2, linewidth=1)

            # Plot the individual model points on top
            plotted_labels = set()
            for model_name in all_model_names:
                points_to_plot = []
                for dim, y_coord in dim_to_y.items():
                    score = model_scores.get(model_name, {}).get(dim, {}).get(metric, np.nan)
                    if not np.isnan(score):
                        points_to_plot.append((score, y_coord))
                
                if points_to_plot:
                    x_vals, y_vals = zip(*points_to_plot)
                    label = model_name if model_name not in plotted_labels else None
                    ax.scatter(x_vals, y_vals, color=model_name_to_color[model_name], marker=model_marker_map[model_name], zorder=3, label=label, s=60, alpha=0.8, edgecolors='black')
                    if label:
                        plotted_labels.add(model_name)

        ax.set_xlim(0, 1)
        ax.tick_params(axis='x', labelsize=two_stage_font_size)
        # ax.set_xlabel("Macro-F1 Score", fontsize=one_stage_font_size)
        ax.set_yticks(range(n))
        ax.set_yticklabels([d.replace('-', ' - ') for d in dims_sorted], fontsize=two_stage_font_size)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        ax.set_title(group, fontsize=one_stage_font_size, pad=10)

    handles, labels = [], []
    model_display_name_map = {
        'gpt-4o-2024-05-13': 'GPT-4o',
        'Qwen2-72B-Instruct': 'Qwen2-72B',
        'Qwen2-7B-Instruct': 'Qwen2-7B',
        'Qwen2-1.5B-Instruct': 'Qwen2-1.5B',
        'Qwen2-0.5B-Instruct': 'Qwen2-0.5B',
        'gemini-1.5-pro-latest': 'Gemini 1.5 Pro',
        'gemini-1.5-flash-latest': 'Gemini 1.5 Flash',
        'gemini-1.0-pro': 'Gemini 1.0 Pro'
    }
    for model_name in all_model_names:
        if model_name in model_marker_map:
            display_name = model_display_name_map.get(model_name, model_name)
            handles.append(Line2D([0], [0], marker=model_marker_map[model_name], color='w', 
                                  markerfacecolor=model_name_to_color.get(model_name, 'grey'), 
                                  markersize=10, label=display_name))
            labels.append(display_name)
    
    if handles:
        # Sorting based on original names to maintain order, but using display names for labels
        sorted_handles_labels = sorted(zip(handles, all_model_names), key=lambda x: all_model_names.index(x[1]))
        handles, _ = zip(*sorted_handles_labels)
        # Get display names in the new sorted order
        sorted_labels = [model_display_name_map.get(name, name) for name in sorted(all_model_names)]

        fig.legend(handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=len(handles), fontsize=one_stage_font_size-4)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = f"{save_path_prefix}_bar_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {save_path}")
    save_path_pdf = f"{save_path_prefix}_bar_comparison.pdf"
    plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {save_path_pdf}")
    plt.close()

    return avg_natural_scores, avg_constructed_scores


def plot_correlation_with_human(correlation_data, human_eval, save_path_prefix, metric='macro_f1_score', average_across_word_groups=False):
    """
    Calculates and plots Pearson correlation between model scores and human scores,
    grouped by word group and input type.
    If average_across_word_groups is True, it first averages the scores for each dimension
    across 'natural' and 'constructed' groups for each input type, then calculates correlation.
    """
    all_model_names = sorted(correlation_data.keys())
    dims = [d for d in constructed_dims if d in human_eval]
    human_scores_list = [human_eval.get(dim, {}).get(metric, np.nan) for dim in dims]

    # Define a fixed order and labels for the groups
    group_order = [
        'Nat.-Original', 'Nat.-IPA', 'Nat.-Audio',
        'Con.-Original', 'Con.-IPA', 'Con.-Audio'
    ]
    
    # Filter groups to only those present in the data
    all_groups_present = set()
    for model_data in correlation_data.values():
        all_groups_present.update(model_data.keys())
    
    plot_groups = [g for g in group_order if g in all_groups_present]
    
    if average_across_word_groups:
        # New logic: Average scores across word groups for each input type first
        input_type_order = ['Original', 'IPA', 'Audio']
        present_input_types = set([g.split('-')[1] for g in plot_groups])
        input_types = [it for it in input_type_order if it in present_input_types]
        plot_groups = input_types # The new groups are the input types themselves
        
        averaged_data = defaultdict(lambda: defaultdict(dict))
        for model_name in all_model_names:
            for input_type in input_types:
                nat_group = f'Nat.-{input_type}'
                con_group = f'Con.-{input_type}'
                
                nat_scores_dim = correlation_data[model_name].get(nat_group, {})
                con_scores_dim = correlation_data[model_name].get(con_group, {})
                
                present_dims = set(nat_scores_dim.keys()) | set(con_scores_dim.keys())
                
                for dim in dims:
                    if dim in present_dims:
                        nat_score = nat_scores_dim.get(dim, np.nan)
                        con_score = con_scores_dim.get(dim, np.nan)
                        # Use nanmean to correctly average if one score is NaN
                        avg_score = np.nanmean([nat_score, con_score])
                        if not np.isnan(avg_score):
                            averaged_data[model_name][input_type][dim] = avg_score
        
        # The data to be used for correlation is the newly averaged data
        data_for_corr = averaged_data
    else:
        # Original logic
        data_for_corr = correlation_data

    correlations = defaultdict(dict)

    for model_name in all_model_names:
        for group in plot_groups:
            if group in data_for_corr[model_name]:
                model_scores = [data_for_corr[model_name][group].get(dim, np.nan) for dim in dims]
                valid_pairs = [(h, m) for h, m in zip(human_scores_list, model_scores) if not np.isnan(h) and not np.isnan(m)]
                
                if len(valid_pairs) > 1:
                    h_scores, m_scores = zip(*valid_pairs)
                    corr, _ = pearsonr(h_scores, m_scores)
                    correlations[model_name][group] = corr
                else:
                    correlations[model_name][group] = np.nan

    # --- Plotting ---
    n_groups = len(plot_groups)
    n_models = len(all_model_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    index = np.arange(n_models)
    bar_width = 0.8 / n_groups
    
    # Define a fixed order for models
    model_order = [
        'Qwen2.5-Omni-3B',
        'Qwen2.5-Omni-7B',
        'gemini-2.5-flash',
        'gpt-4o',
    ]
    order_map = {name: i for i, name in enumerate(model_order)}
    sorted_models = sorted(all_model_names, key=lambda m: order_map.get(m, len(model_order)))

    model_display_name_map = {
        'Qwen2.5-Omni-3B': 'Qwen-3B',
        'Qwen2.5-Omni-7B': 'Qwen-7B',
        'gemini-2.5-flash': 'Gemini',
        'gpt-4o': 'GPT',
    }

    # --- Print correlations to terminal ---
    print("\n--- Pearson Correlation with Human Scores ---")
    for model_name in sorted_models:
        display_name = model_display_name_map.get(model_name, model_name)
        print(f"Model: {display_name}")
        for group in plot_groups:
            corr_value = correlations[model_name].get(group)
            if corr_value is not None and not np.isnan(corr_value):
                print(f"  - {group}: {corr_value:.3f}")
            else:
                print(f"  - {group}: N/A")
    print("-----------------------------------------\n")

    # Define colors and hatches for bars
    color_map = {
        'Original': plt.get_cmap('viridis')(0.2),
        'IPA': plt.get_cmap('viridis')(0.5),
        'Audio': plt.get_cmap('viridis')(0.8),
        'Nat.-Original': plt.get_cmap('Greens')(0.4), 'Nat.-IPA': plt.get_cmap('Greens')(0.65), 'Nat.-Audio': plt.get_cmap('Greens')(0.9),
        'Con.-Original': plt.get_cmap('Oranges')(0.4), 'Con.-IPA': plt.get_cmap('Oranges')(0.65), 'Con.-Audio': plt.get_cmap('Oranges')(0.9)
    }
    hatch_map = {
        'Original': '', 'IPA': '///', 'Audio': '+++',
        'Nat.-Original': '', 'Nat.-IPA': '///', 'Nat.-Audio': '+++',
        'Con.-Original': '...', 'Con.-IPA': '\\\\\\', 'Con.-Audio': 'xxx',
    }

    for i, group in enumerate(plot_groups):
        corrs = [correlations[m].get(group, np.nan) for m in sorted_models]
        pos = index - (bar_width * (n_groups - 1) / 2) + (i * bar_width)
        
        bar_color = color_map.get(group)
        bar_hatch = hatch_map.get(group)

        ax.bar(pos, corrs, bar_width, label=group, 
               color=bar_color, 
               hatch=bar_hatch, 
               edgecolor='black')

    # ax.set_ylabel('Pearson Correlation with Human Scores', fontsize=16)
    # ax.set_title('Model vs. Human Score Correlation by Group', fontsize=18, pad=20)
    ax.set_xticks(index)
    xtick_labels = [model_display_name_map.get(m, m) for m in sorted_models]
    ax.set_xticklabels(xtick_labels, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(-0.4, 1.0)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    legend_elements = []
    # Desired order for the legend
    if average_across_word_groups:
        legend_order = ['Original', 'IPA', 'Audio']
    else:
        legend_order = [
            'Nat.-Original',
            'Nat.-IPA',
            'Nat.-Audio',
            'Con.-Original',
            'Con.-IPA',
            'Con.-Audio',
        ]
    # Filter order to only include groups that are actually plotted
    filtered_legend_order = [group for group in legend_order if group in plot_groups]

    for group in filtered_legend_order:
        legend_elements.append(Patch(facecolor=color_map.get(group),
                                     edgecolor='black',
                                     hatch=hatch_map.get(group, ''),
                                     label=group))

    ax.legend(handles=legend_elements, fontsize=20, loc='upper left', ncol=3 if average_across_word_groups else 2)

    plt.tight_layout()

    plot_suffix = "human_correlation_avg_plot" if average_across_word_groups else "human_correlation_plot"
    save_path = f"{save_path_prefix}_{plot_suffix}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved correlation plot to {save_path}")
    save_path_pdf = f"{save_path_prefix}_{plot_suffix}.pdf"
    plt.savefig(save_path_pdf, dpi=300)
    print(f"Saved correlation plot to {save_path_pdf}")
    plt.close()


def plot_input_type(llm_data, save_path_prefix, metric='macro_f1_score'):
    """
    Plots a grouped scatter plot of model performance by input type and word group.
    X-axis: Input types.
    Y-axis: Average score across all dimensions.
    Color: Word group ('natural' or 'constructed').
    Marker: Model.
    """
    two_stage_font_size = 20
    one_stage_font_size = 24
    input_type_labels = {
        'original': 'Original',
        'original_and_audio': 'Original\n+ Audio',
        'ipa': 'IPA',
        'ipa_and_audio': 'IPA\n+ Audio',
        'audio': 'Audio'
    }
    input_types_ordered = list(input_type_labels.keys())
    word_types = ['natural', 'constructed']

    # Extract all model names from the data
    all_model_names = set()
    for input_type_data in llm_data.values():
        for word_group_data in input_type_data.values():
            all_model_names.update(word_group_data.keys())
    all_model_names = sorted(list(all_model_names))

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Define visual mappings ---
    word_type_colors = {'natural': '#A9E34B', 'constructed': '#FFA94D'}
    model_markers = ['o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h']
    model_marker_map = {name: model_markers[i % len(model_markers)] for i, name in enumerate(all_model_names)}

    # --- Calculate X-axis positions for grouped scatter plot ---
    group_gap = 2.0
    inner_gap = 0.4
    x_map = {}
    x_labels = []
    x_tick_pos = []
    for i, input_type in enumerate(input_types_ordered):
        base = i * group_gap
        for j, word_type in enumerate(word_types):
            x = base + (j - (len(word_types) - 1) / 2) * inner_gap
            x_map[(input_type, word_type)] = x
        x_labels.append(input_type_labels[input_type])
        x_tick_pos.append(base)

    # --- Collect data for plotting ---
    plot_data = []
    for input_type, word_groups in llm_data.items():
        for word_group, models in word_groups.items():
            for model_name, score in models.items():
                if (input_type, word_group) in x_map:
                    plot_data.append({
                        'x': x_map[(input_type, word_group)],
                        'y': score,
                        'color': word_type_colors[word_group],
                        'marker': model_marker_map[model_name],
                        'model': model_name
                    })

    # --- Highlight background for major input types ---
    highlight_input_types = ['original', 'ipa', 'audio']
    for i, input_type in enumerate(input_types_ordered):
        if input_type in highlight_input_types:
            base = i * group_gap
            rect = plt.Rectangle((base - group_gap/2, 0), group_gap, 1,
                               facecolor='lightgray', alpha=0.2, edgecolor='none', zorder=0)
            ax.add_patch(rect)

    # --- Draw vertical connecting lines ---
    points_by_x = defaultdict(list)
    for p in plot_data:
        points_by_x[p['x']].append(p['y'])

    for x_coord, y_coords in points_by_x.items():
        if len(y_coords) > 1:
            y_min, y_max = min(y_coords), max(y_coords)
            ax.plot([x_coord, x_coord], [y_min, y_max], color='gray', alpha=0.7, zorder=1, linewidth=1.5)

    # --- Plot points ---
    for marker_char in set(model_marker_map.values()):
        points = [p for p in plot_data if p['marker'] == marker_char]
        if points:
            ax.scatter(
                [p['x'] for p in points], [p['y'] for p in points],
                c=[p['color'] for p in points], marker=marker_char,
                alpha=0.8, s=120, edgecolors='black', linewidth=1, zorder=10
            )

    # --- Legend ---
    model_display_name_map = {
        'gpt-4o-2024-05-13': 'GPT-4o',
        'Qwen2-72B-Instruct': 'Qwen2-72B',
        'Qwen2-7B-Instruct': 'Qwen2-7B',
        'Qwen2-1.5B-Instruct': 'Qwen2-1.5B',
        'Qwen2-0.5B-Instruct': 'Qwen2-0.5B',
        'gemini-1.5-pro-latest': 'Gemini 1.5 Pro',
        'gemini-1.5-flash-latest': 'Gemini 1.5 Flash',
        'gemini-1.0-pro': 'Gemini 1.0 Pro'
    }
    legend_elements = []
    # for word_type, color in word_type_colors.items():
    #     legend_elements.append(Line2D([0], [0], marker='o', color='w', label=word_type.title(),
    #                                   markerfacecolor=color, markersize=12))
    for model_name, marker in model_marker_map.items():
        display_name = model_display_name_map.get(model_name, model_name)
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=display_name,
                                      markerfacecolor='grey', markersize=10))
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=two_stage_font_size - 4)

    # --- Axes and Grid ---
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_labels, fontsize=two_stage_font_size)
    # ax.set_xlabel("Input Type", fontsize=one_stage_font_size, labelpad=15)
    # ax.set_ylabel(f"Average {metric.replace('_', ' ').replace("macro f1", 'Macro-F1').title()}", fontsize=one_stage_font_size)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=two_stage_font_size)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.7, zorder=1)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    # --- Save Plot ---
    save_path_png = f"{save_path_prefix}_input_type_scatter.png"
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    print(f"Saved scatter plot to {save_path_png}")
    save_path_pdf = f"{save_path_prefix}_input_type_scatter.pdf"
    plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)
    print(f"Saved scatter plot to {save_path_pdf}")
    plt.close()


json_path = 'results/statistics/semdim_stat.json'
save_path_prefix = 'results/plots/semantic_dimension'
metric = 'macro_f1_score'

target_types = [
    'original',
    # 'original_and_audio',
    'ipa',
    # 'ipa_and_audio',
    'audio',
    ]
results_per_model = aggregate_score_per_model(json_path, metric, target_input_types=target_types)
results_by_input_type = aggregate_score_by_input_type(json_path, metric)
correlation_data = aggregate_scores_for_correlation(json_path, metric, target_input_types=target_types)

# debug scores
for model_name, word_groups in results_per_model.items():
    print(f"Model: {model_name}")
    for word_group, dimensions in word_groups.items():
        print(f"  Word Group: {word_group}")
        for dim, score in dimensions.items():
            print(f"    {dim}: {score:.3f}")
average_natural_scores, average_constructed_scores = plot_bar_chart(results_per_model, human_eval_result, save_path_prefix)
plot_correlation_with_human(correlation_data, human_eval_result, save_path_prefix)
plot_correlation_with_human(correlation_data, human_eval_result, save_path_prefix, average_across_word_groups=True)
# plot_input_type(results_by_input_type, save_path_prefix)

# debug average scores
print("\nAverage Natural Scores:")
for dim, score in average_natural_scores.items():
    print(f"  {dim}: {score['macro_f1_score']:.3f}")
print("\nAverage Constructed Scores:")
for dim, score in average_constructed_scores.items():
    print(f"  {dim}: {score['macro_f1_score']:.3f}")
