import json
import pandas as pd
import re
from collections import defaultdict

def generate_latex_tables(data_path):
    """
    Loads statistics from a JSON file and generates a separate LaTeX table for each model.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Use data under the 'all' key which contains all languages
    data = raw_data.get('all', {})

    # Merge gpt-4o and gpt-4o-audio-preview into a single gpt-4o model
    if "gpt-4o" in data and "gpt-4o-audio-preview" in data:
        gpt4o_audio_data = data.pop("gpt-4o-audio-preview")
        for freq, freq_data in gpt4o_audio_data.items():
            if freq in data["gpt-4o"]:
                # Deep merge dimensions data
                for input_type, input_data in freq_data.items():
                    if input_type in data["gpt-4o"][freq]:
                         data["gpt-4o"][freq][input_type].update(input_data)
                    else:
                         data["gpt-4o"][freq][input_type] = input_data
            else:
                data["gpt-4o"][freq] = freq_data

    # Define the order of dimensions and columns as in the LaTeX template
    dimensions_order = [
        "good-bad", "beautiful-ugly", "pleasant-unpleasant", "strong-weak",
        "big-small", "rugged-delicate", "active-passive", "fast-slow",
        "sharp-round", "realistic-fantastical", "structured-disorganized",
        "ordinary-unique", "interesting-uninteresting", "simple-complex",
        "abrupt-continuous", "exciting-calming", "hard-soft", "happy-sad",
        "harsh-mellow", "heavy-light", "inhibited-free", "masculine-feminine",
        "solid-nonsolid", "tense-relaxed", "dangerous-safe"
    ]
    
    # Handle reversed dimensions like 'delicate-rugged'
    dimension_map = {
        "delicate-rugged": "rugged-delicate",
        "passive-active": "active-passive"
    }

    freq_categories = ["natural", "constructed"]
    freq_categories_map = {
        "natural": "Nat.",
        "constructed": "Con."
    }
    
    # Map JSON keys to LaTeX column headers
    input_types_map = {
        "original": "Original",
        "original_and_audio": "Original + Audio",
        "ipa": "IPA",
        "ipa_and_audio": "IPA + Audio",
        "audio": "Audio"
    }
    input_types_order = list(input_types_map.keys())

    # Initialize an empty string to hold all tables
    all_tables_latex_string = ""

    for model_name, model_data in data.items():
        # Create a DataFrame to hold the scores for the current model
        df = pd.DataFrame(index=dimensions_order, columns=pd.MultiIndex.from_product([input_types_order, freq_categories]))

        # Populate the DataFrame
        for freq in freq_categories:
            for input_type in input_types_order:
                for dim in dimensions_order:
                    try:
                        # Normalize dimension names
                        normalized_dim = dimension_map.get(dim, dim)
                        
                        score = model_data[freq][input_type]["dimensions"][normalized_dim]["macro_f1_score"]
                        # Format score: multiply by 100, round to 1 decimal
                        formatted_score = f"{score * 100:.1f}"
                        df.loc[dim, (input_type, freq)] = formatted_score
                    except KeyError:
                        # Fill with '--' if data is missing
                        df.loc[dim, (input_type, freq)] = "--"
        
        # Sanitize model name for LaTeX label
        safe_model_name_label = re.sub(r'[^a-zA-Z0-9]', '', model_name)

        # Start table for the model
        # Generate multicolumn headers for input types
        input_type_headers = " & ".join([f"\\multicolumn{{{len(freq_categories)}}}{{c}}{{\\textbf{{{input_types_map[it]}}}}}" for it in input_types_order])
        
        # Generate subheaders for frequency categories, repeated for each input type
        freq_headers = " & ".join(freq_categories_map.values())
        repeated_freq_headers = " & ".join([freq_headers] * len(input_types_order))

        # Generate cmidrule ranges
        num_freq = len(freq_categories)
        cmidrules = " ".join([f"\\cmidrule(lr){{{2 + i*num_freq}-{2 + i*num_freq + num_freq - 1}}}" for i in range(len(input_types_order))])

        model_table_string = f"""% =====================  {model_name} =====================
\\begin{{table*}}[ht]
\\centering
\\small
\\begin{{tabular}}{{l*{{{len(input_types_order) * len(freq_categories)}}}{{c}}}}
\\toprule
\\multirow{{2}}{{*}}{{\\textbf{{Dimension}}}} 
& {input_type_headers} \\\\
{cmidrules}
& {repeated_freq_headers} \\\\
\\midrule
"""
        # Add data rows
        for index, row in df.iterrows():
            dim_label = index
            row_values = " & ".join(row.fillna("-"))
            model_table_string += f"{dim_label:<35} & {row_values} \\\\ \n"

        # End table
        model_table_string += f"""\\bottomrule
\\end{{tabular}}
\\caption{{Detailed semantic dimension macro-F1 score results for {model_name}. ``--'' denotes a dimension where all ground truth features are classified as ``neither'' thus removed.}}
\\label{{tab:semdim_detailed_{safe_model_name_label}}}
\\end{{table*}}
"""
        all_tables_latex_string += model_table_string + "\n\n"

    # Write the LaTeX string to a file
    with open('results/statistics/semdim_latex_table.tex', 'w', encoding='utf-8') as f:
        f.write(all_tables_latex_string)


if __name__ == '__main__':
    # Path to the JSON file
    json_file_path = 'results/statistics/semdim_stat.json'
    generate_latex_tables(json_file_path)