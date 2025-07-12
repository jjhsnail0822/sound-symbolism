import json
import pandas as pd
import re

def generate_latex_tables(data_path):
    """
    Loads statistics from a JSON file and generates a separate LaTeX table for each model.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Merge gpt-4o and gpt-4o-audio-preview into a single gpt-4o model
    if "gpt-4o" in data and "gpt-4o-audio-preview" in data:
        gpt4o_audio_data = data.pop("gpt-4o-audio-preview")
        for freq, freq_data in gpt4o_audio_data.items():
            if freq in data["gpt-4o"]:
                data["gpt-4o"][freq].update(freq_data)
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

    freq_categories = ["common", "rare", "constructed"]
    
    # Map JSON keys to LaTeX column headers
    input_types_map = {
        "original": "O",
        "original_and_audio": "O\\,+\\,A",
        "ipa": "IPA",
        "ipa_and_audio": "IPA\\,+\\,A",
        "audio": "A"
    }
    input_types_order = list(input_types_map.keys())

    # Initialize an empty string to hold all tables
    all_tables_latex_string = ""

    for model_name, model_data in data.items():
        # Create a DataFrame to hold the scores for the current model
        df = pd.DataFrame(index=dimensions_order, columns=pd.MultiIndex.from_product([freq_categories, input_types_order]))

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
                        df.loc[dim, (freq, input_type)] = formatted_score
                    except KeyError:
                        # Fill with '-' if data is missing
                        df.loc[dim, (freq, input_type)] = "-"
        
        # Sanitize model name for LaTeX label
        safe_model_name_label = re.sub(r'[^a-zA-Z0-9]', '', model_name)

        # Start table for the model
        model_table_string = f"""% =====================  {model_name} =====================
\\begin{{table*}}[ht]
\\centering
\\small
\\begin{{tabular}}{{l*{{15}}{{c@{{\\hspace{{5pt}}}}}}}}
\\toprule
\\multirow{{3}}{{*}}{{\\textbf{{Dimension}}}} 
& \\multicolumn{{5}}{{c}}{{\\textbf{{Common}}}} 
& \\multicolumn{{5}}{{c}}{{\\textbf{{Rare}}}} 
& \\multicolumn{{5}}{{c}}{{\\textbf{{Constructed}}}} \\\\
\\cmidrule(lr){{2-6}}\\cmidrule(lr){{7-11}}\\cmidrule(lr){{12-16}}
& {' & '.join(input_types_map.values())}
& {' & '.join(input_types_map.values())}
& {' & '.join(input_types_map.values())} \\\\
\\midrule
"""
        # Add data rows
        for index, row in df.iterrows():
            # Replace backslashes for LaTeX compatibility
            dim_label = index
            row_values = " & ".join(row.fillna("00.0"))
            model_table_string += f"{dim_label:<25} & {row_values} \\\\ \n"

        # End table
        model_table_string += f"""\\bottomrule
\\end{{tabular}}
\\caption{{Detailed semantic dimension macro-F1 score results for {model_name}.}}
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