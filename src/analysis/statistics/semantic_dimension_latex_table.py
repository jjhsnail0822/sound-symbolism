import json
import re
import pandas as pd

def generate_latex_tables(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
        # Create a DataFrame for the current model
        # Columns will be a MultiIndex of (input_type, freq_category)
        # but for simplicity we'll handle it as flat columns first.
        columns = [f"{it}_{fc}" for it in input_types_order for fc in freq_categories]
        df = pd.DataFrame(index=dimensions_order, columns=columns)
        df.fillna("--", inplace=True) # Initialize with placeholder

        # Fill the DataFrame with the data
        for dim in dimensions_order:
            # Handle reversed dimensions
            lookup_dim = dimension_map.get(dim, dim)
            
            for it in input_types_order:
                for fc in freq_categories:
                    try:
                        # Navigate through the data structure to get the score
                        score = model_data[fc][it][lookup_dim]['all']['macro_f1_score']
                        # Format the score to xx.x
                        df.at[dim, f"{it}_{fc}"] = f"{score * 100:.1f}"
                    except KeyError:
                        # If data is missing, the cell will retain its "--" value
                        pass
        
        # Sanitize model name for LaTeX label
        safe_model_name_label = re.sub(r'[^a-zA-Z0-9]', '', model_name)

        # Generate multicolumn headers for input types
        input_type_headers = " & ".join([f"\\multicolumn{{{len(freq_categories)}}}{{c}}{{\\textbf{{{input_types_map[it]}}}}}" for it in input_types_order])
        
        # Generate subheaders for frequency categories, repeated for each input type
        freq_headers = " & ".join(freq_categories_map.values())
        repeated_freq_headers = " & ".join([freq_headers] * len(input_types_order))

        # Generate cmidrule ranges
        num_freq = len(freq_categories)
        # The first column is the dimension label, so data columns start at index 2 in LaTeX
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
            dim_label = index.replace('_', ' ')
            row_values = " & ".join(row.values)
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
    output_path = 'results/statistics/semdim_latex_table.tex'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(all_tables_latex_string)
    print(f"LaTeX tables generated at: {output_path}")


if __name__ == '__main__':
    # Path to the JSON file
    json_file_path = 'results/statistics/semdim_stat.json'
    generate_latex_tables(json_file_path)