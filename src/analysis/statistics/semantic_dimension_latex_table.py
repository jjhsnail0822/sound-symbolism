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
        "natural": "Natural",
        "constructed": "Constructed"
    }
    
    # Map JSON keys to LaTeX column headers
    input_types_map = {
        "original": "Original",
        "ipa": "IPA",
        "audio": "Audio"
    }
    input_types_order = list(input_types_map.keys())
    metrics = ["accuracy", "macro_f1_score"]
    metrics_map = {
        "accuracy": "Acc.",
        "macro_f1_score": "F1"
    }

    # Initialize an empty string to hold all tables
    all_tables_latex_string = ""

    for model_name, model_data in data.items():
        # Create a DataFrame for the current model
        columns = [f"{fc}_{it}_{m}" for fc in freq_categories for it in input_types_order for m in metrics]
        df = pd.DataFrame(index=dimensions_order, columns=columns)
        df.fillna("--", inplace=True) # Initialize with placeholder

        # Fill the DataFrame with the data
        for dim in dimensions_order:
            # Handle reversed dimensions
            lookup_dim = dimension_map.get(dim, dim)
            
            for fc in freq_categories:
                for it in input_types_order:
                    for m_key in metrics:
                        try:
                            # Navigate through the data structure to get the score
                            score = model_data[fc][it][lookup_dim]['all'][m_key]
                            # Format the score to xx.x
                            df.at[dim, f"{fc}_{it}_{m_key}"] = f"{score * 100:.1f}"
                        except KeyError:
                            # If data is missing, the cell will retain its "--" value
                            pass
        
        # Sanitize model name for LaTeX label
        safe_model_name_label = re.sub(r'[^a-zA-Z0-9]', '', model_name)

        # Generate multicolumn headers for frequency categories
        num_metrics = len(metrics)
        num_input_types = len(input_types_order)
        freq_cat_headers = " & ".join([f"\\multicolumn{{{num_input_types * num_metrics}}}{{c}}{{\\textbf{{{freq_categories_map[fc]}}}}}" for fc in freq_categories])
        
        # Generate subheaders for input types, repeated for each freq category
        input_type_headers_list = [f"\\multicolumn{{{num_metrics}}}{{c}}{{{input_types_map[it]}}}" for it in input_types_order]
        repeated_input_type_headers = " & ".join(input_type_headers_list * len(freq_categories))

        # Generate sub-subheaders for metrics
        metric_headers = " & ".join(metrics_map.values())
        repeated_metric_headers = " & ".join([metric_headers] * (num_input_types * len(freq_categories)))

        # Generate cmidrule ranges
        # Top level (for freq categories)
        cmidrules_top = " ".join([f"\\cmidrule(lr){{{2 + i*num_input_types*num_metrics}-{2 + (i+1)*num_input_types*num_metrics - 1}}}" for i in range(len(freq_categories))])
        # Second level (for input types)
        cmidrules_mid = " ".join([f"\\cmidrule(lr){{{2 + i*num_metrics}-{2 + (i+1)*num_metrics - 1}}}" for i in range(num_input_types * len(freq_categories))])

        total_cols = len(df.columns)
        model_table_string = f"""% =====================  {model_name} =====================
\\begin{{table*}}[ht]
\\centering
\\begin{{tabular}}{{l*{{{total_cols}}}{{c}}}}
\\toprule
\\multirow{{3}}{{*}}{{\\textbf{{Dimension}}}} 
& {freq_cat_headers} \\\\
{cmidrules_top}
& {repeated_input_type_headers} \\\\
{cmidrules_mid}
& {repeated_metric_headers} \\\\
\\midrule
"""
        # Add data rows
        for index, row in df.iterrows():
            dim_label = index
            row_values = " & ".join(row.values)
            model_table_string += f"{dim_label:<35} & {row_values} \\\\ \n"

        # End table
        model_table_string += f"""\\bottomrule
\\end{{tabular}}
\\caption{{Detailed semantic dimension prediction accuracy and macro-F1 score results for {model_name}.}}
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