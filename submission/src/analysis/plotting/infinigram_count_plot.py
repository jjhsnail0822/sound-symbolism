import json
import matplotlib.pyplot as plt
import numpy as np
import os

base_data_path_relative_to_script = "data/processed/nat/"
output_plot_dir_relative_to_script = "results/plots/infinigram_counts"

# Create plot output directory if it doesn't exist
os.makedirs(output_plot_dir_relative_to_script, exist_ok=True)

files_to_process = [
    {"filename": "ko.json", "display_name": "Korean", "file_basename": "korean"},
    {"filename": "ja.json", "display_name": "Japanese", "file_basename": "japanese"},
    {"filename": "fr.json", "display_name": "French", "file_basename": "french"},
    {"filename": "en.json", "display_name": "English", "file_basename": "english"}
]

for file_spec in files_to_process:
    # Create a new Figure and Axes for each plot
    # Adjust figsize to be nearly square for subfigures
    fig_ind, ax_ind = plt.subplots(figsize=(8, 6))

    current_data_file_path = os.path.join(base_data_path_relative_to_script, file_spec['filename'])

    try:
        if not os.path.exists(current_data_file_path):
            error_msg = f"File not found:\n{current_data_file_path}"
            print(error_msg) # Print error message to console
            ax_ind.text(0.5, 0.5, error_msg, ha='center', va='center', color='red', fontsize=16)
            ax_ind.set_title(f"{file_spec['display_name']} - File Not Found", fontsize=20)
            # Save an empty plot or a plot with error message even if an error occurs

        else:
            with open(current_data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            current_counts = [item['infinigram_count']['total'] for item in data]

            if not current_counts:
                no_data_msg = "No data in file"
                print(f"{file_spec['display_name']}: {no_data_msg}")
                ax_ind.text(0.5, 0.5, no_data_msg, ha='center', va='center', fontsize=18)
                ax_ind.set_title(f"{file_spec['display_name']} - No Data", fontsize=20)
            else:
                counts_zero = sum(1 for count in current_counts if count == 0)
                counts_positive = [count for count in current_counts if count > 0]

                # For paper subfigures, recommend concise titles (e.g., ax_ind.set_title(file_spec['display_name'], fontsize=20))
                plot_title = f"{file_spec['display_name']}\n(Total: {len(current_counts)}, Zeros: {counts_zero}, Positive: {len(counts_positive)})"
                ax_ind.set_title(plot_title, fontsize=20)

                if counts_positive:
                    min_val = np.min(counts_positive)
                    max_val = np.max(counts_positive)
                    
                    if min_val == max_val: # All positive values are the same
                         bins = [min_val * 0.9, min_val * 1.1] if min_val > 0 else [0.9, 1.1]
                    elif max_val / min_val < 10: # Value range is narrow
                         bins = np.linspace(min_val, max_val, 20) 
                    else: # Typical case: use log scale bins
                         bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
                    
                    ax_ind.hist(counts_positive, bins=bins, color='skyblue', edgecolor='black')
                    ax_ind.set_xscale('log')
                    ax_ind.set_xlabel('Infini-gram Count (log scale)', fontsize=18)
                else:
                    ax_ind.text(0.5, 0.5, "No positive counts to display", ha='center', va='center', fontsize=18)
                    ax_ind.set_xlabel('Infini-gram Count', fontsize=18)

                ax_ind.set_ylabel('Frequency', fontsize=18)
                ax_ind.tick_params(axis='both', which='major', labelsize=16)
                # ax_ind.grid(True, which="both", ls="--", alpha=0.7)
                ax_ind.grid(False)

    except Exception as e:
        error_detail = f"Error processing file:\n{current_data_file_path}\n{e}"
        print(error_detail) # Print error details to console
        ax_ind.text(0.5, 0.5, error_detail, ha='center', va='center', color='red', fontsize=16)
        ax_ind.set_title(f"{file_spec['display_name']} - Processing Error", fontsize=20)

    # Save individual plot
    output_filename = f"{file_spec['file_basename']}_infinigram_distribution.png"
    output_filepath = os.path.join(output_plot_dir_relative_to_script, output_filename)
    output_filepath_pdf = os.path.join(output_plot_dir_relative_to_script, f"{file_spec['file_basename']}_infinigram_distribution.pdf")

    fig_ind.tight_layout() # Adjust layout for each Figure
    fig_ind.savefig(output_filepath, dpi=300)
    fig_ind.savefig(output_filepath_pdf)  # Also save as PDF
    print(f"Saved plot: {output_filepath} and {output_filepath_pdf}")
    
    plt.close(fig_ind) # Close Figure object to free memory

print(f"All plots saved to: {output_plot_dir_relative_to_script}")