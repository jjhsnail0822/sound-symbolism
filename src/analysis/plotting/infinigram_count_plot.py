import json
import matplotlib.pyplot as plt
import numpy as np
import os

base_data_path_relative_to_script = "data/processed/nat/"
output_plot_dir_relative_to_script = "results/plots/infinigram_counts"

# 플롯 저장 디렉토리 생성 (없는 경우)
os.makedirs(output_plot_dir_relative_to_script, exist_ok=True)

files_to_process = [
    {"filename": "ko.json", "display_name": "Korean (ko.json)", "file_basename": "korean"},
    {"filename": "ja.json", "display_name": "Japanese (ja.json)", "file_basename": "japanese"},
    {"filename": "fr.json", "display_name": "French (fr.json)", "file_basename": "french"},
    {"filename": "en.json", "display_name": "English (en.json)", "file_basename": "english"}
]

for file_spec in files_to_process:
    # 각 플롯에 대해 새 Figure와 Axes 생성
    fig_ind, ax_ind = plt.subplots(figsize=(10, 7))

    current_data_file_path = os.path.join(base_data_path_relative_to_script, file_spec['filename'])

    try:
        if not os.path.exists(current_data_file_path):
            error_msg = f"File not found:\n{current_data_file_path}"
            print(error_msg) # 콘솔에도 에러 메시지 출력
            ax_ind.text(0.5, 0.5, error_msg, ha='center', va='center', color='red', fontsize=9)
            ax_ind.set_title(f"{file_spec['display_name']} - File Not Found")
            # 오류가 발생한 경우에도 빈 플롯이나 오류 메시지가 담긴 플롯을 저장합니다.

        else:
            with open(current_data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            current_counts = [item['infinigram_count']['total'] for item in data]

            if not current_counts:
                no_data_msg = "No data in file"
                print(f"{file_spec['display_name']}: {no_data_msg}")
                ax_ind.text(0.5, 0.5, no_data_msg, ha='center', va='center')
                ax_ind.set_title(f"{file_spec['display_name']} - No Data")
            else:
                counts_zero = sum(1 for count in current_counts if count == 0)
                counts_positive = [count for count in current_counts if count > 0]

                plot_title = f"{file_spec['display_name']}\n(Total: {len(current_counts)}, Zeros: {counts_zero}, Positive: {len(counts_positive)})"
                ax_ind.set_title(plot_title)

                if counts_positive:
                    min_val = np.min(counts_positive)
                    max_val = np.max(counts_positive)
                    
                    if min_val == max_val: # 모든 양수 값이 같을 경우
                         bins = [min_val * 0.9, min_val * 1.1] if min_val > 0 else [0.9, 1.1]
                    elif max_val / min_val < 10: # 값의 범위가 좁을 경우
                         bins = np.linspace(min_val, max_val, 20) 
                    else: # 일반적인 경우 로그 스케일 빈 사용
                         bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
                    
                    ax_ind.hist(counts_positive, bins=bins, color='skyblue', edgecolor='black')
                    ax_ind.set_xscale('log')
                    ax_ind.set_xlabel('Infinigram Count (log scale)')
                else:
                    ax_ind.text(0.5, 0.5, "No positive counts to display", ha='center', va='center')
                    ax_ind.set_xlabel('Infinigram Count')

                ax_ind.set_ylabel('Frequency')
                ax_ind.grid(True, which="both", ls="--", alpha=0.7)

    except Exception as e:
        error_detail = f"Error processing file:\n{current_data_file_path}\n{e}"
        print(error_detail) # 콘솔에도 에러 상세 정보 출력
        ax_ind.text(0.5, 0.5, error_detail, ha='center', va='center', color='red', fontsize=9)
        ax_ind.set_title(f"{file_spec['display_name']} - Processing Error")

    # 개별 플롯 저장
    output_filename = f"{file_spec['file_basename']}_infinigram_distribution.png"
    output_filepath = os.path.join(output_plot_dir_relative_to_script, output_filename)

    fig_ind.tight_layout() # 개별 Figure에 대해 레이아웃 조정
    fig_ind.savefig(output_filepath)
    print(f"Saved plot: {output_filepath}")
    
    plt.close(fig_ind) # Figure 객체 닫아서 메모리 해제

print(f"All plots saved to: {output_plot_dir_relative_to_script}")