import os
import json
import matplotlib.pyplot as plt
input_types = ['audio', 'ipa', 'original', 'original_and_audio', 'ipa_and_audio']
src_dir = "./results/expeirments/semantic_dimension/binary"
file_name = "all_results_Qwen_Qwen2.5-Omni-7B.json"
output_dir = "./results/plots/experiments/semantic_dimension/binary"

def run():
    for input_type in input_types:
        file_path = os.path.join(src_dir, input_type, file_name)
        data = load_data(file_path)
    return

def plot_bar_chart(data, output_dir):
    return

def plot_line_chart(data, output_dir):
    return

def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    run()