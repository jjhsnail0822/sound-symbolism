import os
import pickle as pkl
import argparse
import torch
import json
from tqdm import tqdm
import gc
# python src/analysis/heatmap/combine_pkl_files.py --language ja --data_type audio

NAT_LANGS = ["en", "fr", "ja", "ko"]
CON_LANGS = ["art"]

def get_nat_or_con(language):
    if language in NAT_LANGS:
        return "nat"
    elif language in CON_LANGS:
        return "con"
    else:
        raise ValueError(f"Unknown language: {language}")

def load_word_list(data_path, language):
    with open(data_path, "r") as f:
        data = json.load(f)[language]
    return [sample["word"] for sample in data]

def safe_load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

def combine_pkl_files(base_save_dir, load_dir:str, language, data_path, dim_pairs):
    nat_or_con = get_nat_or_con(language)

    word_list = load_word_list(data_path, language)
    for data_type in ["ipa", "audio"]:
        for word in tqdm(word_list):
            save_path = os.path.join(base_save_dir, data_type, language)
            save_file_name = os.path.join(save_path, f"{word}.pkl")
            if os.path.exists(save_file_name):
                continue
            combined = {}
            count = 0
            for dim1, dim2 in dim_pairs:
                # load_dir
                data_file_name = os.path.join(f"{dim1}_{dim2}", f"{word}_{dim1}_{dim2}.pkl")
                alt_file_name = f"{word}_{dim1}_{dim2}_generation_analysis.pkl"
                data_dir = os.path.join(load_dir.format(nat_or_con=nat_or_con, data_type=data_type, language=language), data_file_name)
                alt_dir = os.path.join(load_dir.format(nat_or_con=nat_or_con, data_type=data_type, language=language), alt_file_name)
                if not os.path.exists(data_dir) or not os.path.exists(alt_dir):
                    continue
                if os.path.exists(data_dir):
                    try:
                        data = safe_load_pkl(data_dir)
                        combined[data_file_name] = data
                        count += 1
                    except Exception as e:
                        print(f"[WARN] Could not load {data_dir}: {e}")
                    finally:
                        if 'data' in locals():
                            del data
                        gc.collect()
                        torch.cuda.empty_cache()
                if os.path.exists(alt_dir):
                    try:
                        data = safe_load_pkl(alt_dir)
                        combined[alt_file_name] = data
                        count += 1
                    except Exception as e:
                        print(f"[WARN] Could not load {alt_dir}: {e}")
                    finally:
                        if 'data' in locals():
                            del data
                        gc.collect()
                        torch.cuda.empty_cache()
            if combined:
                save_path = os.path.join(base_save_dir, data_type, language)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                save_file_name = os.path.join(save_path, f"{word}.pkl")
                with open(save_file_name, 'wb') as f:
                    pkl.dump(combined, f, protocol=pkl.HIGHEST_PROTOCOL)
                print(f"Combined {count} files into {save_file_name}")
            combined.clear()
    print(f"Combined {count} files")

def load_dim_pairs():
    return [
        ("abrupt", "continuous"), ("continuous", "abrupt"), ("active", "passive"), ("passive", "active"),
        ("beautiful", "ugly"), ("ugly", "beautiful"), ("big", "small"), ("small", "big"),
        ("dangerous", "safe"), ("safe", "dangerous"), ("exciting", "calming"), ("calming", "exciting"),
        ("fast", "slow"), ("slow", "fast"), ("good", "bad"), ("bad", "good"),
        ("happy", "sad"), ("sad", "happy"), ("hard", "soft"), ("soft", "hard"),
        ("harsh", "mellow"), ("mellow", "harsh"), ("heavy", "light"), ("light", "heavy"),
        ("inhibited", "free"), ("free", "inhibited"), ("interesting", "uninteresting"), ("uninteresting", "interesting"),
        ("masculine", "feminine"), ("feminine", "masculine"), ("orginary", "unique"), ("unique", "orginary"),
        ("pleasant", "unpleasant"), ("unpleasant", "pleasant"), ("realistic", "fantastical"), ("fantastical", "realistic"),
        ("rugged", "delicate"), ("delicate", "rugged"), ("sharp", "round"), ("round", "sharp"),
        ("simple", "complex"), ("complex", "simple"), ("solid", "nonsolid"), ("nonsolid", "solid"),
        ("strong", "weak"), ("weak", "strong"), ("structured", "disorganized"), ("disorganized", "structured"),
        ("tense", "relaxed"), ("relaxed", "tense"),
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple pkl files into one (using find_basic_info logic).")
    parser.add_argument('--language', type=str, default='en', help='Language prefix for keys')
    parser.add_argument('--data_path', type=str, default='data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json', help='Path to json file for word list')
    parser.add_argument('--data_type', type=str, default='audio', help='Data type')
    args = parser.parse_args()
    dim_pairs = load_dim_pairs()
    language = args.language
    data_type = args.data_type
    nat_or_con = get_nat_or_con(language)
    load_dir = "results/experiments/understanding/attention_heatmap/{nat_or_con}/semantic_dimension/{data_type}/{language}/generation_attention"
    base_save_dir = "results/experiments/understanding/attention_heatmap/combined"
    data_path = args.data_path
    combine_pkl_files(base_save_dir, load_dir, language, data_path, dim_pairs)
