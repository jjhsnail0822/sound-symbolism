'''A source code file for plotting meta data of words.
- total number of words
- total number of words by language
- total number of words by frequency (common vs. rare)
- number of meanings according to words
- several example words that represents each groups (language and frequency)
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from pprint import pprint
import json
import re

data_base_path:os.path = "data/processed/nat"
languages = ["en", "fr", "ko", "ja"]
frequency = ["common", "rare"]
semantic_dimensions = None # TODO

def load_data(language:str=None, freq:str=None):
    if freq is None:
        data_path = os.path.join(data_base_path, f"{language}.json")
    else:
        data_path = os.path.join(data_base_path, f"{freq}_words.json")
        
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def total_number_of_words(*args) -> int:
    # args: word data loaded by load_data()
    if isinstance(args[0], list):
        # extend data
        data = []
        for arg in args:
            data.extend(arg)
    else:
        data = args
    return len(data)

def total_number_of_meanings_by_language(data:list[dict], language:str) -> int:
    if language == "ko":
        meanings_key = "definitions"
    else:
        meanings_key = "meaning"
    
    if isinstance(data[0][meanings_key], list):
        return sum(len(word[meanings_key]) for word in data)
    else:
        return sum(1 for word in data if word[meanings_key] is not None)

def ratio_of_words_by_language(en_data:list[dict], fr_data:list[dict], ja_data:list[dict], ko_data:list[dict]) -> dict[str, float]:
    total_words = total_number_of_words(en_data, fr_data, ja_data, ko_data)
    en_ratio = len(en_data) / total_words
    fr_ratio = len(fr_data) / total_words
    ja_ratio = len(ja_data) / total_words
    ko_ratio = len(ko_data) / total_words
    return {"en": en_ratio, "fr": fr_ratio, "ja": ja_ratio, "ko": ko_ratio}

def freq_data_statistics(common_data:list[dict], rare_data:list[dict], en_data:list[dict]=None, fr_data:list[dict]=None, ja_data:list[dict]=None, ko_data:list[dict]=None) -> dict[str, float]:
    total_words = total_number_of_words(common_data, rare_data)
    common_words = len(common_data)
    rare_words = len(rare_data)
    
    common_en_words = len([word for word in common_data if word["language"] == "en"])
    common_fr_words = len([word for word in common_data if word["language"] == "fr"])
    common_ja_words = len([word for word in common_data if word["language"] == "ja"])
    common_ko_words = len([word for word in common_data if word["language"] == "ko"])
    
    rare_en_words = len([word for word in rare_data if word["language"] == "en"])
    rare_fr_words = len([word for word in rare_data if word["language"] == "fr"])
    rare_ja_words = len([word for word in rare_data if word["language"] == "ja"])
    rare_ko_words = len([word for word in rare_data if word["language"] == "ko"])

    result = {
        "common": common_words / total_words, "rare": rare_words / total_words,
        "common_en": common_en_words, "rare_en": rare_en_words,
        "common_fr": common_fr_words, "rare_fr": rare_fr_words,
        "common_ja": common_ja_words, "rare_ja": rare_ja_words,
        "common_ko": common_ko_words, "rare_ko": rare_ko_words,
        "common_en_ratio": common_en_words / common_words, "rare_en_ratio": rare_en_words / rare_words,
        "common_fr_ratio": common_fr_words / common_words, "rare_fr_ratio": rare_fr_words / rare_words,
        "common_ja_ratio": common_ja_words / common_words, "rare_ja_ratio": rare_ja_words / rare_words,
        "common_ko_ratio": common_ko_words / common_words, "rare_ko_ratio": rare_ko_words / rare_words,
        "common_en_total": common_en_words / total_words, "rare_en_total": rare_en_words / total_words,
        "common_fr_total": common_fr_words / total_words, "rare_fr_total": rare_fr_words / total_words,
        "common_ja_total": common_ja_words / total_words, "rare_ja_total": rare_ja_words / total_words,
        "common_ko_total": common_ko_words / total_words, "rare_ko_total": rare_ko_words / total_words,
    }
    
    # Add ratio_in_lang if language data is provided
    if en_data is not None:
        result["common_en_ratio_in_lang"] = common_en_words / len(en_data)
        result["rare_en_ratio_in_lang"] = rare_en_words / len(en_data)
    if fr_data is not None:
        result["common_fr_ratio_in_lang"] = common_fr_words / len(fr_data)
        result["rare_fr_ratio_in_lang"] = rare_fr_words / len(fr_data)
    if ja_data is not None:
        result["common_ja_ratio_in_lang"] = common_ja_words / len(ja_data)
        result["rare_ja_ratio_in_lang"] = rare_ja_words / len(ja_data)
    if ko_data is not None:
        result["common_ko_ratio_in_lang"] = common_ko_words / len(ko_data)
        result["rare_ko_ratio_in_lang"] = rare_ko_words / len(ko_data)
    
    return result

def print_as_table(processed_data:dict):
    """Print data in formatted tables for better readability"""
    
    # Helper function to format percentage
    def format_percent(value):
        if isinstance(value, float) and value <= 1.0:
            return f"{value * 100:.2f}%"
        elif isinstance(value, float):
            return f"{value:.2f}%"
        else:
            return str(value)
    
    # Helper function to print horizontal table with vertical lines
    def print_horizontal_table(title, headers, rows):
        print(f"\n{'='*120}")
        print(f"{title:^120}")
        print(f"{'='*120}")
        
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)
        
        # Print header with vertical lines
        header_line = "|"
        for i, header in enumerate(headers):
            header_line += f" {header:^{col_widths[i]}} |"
        print(header_line)
        
        # Print separator line
        separator = "|"
        for width in col_widths:
            separator += "-" * (width + 2) + "|"
        print(separator)
        
        # Print rows with vertical lines
        for row in rows:
            row_line = "|"
            for i, cell in enumerate(row):
                row_line += f" {str(cell):^{col_widths[i]}} |"
            print(row_line)
    
    # Calculate language distribution from the actual data
    total_words = 8052
    en_total = processed_data.get("common_en", 0) + processed_data.get("rare_en", 0)
    fr_total = processed_data.get("common_fr", 0) + processed_data.get("rare_fr", 0)
    ja_total = processed_data.get("common_ja", 0) + processed_data.get("rare_ja", 0)
    ko_total = processed_data.get("common_ko", 0) + processed_data.get("rare_ko", 0)
    
    # Table 1: Combined Language Distribution, Common Words, and Rare Words
    combined_headers = ["Language", "Total Words", "Common Words", "Rare Words"]
    
    # Calculate totals
    total_common = sum([processed_data.get("common_en", 0), processed_data.get("common_fr", 0), 
                       processed_data.get("common_ja", 0), processed_data.get("common_ko", 0)])
    total_rare = sum([processed_data.get("rare_en", 0), processed_data.get("rare_fr", 0), 
                     processed_data.get("rare_ja", 0), processed_data.get("rare_ko", 0)])
    
    # combined_rows = [
    #     ["English", f"{en_total} ({format_percent(en_total / total_words)})", 
    #      f"{processed_data.get('common_en', 0)} ({format_percent(processed_data.get('common_en_ratio', 0))}, {format_percent(processed_data.get('common_en_ratio_in_lang', 0))})",
    #      f"{processed_data.get('rare_en', 0)} ({format_percent(processed_data.get('rare_en_ratio', 0))}, {format_percent(processed_data.get('rare_en_ratio_in_lang', 0))})"],
    #     ["French", f"{fr_total} ({format_percent(fr_total / total_words)})",
    #      f"{processed_data.get('common_fr', 0)} ({format_percent(processed_data.get('common_fr_ratio', 0))}, {format_percent(processed_data.get('common_fr_ratio_in_lang', 0))})",
    #      f"{processed_data.get('rare_fr', 0)} ({format_percent(processed_data.get('rare_fr_ratio', 0))}, {format_percent(processed_data.get('rare_fr_ratio_in_lang', 0))})"],
    #     ["Japanese", f"{ja_total} ({format_percent(ja_total / total_words)})",
    #      f"{processed_data.get('common_ja', 0)} ({format_percent(processed_data.get('common_ja_ratio', 0))}, {format_percent(processed_data.get('common_ja_ratio_in_lang', 0))})",
    #      f"{processed_data.get('rare_ja', 0)} ({format_percent(processed_data.get('rare_ja_ratio', 0))}, {format_percent(processed_data.get('rare_ja_ratio_in_lang', 0))})"],
    #     ["Korean", f"{ko_total} ({format_percent(ko_total / total_words)})",
    #      f"{processed_data.get('common_ko', 0)} ({format_percent(processed_data.get('common_ko_ratio', 0))}, {format_percent(processed_data.get('common_ko_ratio_in_lang', 0))})",
    #      f"{processed_data.get('rare_ko', 0)} ({format_percent(processed_data.get('rare_ko_ratio', 0))}, {format_percent(processed_data.get('rare_ko_ratio_in_lang', 0))})"],
    #     ["Total", f"{total_words} (100.000%)", 
    #      f"{total_common} ({format_percent(processed_data.get('common', 0))})",
    #      f"{total_rare} ({format_percent(processed_data.get('rare', 0))})"]
    # ]
    combined_rows = [
        ["English", f"{en_total}", 
         f"{processed_data.get('common_en', 0)}",
         f"{processed_data.get('rare_en', 0)}"],
        ["French", f"{fr_total}",
         f"{processed_data.get('common_fr', 0)}",
         f"{processed_data.get('rare_fr', 0)}"],
        ["Japanese", f"{ja_total}",
         f"{processed_data.get('common_ja', 0)}",
         f"{processed_data.get('rare_ja', 0)}"],
        ["Korean", f"{ko_total}",
         f"{processed_data.get('common_ko', 0)}",
         f"{processed_data.get('rare_ko', 0)}"],
        ["Total", f"{total_words}", 
         f"{total_common}",
         f"{total_rare}"]
    ]
    print_horizontal_table("Language Distribution & Word Frequency Analysis", combined_headers, combined_rows)
    
    print(f"\n{'='*120}")
    print("Note: 'Within Common/Rare' = percentage within that frequency category")
    print("      'Within Language' = percentage within that language's total words")
    print(f"{'='*120}")

if __name__ == "__main__":
    en_data = load_data("en")
    fr_data = load_data("fr")
    ja_data = load_data("ja")
    ko_data = load_data("ko")
    common_data = load_data(freq="common")
    rare_data = load_data(freq="rare")
    total_number_of_words(en_data, fr_data, ja_data, ko_data)
    for lang in languages:
        total_number_of_meanings_by_language(eval(f"{lang}_data"), lang)
    ratio_of_words_by_language(en_data, fr_data, ja_data, ko_data)
    freq_data_statistics(common_data, rare_data, en_data, fr_data, ja_data, ko_data)
    print_as_table(freq_data_statistics(common_data, rare_data, en_data, fr_data, ja_data, ko_data))