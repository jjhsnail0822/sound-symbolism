#!/usr/bin/env python3
"""
단어 리스트를 받아서 IPA 변환만 수행하는 스크립트
사용 예시:
python ipa_batch_converter.py --input words_for_ipa.json --output words_with_ipa.json --lang en
"""
import argparse
import json
import epitran
from tqdm import tqdm

language_code = {"ko": "kor-Hang", "en": "eng-Latn", "fr": "fra-Latn", "ja": "jpn-Hrgn"}

def batch_ipa_convert(input_file, output_file, lang):
    epi = epitran.Epitran(language_code[lang])
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in tqdm(data):
        word = item.get("word", "")
        try:
            item["ipa"] = epi.transliterate(word.strip().lower()) if word else ""
        except Exception as e:
            print(f"[WARNING] IPA 변환 실패: {word} ({e})")
            item["ipa"] = ""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ IPA 변환 결과 저장: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch IPA converter for word list")
    parser.add_argument('--input', '-i', required=True, help='Input JSON file (word list)')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file (with IPA)')
    parser.add_argument('--lang', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], help='Language code')
    args = parser.parse_args()
    batch_ipa_convert(args.input, args.output, args.lang)

if __name__ == "__main__":
    main() 