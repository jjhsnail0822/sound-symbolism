import os
import argparse 
from itertools import product

import epitran

from phonemes import IPA_MAP

def is_exists(word):
    # TODO: Implement a check to see if the word exists in a dictionary or corpus.
    return False

def is_valid(word):
    if word[0] in ["ŋ", "ʒ"]:
        return False
    else:
        return True


def get_phonemes(lang):
    consonants = IPA_MAP[lang]["consonants"]["all"]
    vowels = IPA_MAP[lang]["vowels"]["all"]
    return consonants, vowels
        
def main(args):
    # Get IPA symbols of consonants and vowels for the specified language
    consonants, vowels = get_phonemes(args.lang)

    # Generate CVCV words
    cvcv_patterns = product(consonants, vowels, consonants, vowels)
    constructed_words = [''.join(pattern) for pattern in cvcv_patterns]
    print(f"[INFO] Generated {len(constructed_words)} CVCV patterns.")

    # filter out phonotactically illegal words
    constructed_words = [word for word in constructed_words if is_valid(word)]
    print(f"[INFO] Filtered to {len(constructed_words)} valid CVCV words.")

    # filter out existing word (Not implemented yet)
    constructed_words = [word for word in constructed_words if not is_exists(word)]
  
    # Save the constructed words to a file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'constructed_words_{args.lang}.txt')
    
    with open(output_path, 'w') as f:
        for word in constructed_words:
            f.write(f"{word}\n")
    print(f"[INFO] Generated {len(constructed_words)} CVCV words and saved to '{output_path}'.")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate CVCV words based on phonemes.")
    parser.add_argument('--lang', type=str, choices=IPA_MAP.keys(), default='ko')
    parser.add_argument('--output_dir', type=str, default='data/constructed_words')
    args = parser.parse_args()
    main(args)
