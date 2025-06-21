import os
import unicodedata
import argparse 
from itertools import product

import epitran

from phonemes import IPA_MAP, IPA_TO_ALPHABET



def get_phonemes(mode:str):
    results = []
    for key, value in IPA_MAP[mode].items():
        results.extend(value)

    return results

def convert_to_alphabet(cvcv_ipa_list):
    cvcv_alphabet_list = []
    for cvcv_ipa in cvcv_ipa_list:
        c1, v1, c2, v2 = cvcv_ipa
        
        if c1 == "ɡ" and v1 == "i":
            c1 = 'gh'
        else:
            c1 = IPA_TO_ALPHABET[c1]

        if v1 == "ow":
            if c1 == 'sh':
                v1 = 'ow'
            elif c1 == 'd':
                v1 = 'ough'
            elif c2 == 'n':
                v1 = 'oa'
            else:
                v1 = 'oe'
        else:
            v1 = IPA_TO_ALPHABET[v1]

        if c2 == "ɡ" and v2 == "i":
            c2 = "gg"     
        else:
            c2 = IPA_TO_ALPHABET[c2]

        if v2 == "ow":
            if c2 == 'th':
                c2 = '-th'
                v2 = 'ough'
            elif c2 == 'k':
                v2 = 'o'
            else:
                v2 = "ow"
        elif v2 == "i" and c2 == 'th':
            v2 = 'ey'
        elif v2 == "ej":
            if c2 == 'th':
                c2 = '-th'
                v2 = 'ey'
            else:
                v2 = "aye"
        else:
            v2 = IPA_TO_ALPHABET[v2] 
        
        cvcv_alphabet_list.append([c1, v1, c2, v2])
    return cvcv_alphabet_list

def get_ipa_cvcv_words(consonants, vowels):
    cvcv_ipa_list = product(consonants, vowels, consonants, vowels)
    
    def is_exists(word):
        # TODO: Implement a check to see if the word exists in a dictionary or corpus.
        return False

    def is_valid(word):
        word = [unicodedata.normalize('NFKC', phoneme) for phoneme in word]
        # words starting with "ð", rather than "θ", are less common in English
        if word[0] == "ð":
            return False 
        if word[-1] == unicodedata.normalize("NFKC", "ɑ"):
            # words ending with "ɑ" are less common in English
            return False
        
        return True

    cvcv_ipa_list = [list(cvcv_ipa) for cvcv_ipa in cvcv_ipa_list if is_valid(cvcv_ipa)]
    cvcv_ipa_list = [list(cvcv_ipa) for cvcv_ipa in cvcv_ipa_list if not is_exists(cvcv_ipa)]
    
    return cvcv_ipa_list
    


def main(args):
    consonants = get_phonemes('consonants')
    vowels = get_phonemes('vowels')

    # 1. Generate CVCV words in IPA format
    cvcv_ipa_list = get_ipa_cvcv_words(consonants, vowels)
    print(f"[INFO] Generated {len(cvcv_ipa_list)} CVCV words in IPA format.")

    ## 2. Convert IPA to Alphabet
    cvcv_alphabet_list = convert_to_alphabet(cvcv_ipa_list)

    # 3. Convert lists of lists to strings
    cvcv_ipa_list = [''.join(cvcv) for cvcv in cvcv_ipa_list]
    cvcv_alphabet_list = [''.join(cvcv) for cvcv in cvcv_alphabet_list]

    # 4. Combine IPA and Alphabet lists into a list of tuples
    constructed_words = list(zip(cvcv_ipa_list, cvcv_alphabet_list))


    # 5. Save the constructed words to a file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'constructed_words.csv')

    with open(output_path, 'w') as f:
        f.write("index,ipa,alphabet\n")
    
    with open(output_path, 'a') as f:
        for i, (ipa, alphabet) in enumerate(constructed_words):
            f.write(f"{i},{ipa},{alphabet}\n")

    print(f"[INFO] Generated {len(constructed_words)} CVCV words and saved to '{output_path}'.")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate CVCV words based on phonemes.")
    parser.add_argument('--output_dir', type=str, default='/data/constructed_words')
    args = parser.parse_args()
    main(args)
