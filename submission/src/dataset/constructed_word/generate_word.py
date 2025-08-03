import json
import argparse 
import unicodedata
from typing import List, Dict
from pathlib import Path
from itertools import product

from phoneme_data import PhonemeData


class CVCVWordGenerator:
    def __init__(self, phoneme_data:PhonemeData, ensure_nonword:bool=False):
        self.phoneme_data = phoneme_data
        self.ensure_nonword = ensure_nonword

        self.consonants = phoneme_data.phonemes['consonants']
        self.vowels = phoneme_data.phonemes['vowels']


    def generate(self):
        # IPA format
        print(f'[INFO] consonants: ', self.consonants)
        print(f'[INFO] vowels: ', self.vowels)
        print(f'[INFO] Expected: ', len(self.consonants)*len(self.vowels)* len(self.consonants)*len(self.vowels))
        cvcv_ipa_list = self.generate_cvcv_words() 
        
        # Alphabet Format
        cvcv_alphabet_list = [self.make_alphabet_word(cvcv) for cvcv in cvcv_ipa_list] 

        # Convert to list of strings
        cvcv_ipa_list = [' '.join(cvcv) for cvcv in cvcv_ipa_list]
        # cvcv_alphabet_list = [''.join(cvcv) for cvcv in cvcv_alphabet_list]

        # Ipa, Alphabet Pair 
        pairs_tuple = list(zip(cvcv_ipa_list, cvcv_alphabet_list))
        pairs_dict = [dict(word=word, ipa=ipa) for ipa, word in pairs_tuple]

        return pairs_dict

    def generate_cvcv_words(self) -> List[List[str]]:
        """
        Generate CVCV words in IPA format, using the consonants and vowels defined in IPA_PHONEMES.
        """
        cvcv_list = list(product(self.consonants, self.vowels, self.consonants, self.vowels))
        print("[INFO] Total cvcv ipa list # :", len(cvcv_list))
        
        result = []
        filter_out = []
        for cvcv in cvcv_list:
            if self.ensure_nonword:
                filter_condition = (not self._is_exist(cvcv)) and self._is_valid(cvcv)
            else:
                filter_condition = self._is_valid(cvcv)
            
            if filter_condition:
                result.append(list(cvcv))
            else:
                filter_out.append(list(cvcv))
            
        print(f'[INFO] Filtered {len(result)} valid CVCV combinations.')
        # print(f'[INFO] Removed: ', filter_out)
        return result
    
    # Make new column for Alphabet-written words(from IPA)
    def make_alphabet_word(self, cvcv_ipa) -> List[str]:
        c1, v1, c2, v2 = cvcv_ipa
        
        aph_result = {'c1': self.phoneme_data.ipa_to_alphabet[c1],
                'v1': self.phoneme_data.ipa_to_alphabet[v1],
                'c2': self.phoneme_data.ipa_to_alphabet[c2],
                'v2': self.phoneme_data.ipa_to_alphabet[v2]}
        
        form =  [aph_result['c1'], aph_result['v1'], '-', aph_result['c2'], aph_result['v2']]
        if c2 + v2 == 'ði':
            form = [aph_result['c1'], aph_result['v1'], aph_result['c2'], aph_result['v2']]

        word = ''.join(form)

        word = word.replace('thee', 'they')
        word = word.replace('thay', 'they')
        word = word.replace('tho', 'though')

        word = word.replace('to', 'toe')
        word = word.replace('do', 'doe')
        word = word.replace('gee', 'ghee')

        return word
    
    def _is_exist(self, cvcv):
        cvcv = [unicodedata.normalize('NFKC', phoneme) for phoneme in cvcv]
        cvcv = ''.join(cvcv)
        for lang, ipa_to_word in self.phoneme_data.ipa_to_word.items():
            if cvcv in ipa_to_word:
            #     print("is_exist", lang, cvcv, ipa_to_word[cvcv])
            #     # with open('debug_filtered_out.csv', 'a', encoding="utf-8") as f:
            #     #     f.write(f'{lang},{cvcv},{ipa_to_word[cvcv]}\n')
                return True   
        return False

    def _is_valid(self, cvcv):
        
        cvcv = [unicodedata.normalize('NFKC', phoneme) for phoneme in cvcv]
        # print(cvcv)
        # words starting with "ð", rather than "θ", are less common in English
        if cvcv[0] == "ð" and cvcv[1] == 'i':
            return False 
        if cvcv[0] == "ð" and cvcv[1] == 'ɑ':
            return False 
        if cvcv[2] == "ð" and cvcv[3] == 'ɑ':
            return False 

        
        return True

def add_metadata(words:List[Dict[str, str]]):
    result =[]
    for word in words:
        word.update({
            'ref': 'Higher order factors of sound symbolism',
            'ipa_source' :'epitran',
            'romanization' : word['word'],
            'language' : 'art',

        })
        result.append(word)
    return {'art': result}

def main(args):
    phoneme_data = PhonemeData(args.data_dir)
    cvcv = CVCVWordGenerator(phoneme_data, args.ensure_nonword)
    words = cvcv.generate()

    results = add_metadata(words)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("[INFO] Saved to :", args.output_path)



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate CVCV words based on phonemes.")
    parser.add_argument('--data_dir', type=str, default='data/processed/art/resources')
    parser.add_argument('--output_path', type=str, default='data/processed/art/semantic_dimension/semantic_dimension_binary_gt.json')
    parser.add_argument('--ensure_nonword', action='store_true')
    args = parser.parse_args()
    main(args)
