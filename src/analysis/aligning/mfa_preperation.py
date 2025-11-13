
import json
import unicodedata
from pathlib import Path

import epitran
from ipa_map import EPITRAN_TO_MFA


def update_custom_dict(oov, dict_path, lang):
    EPI_DICT = {
        'en': 'eng-Latn',
        'fr': 'fra-Latn',
        'ja': 'jpn-Hrgn',
        'ko' : 'kor-Hang'
    }
    epi = epitran.Epitran(EPI_DICT[lang])
    result_dict = {}
    for vocab in oov:
        if vocab:
            ipa_epitran = convert_to_ipa(vocab, epi)
            try:
                ipa_mfa_list = [EPITRAN_TO_MFA[lang][ipa_epi] for ipa_epi in  ipa_epitran.split(' ')]
                result_dict[vocab] = ' '.join(ipa_mfa_list)
            except KeyError as e:
                if ipa_epitran == 'ゃ':
                    result_dict[vocab] = 'aː'
                elif ipa_epitran == 'ゅ':
                    result_dict[vocab] = 'ɯː'
                else:
                    print('='*10)
                    print('[ERROR] vocab: ', vocab)
                    print('[ERROR] ipa_epitran:', ipa_epitran)
                    print('[ERROR] Skip')
                    print('='*10)
                    continue
            
    with open(dict_path, 'a', encoding='utf-8' ) as f:
        for word, ipa in result_dict.items():
            f.write(f'{word}\t{ipa}\n')     


def convert_to_ipa(input_word, epi):
    try:
        segments = epi.trans_list(input_word)
        word = ""
        for segment in segments:
            if segment == '͈':
                word += segment
            elif segment == '͡':
                word += segment
            elif word and word[-1] == '͡':
                word += segment
            elif segment == "," or segment == "-" or segment == "̀" or segment == "̂" or segment == " ":
                continue
            elif segment == "ÿ":
                word += "y"
            elif segment == "ü":
                word += "u"
            else:
                word = word + ' ' + segment
        return word.strip()
    except IndexError:
        print(f"[WARN] Failed: {input_word}")
        return ''

def create_lab_files(corpus_dir):
    audio_files = Path(corpus_dir).glob("*.wav")
    for audio_file in audio_files:
        transcription = unicodedata.normalize('NFKC', audio_file.stem)
        output_path = Path(corpus_dir) / f'{transcription}.lab'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"[INFO] LAB file saved: {output_path}")


def create_custom_dict(words_path, dict_path, lang):
    ipa_mapping_dict = EPITRAN_TO_MFA[lang]

    with open(words_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if lang == 'art':
            data = data['art']

    results = {}


    word_ipa_pairs = [( item['word'], item['ipa']) for item in data]
    for word, ipa_epitran in word_ipa_pairs:
        ipa_epitran_list = ipa_epitran.split(' ')
        # word level
        ipa_mfa_list = [ipa_mapping_dict[ipa_epitran] for ipa_epitran in ipa_epitran_list]
        results[word] = ' '.join(ipa_mfa_list)
    
    with open(dict_path, 'w', encoding='utf-8') as f: 
        for word, ipa_mfa in results.items():
            f.write(f"{word}\t{ipa_mfa}\n")
      
    print(f"[INFO] Pronunciation dictionary saved: {dict_path}")
