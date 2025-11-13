import epitran
import json
from ko_pron import romanise
import pykakasi
from tqdm import tqdm

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

def convert_to_romanization(input_word, lang, kks):
    if lang == 'en':
        return input_word
    elif lang == 'fr':
        return input_word
    elif lang == 'ja':
        return kks.convert(input_word)[0]['hepburn']
    elif lang == 'ko':
        return romanise(input_word, 'mr')

if __name__ == "__main__":
    kks = pykakasi.Kakasi()
    epi = {'en': epitran.Epitran('eng-Latn'), 'fr': epitran.Epitran('fra-Latn'), 'ja': epitran.Epitran('jpn-Hrgn'), 'ko': epitran.Epitran('kor-Hang')}
    lang = ['en', 'fr', 'ja', 'ko']
    data = {}
    for l in lang:
        with open(f'data/processed/nat/{l}.json', 'r', encoding='utf-8') as f:
            data[l] = json.load(f)
            for d in tqdm(data[l]):
                d['ipa'] = convert_to_ipa(d['word'], epi[l])
                d['ipa_source'] = 'epitran'
                d['romanization'] = convert_to_romanization(d['word'], l, kks)
                if d['ipa'] == '':
                    print(f"IPA conversion failed for {d['word']} in {l}")

        print('IPA conversion completed.')
        
        with open(f'data/processed/nat/{l}_ipa.json', 'w', encoding='utf-8') as f:
            json.dump(data[l], f, ensure_ascii=False, indent=4)

        print('IPA conversion saved.')