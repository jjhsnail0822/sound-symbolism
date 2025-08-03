import json
from pathlib import Path
import unicodedata
from argparse import ArgumentParser
from mfa_wrapper import MFAWrapper
from mfa_preperation import create_lab_files, create_custom_dict, update_custom_dict
from parse_textgrid import parse_textgrid, textgrid_to_array

ROOT = Path('.')

parser = ArgumentParser()
parser.add_argument("--lang", type=str, required=True, choices=['en', 'ko', 'fr', 'ja', 'art'])
parser.add_argument("--dev",  action='store_true', 
                    help="If `dev` is True, it will create lab files and pronunciation dictionary")
parser.add_argument("--frame_duration_ms", type=int, default=40)

def main(args):
    lang = args.lang
   
    data_dir  = ROOT / 'data' / 'processed'
    if lang == 'art':
        dict_path = data_dir / 'art' / 'resources' / 'ARPA_pronunciation_dict.txt'
        words_path = data_dir / 'art' / 'constructed_words.json' 
        corpus_dir = data_dir / 'art' / 'tts'
        textgrid_dir = data_dir / 'art' / 'textgrids' 
        output_dir = data_dir / 'art' / 'alignment'
        output_path = output_dir /'constructed_words.json' 
    else:
        dict_path = data_dir / 'nat' / 'resources' / f'{lang}_MFA_pronunciation_dict.txt'
        words_path = data_dir / 'nat' / f'{lang}.json'
        corpus_dir = data_dir / 'nat' / 'tts' / lang
        textgrid_dir = data_dir / 'nat' / 'textgrids' / lang
        output_dir = data_dir / 'nat' / 'alignment'
        output_path = output_dir / f'{lang}.json'


    Path(textgrid_dir).mkdir(parents=True, exist_ok=True)
    Path(dict_path.parent).mkdir(parents=True, exist_ok=True)

    if args.dev:
        # 1. Create Corpus - (.wav, .lab) pairs
        create_lab_files(corpus_dir=corpus_dir)

        # 2. Create Dictionary
        create_custom_dict(words_path, dict_path, lang=lang)
    
    # 3. Montreal Forced Alignment
    mfa = MFAWrapper(lang, dict_path=dict_path)
    mfa.run(
            corpus_dir=corpus_dir, 
            textgrid_dir=textgrid_dir
            )

    if args.dev:
        # 4. Parse Textgrid Files
        oov = set()
        textgrid_files = sorted(textgrid_dir.glob("*.TextGrid"))
        for textgrid_path in textgrid_files:
            data = parse_textgrid(textgrid_path=textgrid_path)

            # 5. Find OOV set
            phones = [interval['text'] for interval in data['phones']]
            if 'spn' in phones:
                oov.update(set([interval['text'] for interval in data['words']]))
        print(f'[INFO] Initial OOV # {len(oov)}')
        # print(f'[DEBUG] OOV: ', oov)

        # 5. update custom dict
        if oov:
            print(f'[INFO] Update custom dict')
            update_custom_dict(oov, dict_path, lang=lang)

            mfa.run(
                corpus_dir=corpus_dir, 
                textgrid_dir=textgrid_dir
                )
            
    # 6. Parse Textgrid and Convert to Array
    results =  [] 
    oov = []
    textgrid_files = sorted(textgrid_dir.glob("*.TextGrid"))
    for textgrid_path in textgrid_files:
        word = unicodedata.normalize('NFKC', str(textgrid_path.stem))
        data = parse_textgrid(textgrid_path=textgrid_path)
        phones = textgrid_to_array(xmax = data['xmax'], 
                                   phones_interval=data['phones'],
                                   frame_duration_ms=args.frame_duration_ms)

        if 'spn' in phones:
            oov.append(word)

        result_dict = {
            'word':  word,
            'sampling_rate': 16000,
            'frame_duration_ms': args.frame_duration_ms,
            'phones': phones,
            'mfa_raw': data
        }
        results.append(result_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("[INFO] File saved to: ", output_path)
    print(f"[INFO] Final OOV(Out of Vocabulary) #{len(oov)}: {oov}")

if __name__ == '__main__':
    args = parser.parse_args()
    for lang in ['en', 'ko', 'fr', 'ja', 'art']:
        args.lang = lang
        main(args)