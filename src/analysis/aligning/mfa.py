import json
import time
import subprocess
import unicodedata
from pathlib import Path

class MFAWrapRunner():
    '''
    CLI 도구인 MFA를 Python에서 Wrapping하여 명령어 실행 
    '''
    def __init__(self, lang, dict_path=None ):
        self.model_name = self.set_model_name(lang)
        self.dict_path = dict_path
        
        self.check_mfa_installation()
        if not self.check_mfa_installation():
            raise ModuleNotFoundError("MFA is not installed. Run `conda install -c conda-forge montreal-forced-aligner`")      
        
        # self.validate_dictionary(dict_path)

        if not self.check_model_downloaded('acoustic'):
            self.download_model('acoustic') 

        if not self.check_model_downloaded('dictionary'):
            if dict_path is None:
                self.download_model('dictionary')
          


    def run(self, corpus_dir, output_dir, validate=False, clean=True):
        if validate:
            print("[INFO] Validate Corpus")
            self.validate_corpus(corpus_dir)
            print("[INFO] Finished")

        print("[INFO] Start Aligning")
        tic = time.time()
        if self.dict_path is None:
            self.align_with_pretrained_model(corpus_dir, output_dir, clean)
        else:
            self.align_with_custom_dict(corpus_dir, self.dict_path, output_dir, clean)
        toc = time.time()
        processed_time = toc - tic
        print(f"[INFO] Finished: {processed_time}s ")

    
    def check_mfa_installation(self):
        try:
            result = subprocess.run(['mfa', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f'MFA is already installed: {result.stdout.strip()}')
                return True
            else:
                print("MFA is not installed")
                return False
        except FileNotFoundError:
            return False

    def set_model_name(self, lang):
        if lang not in ['art', 'en', 'ko', 'ja', 'fr']:
            print(f'[ERROR] Language {lang} is not supported')
            raise ValueError
        
        model_map = {
            'en': 'english_us_arpa',
            'fr': 'french_mfa',
            'ja': 'japanese_mfa',
            'ko': 'korean_mfa'
        }
        return model_map['en'] if lang == 'art' else model_map[lang]

    def check_model_downloaded(self, model_type ):
        try:
            cmd = ['mfa', 'model', 'download', model_type, self.model_name]
            result = self.run_mfa_command(cmd)

            if result.returncode == 0:
                print(f'[INFO] Model is already downloaded')
                return True
            else:
                print(f'[INFO] Model is not downladed yet')
                return False
        except Exception as e:
            print(e)
            raise e

 
    def download_model(self, model_type:str):
        cmd = ['mfa', 'model', 'download', model_type, self.model_name]   
        self.run_mfa_command(cmd)

    def validate_corpus(self, corpus_dir):
        cmd = ['mfa', 'validate', corpus_dir, self.model_name, self.model_name ]
        self.run_mfa_command(cmd)

    def validate_dictionary(self, dict_path):
        cmd = ['mfa', 'validate_dictionary', dict_path]
        self.run_mfa_command(cmd)

    def align_with_custom_dict(self, corpus_dir, dict_path, output_dir, clean):
        cmd = ['mfa', 'align', corpus_dir, dict_path, self.model_name, output_dir ]
        if clean:
            cmd.append('--clean')
        
        return self.run_mfa_command(cmd)
    
    
    def align_with_pretrained_model(self, corpus_dir, output_dir, clean):
        '''
        사전 훈련된 모델로 정렬하여 'word-ipa' 딕셔너리 필요 X
        '''
        if self.model_name is None:
            print('call `set_model_name(lang)` method first.')
            return 
        
        cmd = ['mfa', 'align', corpus_dir, self.model_name, self.model_name, str(output_dir)]
         
        if clean:
            cmd.append('--clean')

        return self.run_mfa_command(cmd)
    
    def run_mfa_command(self, cmd):
        try:
            cmd = list(map(str, cmd))
            print(f'[INFO] Running... `{" ".join(cmd)}`')
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print('[INFO] Finished')
            if result.stdout:
                print('[INFO]', result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f'[Error] {e.stderr}')
            return -1
        except Exception as e:
            print('ERROR', cmd)
   
    def list_models(self, model_type):
        cmd = ['mfa', 'model', 'download', model_type, '--list']
        return self.run_mfa_command(cmd)


def create_lab_files(corpus_dir):
    audio_files = Path(corpus_dir).glob("*.wav")
    for audio_file in audio_files:
        transcription = unicodedata.normalize('NFKC', audio_file.stem)
        output_path = Path(corpus_dir) / f'{transcription}.lab'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"[INFO] LAB file saved: {output_path}")



def create_custom_dict(words_path, dict_path):
    ipa_to_arpa = {
    "l": "L",
    "m": "M",
    "n": "N",
    "v": "V",
    "ð": "DH",
    "z": "Z",
    "f": "F",
    "s": "S",
    "ʃ": "SH",
    "b": "B",
    "d": "D",
    "ɡ": "G",
    "p": "P",
    "t": "T",
    "k": "K",
    "i": "IY0",
    "ej": "EY0",
    "ɑ": "AA0",
    "ow": "OW0"
    }    
   
    with open(words_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    word_ipa_pairs = [( item['word'], item['ipa']) for item in data['art']]
    
    for word, ipa in word_ipa_pairs:
        arpa = [ipa_to_arpa[symbol] for symbol in ipa.split(' ')]
        with open(dict_path,'a', encoding='utf-8' ) as f:
            f.write(f"{word}\t{' '.join(arpa)}\n")
    print(f"[INFO] Pronounciation dictionary saved: {dict_path}")

            


def main():

    lang = 'ko'
    root = Path('/home/sunahan/workspace/sound-symbolism')
    data_dir  = root / 'data' / 'processed'
    if lang == 'art':
        dict_path = data_dir / 'art' / 'resources' / 'ARPA_pronunciation_dict.txt'
        words_path = data_dir / 'art' / 'constructed_words.json' 
        corpus_dir = data_dir / 'art' / 'tts'
        output_dir = data_dir / 'art' / 'textgrids' 
    else:
        dict_path = None
        words_path = data_dir / 'nat' / f'{lang}.json'
        corpus_dir = data_dir / 'nat' / 'tts' / lang
        output_dir = data_dir / 'nat' / 'textgrids' / lang


    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Create Corpus - (.wav, .lab) pairs
    create_lab_files(corpus_dir=corpus_dir)

    # 2. Create Dictionary
    if lang == 'art':
        create_custom_dict(words_path, dict_path)
    
    mfa = MFAWrapRunner(lang, dict_path=dict_path)
    mfa.run(
            corpus_dir=corpus_dir, 
            output_dir=output_dir
            )




if __name__ == '__main__':
    main()