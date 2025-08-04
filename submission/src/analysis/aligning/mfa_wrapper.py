import time
import subprocess


class MFAWrapper():
    '''
    Wraps the CLI tool MFA to execute commands from Python
    '''
    LANGUAGE_MODEL_MAP = {
            'en': 'english_us_arpa',
            'fr': 'french_mfa',
            'ja': 'japanese_mfa',
            'ko': 'korean_mfa'
        }
    def __init__(self, lang, dict_path=None, ensure_installed=False, ensure_downloaded=False ):
        self.model_name = self.set_model_name(lang)
        self.dict_path = dict_path

        if ensure_installed:
            if not self.check_mfa_installation():
                raise ModuleNotFoundError("MFA is not installed. Run `conda install -c conda-forge montreal-forced-aligner`")      
            
        if ensure_downloaded:
            if not self.check_model_downloaded('acoustic'):
                self.download_model('acoustic') 

            if not self.check_model_downloaded('dictionary'):
                if dict_path is None:
                    self.download_model('dictionary')
          


    def run(self, corpus_dir, textgrid_dir, validate=False, clean=True):
        if validate:
            print("[INFO] Validate Corpus")
            self.validate_corpus(corpus_dir)
            print("[INFO] Finished")

        print("[INFO] Start Aligning")
        tic = time.time()
        if self.dict_path is None:
            result = self.align_with_pretrained_model(corpus_dir, textgrid_dir, clean)
        else:
            result = self.align_with_custom_dict(corpus_dir, self.dict_path, textgrid_dir, clean)

        if result is None:
            print("[ERROR] Alignment Failed")
            return False
        
        toc = time.time()
        processing_time = toc - tic
        print(f"[INFO] Finished: {processing_time}s ")
        return True
    
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
        return self.LANGUAGE_MODEL_MAP['en'] if lang == 'art' else self.LANGUAGE_MODEL_MAP[lang]

    def check_model_downloaded(self, model_type):
        """Check if the model is downloaded"""
        try:
            # Actually, should use the list command
            command = ['mfa', 'model', 'list', model_type]
            result = self.run_mfa_command(command)
            
            if result is not None and result.returncode == 0:
                # Check if model_name is in stdout
                return self.model_name in result.stdout
            return False
        except Exception as e:
            print(f"Error checking model: {e}")
            return False
 
    def download_model(self, model_type:str):
        command = ['mfa', 'model', 'download', model_type, self.model_name]   
        self.run_mfa_command(command)

    def validate_corpus(self, corpus_dir):
        command = ['mfa', 'validate', corpus_dir, self.model_name, self.model_name ]
        self.run_mfa_command(command)

    def validate_dictionary(self, dict_path):
        command = ['mfa', 'validate_dictionary', dict_path]
        self.run_mfa_command(command)

    def align_with_custom_dict(self, corpus_dir, dict_path, textgrid_dir, clean):
        command = ['mfa', 'align', corpus_dir, dict_path, self.model_name, textgrid_dir ]
        if clean:
            command.append('--clean')
        
        return self.run_mfa_command(command)
    
    
    def align_with_pretrained_model(self, corpus_dir, textgrid_dir, clean):
        '''
        Align with a pretrained model, no need for 'word-ipa' dictionary
        '''
        if self.model_name is None:
            print('call `set_model_name(lang)` method first.')
            return 
        
        command = ['mfa', 'align', corpus_dir, self.model_name, self.model_name, str(textgrid_dir)]
         
        if clean:
            command.append('--clean')

        return self.run_mfa_command(command)
    
    def run_mfa_command(self, command):
        try:
            command = list(map(str, command))
            print(f'[INFO] Running... `{" ".join(command)}`')
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print('[INFO] Finished')
            if result.stdout:
                print('[INFO]', result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f'[Error] {e.stderr}')
            return None
        except Exception as e:
            print('ERROR', command)
            return None
