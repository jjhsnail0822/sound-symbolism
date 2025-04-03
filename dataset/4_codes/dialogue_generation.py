# Example: python dialogue_generation.py -l ko -b 1 -s 1
import os
import json
import argparse
import time
from pprint import pprint
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import random
import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
import concurrent.futures

# Load environment variables from .env.local file
env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DialogueGenerator:
    def __init__(self, language: str, api_key: Optional[str] = None, model: str = "chatgpt-4o-latest", batch_size: int = 16, dialogues_per_word: int = 5, step_by_step: bool = False):
        """
        Initialize the dialogue generator
        
        Args:
            language (str): Language code (en/fr/ko/ja)
            api_key (str, optional): OpenAI API key. If None, will try to get from environment variable
            model (str): OpenAI model to use
            batch_size (int): Number of dialogues to generate in a single batch
            dialogues_per_word (int): Number of dialogues to generate for each word
        """
        self.language = language.lower()
        if self.language not in ['en', 'fr', 'ko', 'ja']:
            raise ValueError("Language must be one of 'en', 'fr', 'ko', 'ja'.")
        
        # Set batch size and dialogues per word
        self.batch_size = batch_size
        self.dialogues_per_word = dialogues_per_word
        self.step_by_step = step_by_step
        
        # Set paths
        self.input_path = os.path.join('../1_preprocess/nat', f"{self.language}.json")
        # /scratch2/sheepswool/workspace/sound-symbolism/dataset/1_preprocess/nat/ko.json
        self.output_path = '../2_dialogue/nat'
        self.output_file = os.path.join(self.output_path, f"{self.language}.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set OpenAI API key and client
        if api_key:
            self.api_key = api_key
        elif OPENAI_API_KEY:
            self.api_key = OPENAI_API_KEY
        else:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI clients (both sync and async)
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Set model
        self.model = model
        
        # Load language-specific templates
        self.templates = self._get_language_templates()
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter data to only include entries with meanings
            filtered_data = [item for item in data if item.get('meaning') and len(item.get('meaning', [])) > 0 and item.get('found', False)]
            
            print(f"Loaded {len(filtered_data)} entries with meanings out of {len(data)} total entries")
            return filtered_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def _get_language_templates(self) -> Dict[str, Dict[str, str]]:
        """Get language-specific templates for prompts"""
        templates = {
            'en': {
                "system_prompt": "You are an assistant that generates a natural conversation between two people. The dialog should be naturally infused with onomatopoeia/mimetic word '{word}' that conveys the meaning of '{meaning}'. Make sure the whole dialog is contextualized and has a consistent flow.",
                "user_prompt_set1": "Condition 1: Generate a dialog between two people that naturally contains the onomatopoeia/mimetic word '{word}' that conveys the meaning of '{meaning}'.\nCondition 2: The dialog consists of a total of {num_turns} turns ({num_utterances} utterances).\nFirst try to envision the overall flow and structure of the dialog, and then organize your answer step by step.",
                "user_prompt_set2": "Condition 3: '{word}' must be contained within the {ss_idx}th utterance; the preceding and following utterances can be adjusted as needed to keep the utterance within its natural context.\nCondition 4: Make sure that the meaning of the onomatopoeia/mimetic word never appears in any other part of the conversation, and that it fits into the context.\nThink through each step to deduce at your answer.",
                "user_prompt_set3": "Condition 5: Choose names for your characters that are actually used in their language group. There must be a total of two characters.\nCondition 6: Instead of using typical greetings or farewells (e.g., “hello” or “goodbye”), construct sentences using expressions that reflect the context and personality of the conversation.\nThink logically through each step and organize your answer.",
                "dialogue_format": "Please format your response as a JSON array like this:\n```json[\n    {\"Speaker\": \"Person1\", \"Utterance Number\": 1, \"Utterance\": \"Content\"},\n    {\"Speaker\": \"Person2\", \"Utterance Number\": 2, \"Utterance\": \"Content\"},\n    {\"Speaker\": \"Person1\", \"Utterance Number\": 3, \"Utterance\": \"Content\"},\n    {\"Speaker\": \"Person2\", \"Utterance Number\": 4, \"Utterance\": \"Content\"},\n    {\"Speaker\": \"Person1\", \"Utterance Number\": 5, \"Utterance\": \"Content\"},\n    {\"Speaker\": \"Person2\", \"Utterance Number\": 6, \"Utterance\": \"Content\"}\n]```"
            },
            "fr": {
                "system_prompt": "Vous êtes un assistant qui crée un dialogue naturel entre deux personnes. Le dialogue doit être naturellement imprégné d'onomatopées/idéophones « {word} » qui véhiculent le sens de « {meaning} ». Veillez à ce que l'ensemble du dialogue soit contextuel et s'enchaîne de manière cohérente.",
                "user_prompt_set1": "Condition 1 : Générer un dialogue entre deux personnes qui contient naturellement l'onomatopée/idéophones « {word} » qui transmet le sens de « {meaning} ».\nCondition 2 : Le dialogue consiste en un total de {num_turns} tours ({num_utterances} énoncés).\nTentez d'abord d'imaginer le flux et la structure du dialogue global, puis organisez votre réponse étape par étape.",
                "user_prompt_set2": "Condition 3 : « {word} » doit être contenu dans le {ss_idx}ème énoncé ; les énoncés précédents et suivants peuvent être ajustés si nécessaire pour garantir que l'énoncé se trouve dans son contexte naturel.\nCondition 4 : Veillez à ne pas utiliser l'onomatopée/idéophone dans une autre partie du dialogue, et essayez de l'intégrer dans le contexte.\nRéfléchissez à chaque étape pour parvenir à votre réponse.",
                "user_prompt_set3": "Condition 5 : Choisissez pour vos personnages des noms qui sont réellement utilisés dans leur zone linguistique. Il doit y avoir deux personnages au total.\nCondition 6 : Au lieu d'utiliser des salutations ou des adieux typiques (par exemple, « bonjour », « au revoir »), construisez des phrases en utilisant des expressions qui reflètent le contexte et la personnalité du dialogue.\nRéfléchissez logiquement à chaque étape et structurez votre réponse.",
                "dialogue_format": "Veuillez formater votre réponse sous la forme d'un tableau JSON comme suit:\n```json[\n    {\"Locuteur\": \"Personne1\", \"Numéro de réplique\": 1, \"Réplique\": \"Contenu\"},\n    {\"Locuteur\": \"Personne2\", \"Numéro de réplique\": 2, \"Réplique\": \"Contenu\"},\n    {\"Locuteur\": \"Personne1\", \"Numéro de réplique\": 3, \"Réplique\": \"Contenu\"},\n    {\"Locuteur\": \"Personne2\", \"Numéro de réplique\": 4, \"Réplique\": \"Contenu\"},\n    {\"Locuteur\": \"Personne1\", \"Numéro de réplique\": 5, \"Réplique\": \"Contenu\"},\n    {\"Locuteur\": \"Personne2\", \"Numéro de réplique\": 6, \"Réplique\": \"Contenu\"}\n]```"
            },
            "ko": {
                "system_prompt": "당신은 두 사람의 대화를 자연스럽게 생성하는 도우미입니다. 대화에는 반드시 '{meaning}'의 의미를 전달하는 의성어/의태어 '{word}'가 자연스레 녹아들어야 합니다. 전체 대화가 문맥에 맞고 일관된 흐름을 유지하도록 해주세요.",
                "user_prompt_set1": "조건 1: '{meaning}'의 의미를 전달하는 의성어/의태어 '{word}'가 자연스럽게 포함된 두 사람 간의 대화를 생성합니다.\n조건 2: 대화는 총 {num_turns}턴(발화 {num_utterances}회)으로 구성됩니다.\n먼저 전체적인 대화의 흐름과 구조를 구상해보고, 단계별로 답안을 구성해 주세요.",
                "user_prompt_set2": "조건 3: '{word}'는 반드시 {ss_idx}번째 발화 내에 포함되어야 합니다. 해당 발화가 자연스러운 문맥 내에서 이루어지도록 앞뒤 내용도 필요시 조정할 수 있습니다.\n조건 4: 대화의 다른 부분에서는 의성어/의태어의 의미가 절대 등장하지 않도록 주의하며, 문맥에 녹아들도록 표현해 주세요.\n각 단계별로 생각하며 답안을 도출해 주세요.",
                "user_prompt_set3": "조건 5: 등장 인물의 이름은 해당 언어권에서 실제로 사용되는 이름으로 선택합니다. 인물은 총 2명이어야 합니다.\n조건 6: 전형적인 인사말이나 작별 인사(예: '안녕', '잘 가') 대신 대화의 맥락과 개성을 반영한 표현을 사용하여 문장을 구성해 주세요.\n각 단계별로 논리적으로 생각하고 답안을 구성해 주세요.",
                "dialogue_format": "응답을 다음과 같은 JSON 배열 형식으로 작성해주세요:\n```json[\n    {\"화자\": \"사람1\", \"발화 번호\": 1, \"발화\": \"내용\"},\n    {\"화자\": \"사람2\", \"발화 번호\": 2, \"발화\": \"내용\"},\n    {\"화자\": \"사람1\", \"발화 번호\": 3, \"발화\": \"내용\"},\n    {\"화자\": \"사람2\", \"발화 번호\": 4, \"발화\": \"내용\"},\n    {\"화자\": \"사람1\", \"발화 번호\": 5, \"발화\": \"내용\"},\n    {\"화자\": \"사람2\", \"발화 번호\": 6, \"발화\": \"내용\"}\n]```"
            },
            'ja': {
                "system_prompt": "あなたは二人の自然な会話を作成するアシスタントです。会話には、必ず '{meaning}' を表す擬音語・擬態語 '{word}' を自然な形で含める必要があります。全体の会話が文脈に適しており、一貫性のある流れになるようにしてください。",
                "user_prompt_set1": "条件1: '{meaning}' を意味する擬音語・擬態語 '{word}' を自然に含む二人の会話を作成してください。\n条件2: 会話は合計 {num_turns}ターン（{num_utterances}回の発話）で構成してください。\nまず、全体の会話の流れや構成を考え、段階的に回答を作成してください。",
                "user_prompt_set2": "条件3: '{word}' は必ず {ss_idx} 番目の発話に含めてください。必要に応じて前後の文脈を調整し、自然に馴染むようにしてください。\n条件4: ほかの発話では、擬音語・擬態語の意味を直接的に強調しすぎず、文脈に溶け込むようにしてください。\n段階的に考えながら回答を作成してください。",
                "user_prompt_set3": "条件5: 登場人物の名前は、その言語圏で実際に使われるものにしてください。登場人物は二人です。\n条件6: 一般的な挨拶や別れの言葉（例：「こんにちは」「さようなら」）は使わず、会話の流れやキャラクターに適した表現を使用してください。\n論理的に考え、段階的に回答を作成してください。",
                "dialogue_format": "回答を次のようなJSON配列の形式で作成してください:\n```json[\n    {\"話者\": \"人物1\", \"発話番号\": 1, \"発話\": \"発話内容\"},\n    {\"話者\": \"人物2\", \"発話番号\": 2, \"発話\": \"発話内容\"},\n    {\"話者\": \"人物1\", \"発話番号\": 3, \"発話\": \"発話内容\"},\n    {\"話者\": \"人物2\", \"発話番号\": 4, \"発話\": \"発話内容\"},\n    {\"話者\": \"人物1\", \"発話番号\": 5, \"発話\": \"発話内容\"},\n    {\"話者\": \"人物2\", \"発話番号\": 6, \"発話\": \"発話内容\"}\n]```"
            }
        }
        return templates
    
    async def _generate_dialogue_async(self, word_data: Dict[str, Any], dialogue_num: int = 1) -> Dict[str, Any]:
        """Generate dialogue for a given word asynchronously"""
        word = word_data.get('word', '')
        
        # Get the first meaning if it's a list, or use the meaning directly
        meaning = word_data.get('meaning', [])
        if isinstance(meaning, list) and meaning:
            meaning = meaning[0]
        elif not meaning:
            meaning = "an onomatopoeic word"
        
        num_utterances = 6
        ss_idx = random.randint(1, num_utterances)
        num_turns = num_utterances // 2
        
        # Format system prompt
        system_prompt = self.templates[self.language]['system_prompt'].format(word=word, meaning=meaning)
        
        try:
            # STEP 1: Generate initial dialogue with first set of conditions
            user_prompt_set1 = self.templates[self.language]['user_prompt_set1'].format(
                word=word, 
                meaning=meaning, 
                num_utterances=num_utterances, 
                num_turns=num_turns
            )
            
            # STEP 2: Apply second set of conditions to the initial dialogue
            user_prompt_set2 = self.templates[self.language]['user_prompt_set2'].format(
                word=word,
                ss_idx=ss_idx
            )

            # STEP 3: Apply final set of conditions
            user_prompt_set3 = self.templates[self.language]['user_prompt_set3']
                       
            if self.step_by_step:
                
                response1 = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_set1},
                        {"role": "user", "content": self.templates[self.language]['dialogue_format']}
                    ],
                    temperature=1,
                    max_tokens=1000
                )

                dialogue1 = response1.choices[0].message.content.strip()
                
                response2 = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_set1},
                        {"role": "assistant", "content": dialogue1},
                        {"role": "user", "content": user_prompt_set2},
                        {"role": "user", "content": self.templates[self.language]['dialogue_format']}
                    ],
                    temperature=1,
                    max_tokens=1000
                )
                
                dialogue2 = response2.choices[0].message.content.strip()
                
                response3 = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_set1},
                        {"role": "user", "content": user_prompt_set2},
                        {"role": "assistant", "content": dialogue2},
                        {"role": "user", "content": user_prompt_set3},
                        {"role": "user", "content": self.templates[self.language]['dialogue_format']}
                    ],
                    temperature=1,
                    max_tokens=1000
                )
            
            else:
                response3 = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_set1},
                        {"role": "user", "content": user_prompt_set2},
                        {"role": "user", "content": user_prompt_set3},
                        {"role": "user", "content": self.templates[self.language]['dialogue_format']}
                    ],
                    temperature=1,
                    max_tokens=1000
                )
            
            final_dialogue = response3.choices[0].message.content.strip()
            
            # Parse the final dialogue
            parsed_dialogue = self._parse_dialogue(final_dialogue)
            
            # Check for JSON code blocks and extract content if needed
            match = re.search(r"```json.*?\n(\[.*?\])", final_dialogue, re.DOTALL)
            if match:
                json_content = match.group(1)
                # Replace keys to standardized English format
                json_content = self._replace_keys(json_content)
                try:
                    parsed_dialogue = json.loads(json_content)
                except json.JSONDecodeError:
                    # If parsing fails, keep the original parsed dialogue
                    pass
            
            # Verify that the word appears in the correct utterance
            retry_count = 0 
            max_retries = 3
            
            # breakpoint()
            
            while retry_count < max_retries:
                # Check if the word is in the correct utterance and not in others
                word_in_correct_place = False
                word_in_wrong_place = False
                correct_utterance_num = False
                
                for i, utterance in enumerate(parsed_dialogue):
                    utterance_text = utterance.get("text", "") if isinstance(utterance, dict) else ""
                    if i == ss_idx - 1:
                        if word in utterance_text:
                            word_in_correct_place = True
                    else:
                        if word in utterance_text:
                            word_in_wrong_place = True
                    if len(parsed_dialogue) == num_utterances:
                        correct_utterance_num = True
                        
                
                if word_in_correct_place and not word_in_wrong_place and correct_utterance_num:
                    break
                
                # If validation fails, retry with more explicit instructions
                retry_count += 1
                
                # Create language-specific fix prompts
                if self.language == 'en':
                    fix_prompt = f"""The dialogue needs to be fixed. The word '{word}' must ONLY appear in the {ss_idx}th utterance, not in any other utterance.
                        Current dialogue: {parsed_dialogue}
                        Please rewrite the dialogue to fix this issue. Make sure the word '{word}' appears only in utterance #{ss_idx}.
                        Please format your response as a JSON array like this:
                        {self.templates[self.language]['dialogue_format'].split("\n", 1)[1]}"""
                elif self.language == 'fr':
                    fix_prompt = f"""Le dialogue doit être corrigé. Le mot « {word} » doit apparaître UNIQUEMENT dans la {ss_idx}ème réplique, et dans aucune autre.
                        Dialogue actuel : {parsed_dialogue}
                        Veuillez réécrire le dialogue pour corriger ce problème. Assurez-vous que le mot « {word} » n'apparaît que dans la réplique n°{ss_idx}.
                        Veuillez formater votre réponse comme un tableau JSON comme ceci :
                        {self.templates[self.language]['dialogue_format'].split("\n", 1)[1]}"""
                elif self.language == 'ko':
                    fix_prompt = f"""대화를 수정해야 합니다. '{word}' 단어는 반드시 {ss_idx}번째 발화에만 나타나야 하며, 다른 발화에는 나타나면 안 됩니다.
                        현재 대화: {parsed_dialogue}
                        이 문제를 해결하기 위해 대화를 다시 작성해 주세요. '{word}' 단어가 {ss_idx}번째 발화에만 나타나도록 해주세요.
                        응답을 다음과 같은 JSON 배열 형식으로 작성해주세요:
                        {self.templates[self.language]['dialogue_format'].split("\n", 1)[1]}"""
                elif self.language == 'ja':
                    fix_prompt = f"""対話を修正する必要があります。単語「{word}」は{ss_idx}番目の発話にのみ出現し、他の発話には出現してはいけません。
                        現在の対話: {parsed_dialogue}
                        この問題を修正するために対話を書き直してください。「{word}」が{ss_idx}番目の発話にのみ出現するようにしてください。
                        回答を次のようなJSON配列の形式で作成してください:
                        {self.templates[self.language]['dialogue_format'].split("\n", 1)[1]}"""
                else:
                    raise "Unsupported language code used."
                
                fix_response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": fix_prompt}
                    ],
                    temperature=1,
                    max_tokens=1000
                )
                
                final_dialogue = fix_response.choices[0].message.content.strip()
                parsed_dialogue = self._parse_dialogue(final_dialogue)
                
                if retry_count >= max_retries:
                    print(f"Warning: Could not place '{word}' correctly after {max_retries} attempts.")
            
            # Create result dictionary with dialogue_num
            result = {
                'dialogue': parsed_dialogue,
                'meta_data': {
                    'dialogue_num': dialogue_num,
                    'language': self.language,
                    'num_utterances': num_utterances,
                    'ss_idx': ss_idx,
                    'success': True,
                },
            }
            
            return result
        except Exception as e:
            print(f"Error generating dialogue {dialogue_num} for word '{word}': {e}")
            # Return empty dialogue on error
            return {
                'dialogue': [],
                'meta_data': {
                    'dialogue_num': dialogue_num,
                    'language': self.language,
                    'num_utterances': num_utterances,
                    'ss_idx': ss_idx,
                    'success': False,
                },
            }
    
    async def _generate_multiple_dialogues_async(self, word_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple dialogues for a given word asynchronously"""
        word = word_data.get('word', '')
        meaning = word_data.get('meaning', [])
        
        # Generate multiple dialogues for the word
        dialogue_tasks = [self._generate_dialogue_async(word_data, i+1) for i in range(self.dialogues_per_word)]
        dialogue_results = await asyncio.gather(*dialogue_tasks)
        
        # Create result with multiple dialogues
        result = {
            'word': word,
            'meaning': meaning,
            'dialogues': dialogue_results
        }
        
        return result
    
    async def _process_batch_async(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of words asynchronously"""
        tasks = [self._generate_multiple_dialogues_async(item) for item in batch]
        return await asyncio.gather(*tasks)
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of words using asyncio event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._process_batch_async(batch))
        finally:
            loop.close()
    
    def generate_all_dialogues(self, limit: Optional[int] = None, shuffle: bool = True) -> List[Dict[str, Any]]:
        """Generate dialogues for all words in the data using batch processing"""
        if not self.data:
            print("No data available to generate dialogues")
            return []
        
        # Optionally shuffle the data
        data_to_process = self.data.copy()
        if shuffle:
            random.shuffle(data_to_process)
        
        # Limit the number of items to process if specified
        if limit and limit > 0:
            data_to_process = data_to_process[:limit]
        
        results = []
        
        temp_stop_idx = 7
        
        # Process data in batches
        batches = [data_to_process[i:i + self.batch_size] for i in range(0, len(data_to_process), self.batch_size)]
        
        for batch_idx, batch in enumerate(tqdm(batches, desc=f"Processing batches ({self.language})")):
            # Process the batch
            batch_results = self._process_batch(batch)
            
            # Add batch results to overall results
            results.extend(batch_results)
            
            # Save intermediate results after each batch
            self._save_intermediate_results(results)
            
            # Add a delay between batches to avoid rate limiting
            if batch_idx < len(batches) - 1:
                time.sleep(2)
            
            if batch_idx == temp_stop_idx:
                break
        
        return results
    
    def _clean_speaker_name(self, speaker: str) -> str:
        """Clean speaker name by removing special characters and keeping only valid characters for the language"""
        if not speaker:
            return ""
        
        # Remove any JSON formatting artifacts
        speaker = speaker.replace('"', '').replace('{', '').replace('}', '')
        
        # Language-specific cleaning
        if self.language == 'ko':
            # Keep Korean characters, numbers, and English letters
            return re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', '', speaker).strip()
        elif self.language == 'ja':
            # Keep Japanese characters, numbers, and English letters
            return re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAFa-zA-Z0-9\s]', '', speaker).strip()
        elif self.language == 'fr':
            # Keep French characters with accents, numbers, and basic letters
            return re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', speaker).strip()
        else:  # en
            # Keep English letters and numbers
            return re.sub(r'[^a-zA-Z0-9\s]', '', speaker).strip()
    
    def _replace_keys(self, dialogue_text: str) -> str:
        """
        Replace language-specific keys with standardized English keys
        
        Args:
            dialogue_text (str): The dialogue text with language-specific keys
            
        Returns:
            str: The dialogue text with standardized English keys
        """
        if self.language == 'ko':
            # Korean key replacements
            dialogue_text = dialogue_text.replace('"화자"', '"speaker"')
            dialogue_text = dialogue_text.replace('"발화 번호"', '"index"')
            dialogue_text = dialogue_text.replace('"발화"', '"text"')
        elif self.language == 'ja':
            # Japanese key replacements
            dialogue_text = dialogue_text.replace('"話者"', '"speaker"')
            dialogue_text = dialogue_text.replace('"発話番号"', '"index"')
            dialogue_text = dialogue_text.replace('"発話"', '"text"')
        elif self.language == 'fr':
            # French key replacements
            dialogue_text = dialogue_text.replace('"Locuteur"', '"speaker"')
            dialogue_text = dialogue_text.replace('"Numéro de réplique"', '"index"')
            dialogue_text = dialogue_text.replace('"Réplique"', '"text"')
        elif self.language == 'en':
            # English key replacements (might have different formats)
            dialogue_text = dialogue_text.replace('"Speaker"', '"speaker"')
            dialogue_text = dialogue_text.replace('"Utterance Number"', '"index"')
            dialogue_text = dialogue_text.replace('"Utterance"', '"text"')
            
        # Additional replacements for other possible key formats
        dialogue_text = dialogue_text.replace('"utterance_number"', '"index"')
        dialogue_text = dialogue_text.replace('"speaker_name"', '"speaker"')
        dialogue_text = dialogue_text.replace('"utterance_text"', '"text"')
        
        return dialogue_text
    
    def _parse_dialogue(self, dialogue_text: str) -> List[Dict[str, str]]:
        """Parse dialogue text into a structured list of utterances"""
        result = []
        
        # Clean up the dialogue text
        dialogue_text = dialogue_text.strip()
        
        # Remove any markdown code block formatting and #### headers
        dialogue_text = re.sub(r'```(?:json)?\s*|\s*```', '', dialogue_text)
        
        # Standardize keys to English
        dialogue_text = self._replace_keys(dialogue_text)
        
        # Standardize keys to English
        dialogue_text = self._replace_keys(dialogue_text)
        
        # Try to parse as JSON if it looks like a JSON array
        if dialogue_text.startswith('[') and dialogue_text.endswith(']'):
            try:
                # Try to parse as JSON
                parsed_list = json.loads(dialogue_text)
                if isinstance(parsed_list, list):
                    for item in parsed_list:
                        if isinstance(item, dict):
                            # Keys should now be standardized to English
                            speaker = item.get('speaker', '')
                            # Clean speaker name
                            speaker = self._clean_speaker_name(speaker)
                            
                            # Get index and text with standardized keys
                            utterance_num = item.get('index', 0)
                            text = item.get('text', '')
                            
                            result.append({
                                "speaker": speaker,
                                "index": utterance_num,
                                "text": text
                            })
                    return result
            except json.JSONDecodeError:
                # If JSON parsing fails, continue to other methods
                pass
        
        # If JSON parsing failed, try to parse using ast.literal_eval
        try:
            parsed_list = ast.literal_eval(dialogue_text)
            if isinstance(parsed_list, list):
                for item in parsed_list:
                    if isinstance(item, dict):
                        # Keys should now be standardized to English
                        speaker = item.get('speaker', '')
                        # Clean speaker name
                        speaker = self._clean_speaker_name(speaker)
                        
                        # Get index and text with standardized keys
                        utterance_num = item.get('index', 0)
                        text = item.get('text', '')
                        
                        result.append({
                            "speaker": speaker,
                            "index": utterance_num,
                            "text": text
                        })
                return result
        except (SyntaxError, ValueError):
            # If ast parsing fails, continue to line-by-line parsing
            pass
        
        # If structured parsing failed, try line-by-line parsing as a last resort
        lines = dialogue_text.split('\n')
        utterance_index = 1
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                result.append({
                    "speaker": speaker,
                    "index": utterance_index,
                    "text": text
                })
                utterance_index += 1
        
        return result
    
    def generate_all_dialogues_no_batch(self, limit: Optional[int] = None, shuffle: bool = True) -> List[Dict[str, Any]]:
        """Generate dialogues for all words in the data"""
        if not self.data:
            print("No data available to generate dialogues")
            return []
        
        # Optionally shuffle the data
        data_to_process = self.data.copy()
        if shuffle:
            random.shuffle(data_to_process)
        
        # Limit the number of items to process if specified
        if limit and limit > 0:
            data_to_process = data_to_process[:limit]
        
        results = []
        
        # temp_stop_idx = 10
        # Process each word with progress bar
        for i, item in tqdm(enumerate(data_to_process), desc=f"Generating {self.language} dialogues", total=len(data_to_process)):
            # Generate dialogue
            result = self.generate_dialogue(item)
            
            # Add to results list
            results.append(result)
            
            # Save intermediate results every 10 items
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(results)
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
            
            # if i == temp_stop_idx:
            #     break
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]) -> None:
        """Save intermediate results to avoid losing progress"""
        try:
            # Create temporary filename
            temp_filename = f"{self.output_file}.temp"
            
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            print(f"Saved intermediate results ({len(results)} items)")
        except Exception as e:
            print(f"Error saving intermediate results: {e}")
    
    def save_results(self, dialogues: List[Dict[str, Any]]) -> bool:
        """Save all dialogues to a single JSON file"""
        if not dialogues:
            print("No dialogues to save")
            return False
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(dialogues, f, ensure_ascii=False, indent=4)
            print(f"All dialogues saved to {self.output_file}")
            return True
        except Exception as e:
            print(f"Error saving dialogues to file: {e}")
            return False
    
    def run(self, limit: Optional[int] = None, shuffle: bool = True) -> None:
        """Run the dialogue generation process with batch processing"""
        print(f"Starting dialogue generation for language: {self.language} with batch size: {self.batch_size}")
        
        # Generate dialogues
        dialogues = self.generate_all_dialogues(limit=limit, shuffle=shuffle)
        
        # Save all dialogues to a single file
        success = self.save_results(dialogues)
        
        print(f"Dialogue generation completed. Generated {len(dialogues)} dialogues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialogue Generation Tool')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], 
                        help='Language code (en/fr/ko/ja)')
    parser.add_argument('--model', '-m', default="chatgpt-4o-latest", 
                        help='OpenAI model to use (default: chatgpt-4o-latest)')
    parser.add_argument('--limit', '-n', type=int, help='Limit the number of dialogues to generate')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the data before processing')
    parser.add_argument('--batch-size', '-b', type=int, default=16, 
                        help='Number of dialogues to generate in a single batch (default: 16)')
    parser.add_argument('--dialogues-per-word', '-d', type=int, default=5,
                        help='Number of dialogues to generate for each word (default: 5)')
    parser.add_argument('--step-by-step', '-s', default=1, type=int, help='Generate dialogues step by step. 0 is False and 1 is True. (default: 1)')
    
    args = parser.parse_args()
    
    generator = DialogueGenerator(
        language=args.language,
        model=args.model,
        batch_size=args.batch_size,
        dialogues_per_word=args.dialogues_per_word,
        step_by_step=args.step_by_step
    )
    
    generator.run(limit=args.limit, shuffle=not args.no_shuffle)