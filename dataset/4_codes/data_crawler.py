# EXAMPLE: python data_crawler.py -l en
# Remove " "
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import argparse
import os
import time
from tqdm import tqdm
from urllib.parse import urljoin, quote
from random import uniform

class DataCrawler:
    def __init__(self, language):
        """
        Initialize the data crawler
        
        Args:
            language (str): Language code (en/ja/ko/fr)
        """
        self.language = language.lower()
        if self.language not in ['en', 'ja', 'ko', 'fr']:
            raise ValueError("Language must be one of 'en', 'ja', 'ko', 'fr'.")
        
        # Set base path
        self.input_csv = os.path.join('../0_raw/nat', f"{self.language}.csv")
        
        # Set output path
        self.output_path = os.path.join("../1_preprocess/nat")
        self.output_json = os.path.join(self.output_path, f"{self.language}.json")
        
        # Set language-specific default base URLs
        self.base_urls = self._get_default_base_urls()
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # User agent rotation to avoid being blocked
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        # Request session for better performance and cookie handling
        self.session = requests.Session()
    
    def _get_default_base_urls(self):
        """Get default base URLs for the selected language"""
        default_urls = {
            'en': [
                'https://en.wiktionary.org/wiki/'
            ],
            'fr': [
                "https://www.le-dictionnaire.com/definition/",
                "https://dictionnaire.lerobert.com/definition/",
                "https://dictionnaire.reverso.net/francais-definition/",
                "https://www.dictionnaire-academie.fr/article/",
                "https://fr.wiktionary.org/wiki/"
                ],
            'ja': [
                'https://ja.wiktionary.org/wiki/'
            ],
            'ko': [
                'https://ko.wiktionary.org/wiki/'
            ]
        }
        return default_urls.get(self.language, [])
    
    def _build_url(self, search_term, base_url):
        """Create URL using search term based on language"""
        # Encode search term for URL
        encoded_term = quote(search_term)
        
        # Build URL
        return urljoin(base_url, encoded_term)
    
    def _crawl_with_bs4(self, url, search_term):
        """Crawl web page using BeautifulSoup with error handling"""
        try:
            # Random user agent to avoid being blocked
            headers = {
                'User-Agent': self.user_agents[hash(url) % len(self.user_agents)],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            }
            
            # Add a delay to avoid being blocked (random between 1-3 seconds)
            time.sleep(uniform(1, 3))
            
            # Use session for better performance and cookie handling
            response = self.session.get(url, headers=headers, timeout=10)
            
            # Check if the request was successful
            if response.status_code != 200:
                return {
                    'word': search_term,
                    'meaning': [],
                    'url': url,
                    'found': False,
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Language-specific parsing
            if self.language == 'en':
                return self._parse_english(soup, url, search_term)
            elif self.language == 'fr':
                return self._parse_french(soup, url, search_term)
            elif self.language == 'ja':
                return self._parse_japanese(soup, url, search_term)
            elif self.language == 'ko':
                return self._parse_korean(soup, url, search_term)
            else:
                raise ValueError(f"Unsupported language: {self.language}")
        except requests.exceptions.RequestException as e:
            # Handle request exceptions (timeout, connection error, etc.)
            return {
                'word': search_term,
                'meaning': [],
                'url': url,
                'found': False,
                'error': f"Request error: {str(e)}"
            }
        except Exception as e:
            # Handle other exceptions
            return {
                'word': search_term,
                'meaning': [],
                'url': url,
                'found': False,
            }
    
    def _parse_english(self, soup:BeautifulSoup, url:str, search_term:str) -> dict:
        #### TODO: Implement English parsing
        return {
            'word': search_term,
            'meaning': [],
            'url': url,
            'found': False
        }
    
    def _parse_japanese(self, soup:BeautifulSoup, url:str, search_term:str) -> dict:
        #### TODO: Implement Japanese parsing
        return {
            'word': search_term,
            'meaning': [],
            'url': url,
            'found': False
        }
        
    def _parse_korean(self, soup:BeautifulSoup, url:str, search_term:str) -> dict:
        #### TODO: Implement Korean parsing
        return {
            'word': search_term,
            'meaning': [],
            'url': url,
            'found': False
        }
    
    def _parse_french(self, soup:BeautifulSoup, url:str, search_term:str) -> dict:
        """Parse French websites for onomatopoeic words"""
        try:
            # Wiktionary
            if url.startswith("https://fr.wiktionary.org/wiki/"):
                # Check for etymology sections containing onomatopoeia-related words
                etymology_sections = soup.find_all("span", id=lambda x: x and "tymologie" in x)
                has_onomatopoeia_etymology = False
                
                for section in etymology_sections:
                    section_content = section.find_next("p")
                    if section_content:
                        text = section_content.text.lower()
                        if any(word in text for word in ["onomatopée", "cri", "bruit", "son", "idéophone"]):
                            has_onomatopoeia_etymology = True
                            break
                
                if not has_onomatopoeia_etymology:
                    return {
                        'word': search_term,
                        'meaning': [],
                        'url': url,
                        'found': False
                    }
                
                # Find meaning sections
                meanings = []
                definition_lists = soup.find_all("ol", class_="liste_de_traductions")
                
                for dl in definition_lists:
                    for li in dl.find_all("li"):
                        text = li.text.strip()
                        if any(word in text.lower() for word in ["onomatopée", "cri", "bruit", "son", "idéophone"]):
                            meanings.append(text)
                
                return {
                    'word': search_term,
                    'meaning': meanings,
                    'url': url,
                    'found': len(meanings) > 0
                }
                
            # le-dictionnaire.com
            elif url.startswith("https://www.le-dictionnaire.com/definition/"):
                def_boxes = soup.find_all("div", class_="defbox")
                
                # Pass if there is no word found in the dictionary
                is_error = soup.find("span").text.strip().startswith("Aucun mot trouvé") or soup.find("span").text.strip().startswith("Le mot exact n'a pas été trouvé")
                if is_error:
                    return {
                        'word': search_term,
                        'meaning': [],
                        'url': url,
                        'found': False
                    }
                
                # Iterate over all the definitions
                for def_box in def_boxes:
                    word = def_box.find("b").text.strip()
                    if word != search_term:
                        continue
                    sub_word = def_box.find("span").text.strip().split()[-1].strip("()").lower()
                    
                    # Only keep the definitions that are explicitly onomatopoeia or ideophone
                    if sub_word not in ["onomatopée", "idéophone"]:
                        continue
                    
                    # Get all the definitions
                    definitions = [p.text.strip() for p in def_box.find_all('li')]
                    found = True
                    return {
                        'word': search_term,
                        'meaning': definitions,
                        'url': url,
                        'found': bool(definitions)
                    }
                    
                return {
                    'word': search_term,
                    'meaning': [],
                    'url': url,
                    "found": False,
                }
                
            # lerobert.com
            elif url.startswith("https://dictionnaire.lerobert.com/definition/"):
                def_box = soup.find("div", class_="d_ptma")
                definitions = [p.text.strip() for p in def_box.find_all('span', class_="d_dfn")]
                if len(definitions) == 0:
                    return {
                        'word': search_term,
                        'meaning': [],
                        'url': url,
                        'found': False
                    }
                return {
                    'word': search_term,
                    'meaning': definitions,
                    'url': url,
                    'found': True
                }
            
            # reverso.net
            elif url.startswith("https://dictionnaire.reverso.net/francais-definition/"):
                translate_box = soup.find("div", class_="translate_box0")
                definitions = [p.text.strip() for p in translate_box.find_all('span', id_="ID0EYB")]
                return {
                    'word': search_term,
                    'meaning': [],
                    'url': url,
                    'found': False
                }
            
            else:
                raise ValueError(f"Unsupported url: {url}")
        except Exception as e:
            # Handle parsing errors
            return {
                'word': search_term,
                'meaning': [],
                'url': url,
                'found': False,
            }

    def crawl_data(self):
        """Read CSV file and perform crawling for each row"""
        # Read CSV file
        try:
            # Read existing JSON file
            with open(self.output_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Successfully read JSON file. Total {len(results)} items.")
            
            # Process each item
            for item in tqdm(results, desc=f"Processing {self.language} data"):
                # Skip if already found
                if item['found']:
                    continue
                    
                # Fix items with "dictionnaire des onomatopées" url
                if item['url'] == "dictionnaire des onomatopées" and not item['found']:
                    item['found'] = True
                    continue
                    
                # Try Wiktionary for items still not found
                if not item['found'] and self.language == 'fr':
                    search_term = item['word']
                    url = self._build_url(search_term, "https://fr.wiktionary.org/wiki/")
                    result = self._crawl_with_bs4(url, search_term)
                    
                    if result['found']:
                        item['meaning'].extend(result['meaning'])
                        item['url'] = result['url']
                        item['found'] = True
                    
                    # Add a delay between requests
                    time.sleep(uniform(1, 2))
            
            return results
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
    
    def save_to_json(self, results:list[dict]):
        """Save results list to JSON file"""
        if results is not None and len(results) > 0:
            try:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(self.output_json)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(self.output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                
                print(f"Results successfully saved to {self.output_json}.")
                return True
            except Exception as e:
                print(f"Error saving JSON file: {e}")
        else:
            print("No data to save.")
        return False
    
    def run(self):
        """Run the crawling process"""
        results = self.crawl_data()
        success = self.save_to_json(results)
        
        return success
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Crawling Tool')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'ja', 'ko', 'fr'], 
                        help='Language code (en/ja/ko/fr)')
    
    args = parser.parse_args()
    
    crawler = DataCrawler(
        language=args.language,
    )
    
    success = crawler.run()
    
    if success:
        print("Crawling completed successfully.")
    else:
        print("Error occurred during crawling.")