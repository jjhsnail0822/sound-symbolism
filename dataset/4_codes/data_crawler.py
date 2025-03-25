# EXAMPLE: python data_crawler.py -l en

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
                "https://www.dictionnaire-academie.fr/article/"
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
        """Parse French dictionary pages"""
        try:
            # le-dictionnaire.com
            if url.startswith("https://www.le-dictionnaire.com/definition/"):
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
                breakpoint()
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
            df = pd.read_csv(self.input_csv)
            print(f"Successfully read CSV file. Total {len(df)} rows.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
        
        # List to store results
        results = []
        
        # Iterate over each row with tqdm progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Crawling {self.language} data"):
            
            # Get search term from the word column
            search_term = row.get("word", '')
            if not search_term:
                print(f"No search term in row {index+1}. Skipping.")
                continue
            
            # Get reference from the ref column for tracking purposes
            ref = row.get("ref", '')
            
            # Try each base URL until we find meaningful content
            found_data = None
            for base_url in self.base_urls:
                # Search for each urls
                url = self._build_url(search_term, base_url)
                result:dict = self._crawl_with_bs4(url, search_term)
                
                # If we found meaningful content, stop searching
                if result.get('found', False):
                    found_data = result
                    break
                
                # Add a delay between requests to different sites
                time.sleep(uniform(1, 2))
            
            # If no meaningful content was found, use the last result or create a default one
            if not found_data:
                if len(self.base_urls) > 0:
                    url = self._build_url(search_term, self.base_urls[0])
                    found_data = {
                        'word': search_term,
                        'meaning': [],
                        'url': url,
                        'found': False
                    }
                else:
                    found_data = {
                        'word': search_term,
                        'meaning': [],
                        'url': '',
                        'found': False,
                    }
            
            # Add reference to the result for tracking purposes
            found_data['ref'] = ref
            
            # Add to results
            results.append(found_data)
        
        return results
    
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