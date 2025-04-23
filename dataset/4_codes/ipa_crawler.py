import json
import argparse
import os
import time
import re
from tqdm import tqdm
from random import uniform
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote
from data_crawler import DataCrawler

class IPACrawler(DataCrawler):
    def __init__(self, language: str):
        """
        Initialize the IPA crawler
        
        Args:
            language (str): Language code (fr)
        """
        super().__init__(language)
        
        # Set paths
        self.input_path = f"../1_preprocess/nat/{self.language}_ipa.json"
        self.output_path = f"../1_preprocess/nat/{self.language}_ipa_wiktionary.json"
        
        # User agent rotation to avoid being blocked
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        # Define headers for requests
        self.headers = {
            'User-Agent': self.user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
    
    def crawl_ipa(self):
        """Crawl IPA for words in the input file"""
        try:
            # Read input JSON file
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Successfully read JSON file. Total {len(data)} items.")
            
            # Process each item
            results = []
            for item in tqdm(data, desc=f"Processing {self.language} IPA data"):
                # Initialize IPA as empty string for each item
                item_with_ipa = item.copy()
                item_with_ipa['ipa'] = ''
                
                # Only process items with found=true and empty IPA
                if item.get('found', False) and not item.get('ipa'):
                    word = item.get('word', '')
                    ipa_data = self._get_ipa_for_word(word)
                    
                    # Add IPA data to the item
                    if ipa_data.get('ipa'):
                        item_with_ipa['ipa'] = ipa_data.get('ipa', '')
                        item_with_ipa['ipa_source'] = ipa_data.get('ipa_source', '')
                
                results.append(item_with_ipa)
                
                # Add a delay between requests to avoid overloading the server
                time.sleep(uniform(1, 2))
            
            return results
        
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
    
    def _get_ipa_for_word(self, word: str) -> dict:
        """
        Get IPA for a specific word from Wiktionary
        
        Args:
            word (str): The word to get IPA for
            
        Returns:
            dict: Dictionary with IPA data
        """
        # Encode the word for URL
        encoded_word = quote(word)
        url = f"https://fr.wiktionary.org/wiki/{encoded_word}"
        
        try:
            # Random user agent to avoid being blocked
            self.headers['User-Agent'] = self.user_agents[hash(url) % len(self.user_agents)]
            
            # Get the page content
            response = requests.get(url, headers=self.headers, timeout=10)
            
            # Check if the request was successful
            if response.status_code != 200:
                return {
                    'ipa': '',
                    'ipa_source': url
                }
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all spans with class "API"
            api_spans = soup.find_all('span', class_="API")
            
            # Extract IPA from API spans
            ipa_values = []
            for span in api_spans:
                text = span.get_text()
                # Remove backslashes and extract the IPA
                ipa = text.replace('\\', '').strip()
                if ipa:
                    ipa_values.append(ipa)
            
            # If IPA values were found, return the first one
            if ipa_values:
                return {
                    'ipa': ipa_values[0],
                    'ipa_source': url
                }
            else:
                return {
                    'ipa': '',
                    'ipa_source': url
                }
                
        except Exception as e:
            print(f"Error getting IPA for word '{word}': {e}")
            return {
                'ipa': '',
                'ipa_source': url
            }
    
    def save_results(self, results):
        """Save results to output file"""
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def run(self):
        """Run the IPA crawler"""
        print(f"Starting IPA crawling for language: {self.language}")
        
        # Crawl IPA data
        results = self.crawl_ipa()
        
        if results:
            # Save results
            success = self.save_results(results)
            
            if success:
                print(f"IPA crawling completed successfully. Processed {len(results)} items.")
                
                # Count items with IPA
                items_with_ipa = sum(1 for item in results if item.get('ipa'))
                print(f"Found IPA for {items_with_ipa} out of {len(results)} items.")
                
                return True
        
        print("IPA crawling failed.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IPA Crawling Tool')
    parser.add_argument('--language', '-l', required=True, choices=['fr'], 
                        help='Language code (fr)')
    
    args = parser.parse_args()
    
    crawler = IPACrawler(
        language=args.language,
    )
    
    success = crawler.run()
    
    if success:
        print("IPA crawling completed successfully.")
    else:
        print("Error occurred during IPA crawling.")
