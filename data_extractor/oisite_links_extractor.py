import csv
import os
import sys
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dotenv import load_dotenv
from status_file import checking_or_writing_status
from text_extract_from_tg_links import process_extracted_links

load_dotenv()
PATH_INFO = os.getenv('PATH_INFO')

if not PATH_INFO:
    print('The PATH_INFO environment variable is NOT set')
    sys.exit(1)

# Specify the path to your input and output CSV files
input_file_path = os.path.join(PATH_INFO, 'data/processed/news_data_oisite.csv')
output_file_path = os.path.join(PATH_INFO, 'data/raw/extracted_links_oisite.csv')
status_file_path = os.path.join(PATH_INFO, 'status.json')


# Domains to exclude from the output
excluded_domains = {'ovd.info', 'ovd.news', 'ovdinfo.media', 'ovdinfo.org', 'repression.info', 'reports.ovd.in'}


# Function to extract links and text from HTML content
def extract_links_and_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    links = [(a['href'], a.get_text()) for a in soup.find_all('a', href=True)]
    return links

# Function to check if a link should be excluded based on its domain
def is_excluded(link):
    parsed_url = urlparse(link)
    domain = parsed_url.netloc
    return domain in excluded_domains or domain == ''

def links_extractor():
    try:
    # Open the input CSV file and read its contents
        
        with open(input_file_path, mode='r', newline='', encoding='utf-8') as input_file:
            reader = csv.DictReader(input_file, delimiter=';')

            # Prepare to write to the output CSV file
            with open(output_file_path, mode='w', newline='', encoding='utf-8') as output_file:
                fieldnames = ['nid', 'index', 'text', 'link']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=';')
                
                # Write the header to the output CSV
                writer.writeheader()
                
                # Iterate through each row in the input CSV
                for row in reader:
                    # Extract the 'text' column containing HTML
                    html_text = row['text']
                    
                    # Extract all links and their text from the 'text' column
                    links_with_text = extract_links_and_text_from_html(html_text)
                    
                    # Write each extracted link and text to the output CSV, excluding certain domains
                    for index, (href, text) in enumerate(links_with_text):
                        if not is_excluded(href):
                            writer.writerow({
                                'nid': row['nid'], 
                                'index': index, 
                                'text': text, 
                                'link': href})
                        
        checking_or_writing_status('oisite_links_finished', value=True, mode='w')
        print(f'Extracted links and their texts have been written to {output_file_path}, excluding specified domains')

    except Exception as e:
        print(f'Error when executing the script: {e}')
        sys.exit(1)
        
    try:
        print("Running the second script...")
        process_extracted_links()
        print("The second script was executed successfully")
        
    except Exception as e:
        print(f"Error when executing the second script: {e}")
        sys.exit(1) 

if __name__ == "__main__":
    links_extractor()
