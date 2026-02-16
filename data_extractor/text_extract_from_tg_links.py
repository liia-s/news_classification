import os
import sys
import csv
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
from status_file import checking_or_writing_status


load_dotenv()
PATH_INFO = os.getenv('PATH_INFO')

if not PATH_INFO:
    print('The PATH_INFO environment variable is NOT set')
    sys.exit(1)

input_file = os.path.join(PATH_INFO, 'data/raw/extracted_links_oisite.csv')
output_file = os.path.join(PATH_INFO, 'data/processed/parsed_news_text_oisite.csv')

# Function for extracting text from a div on a web page
def extract_text_from_div(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    message_div = soup.find('div', class_='tgme_widget_message_text js-message_text')
    
    if message_div:
        for unwanted in message_div.find_all(['video', 'img']):
            unwanted.decompose()

    text = ""

    if message_div:
        
        for br in message_div.find_all('br'):
            br.replace_with(' ')
        
        # Convert links to text format
        for a in message_div.find_all('a'):
            a.replace_with(f"{a.get_text()} ({a['href']})")
        
        text = message_div.get_text()
        
        # Removing extra spaces
        text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Function for counting lines in a CSV file
def count_lines_in_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for line in f)

# The main function for processing a CSV file
def process_extracted_links():
    try:
        checking_or_writing_status('oisite_links_finished', mode='c')
        total_lines = count_lines_in_file(input_file) - 1  # Decrease by 1 since the first line is the title

        with open(input_file, newline='', encoding='utf-8') as csvfile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            fieldnames = reader.fieldnames + ['extracted_text'] # Adding a new field to the output CSV
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')

            writer.writeheader()

            # Using tdm to display progress
            for row in tqdm(reader, total=total_lines):
                link = row.get('link', '')

                if 't.me' in link:
                    modified_link = f"{link}?embed=1&mode=tme"
                    try:
                       # Loading the page content
                        response = requests.get(modified_link, timeout=5)
                        response.raise_for_status()
                        extracted_text = extract_text_from_div(response.text)
                    except requests.exceptions.RequestException as e:
                        print(f"Error loading {modified_link}: {e}")
                        extracted_text = 'Error loading'
                else:
                    extracted_text = ""

                # Adding the extracted text to the data string
                row['extracted_text'] = extracted_text
                writer.writerow(row)

        checking_or_writing_status('text_extract_finished', value=True, mode='w')
        print(f"The results are saved in {output_file}")
        
    except Exception as e:
        print(f"Error when executing the script: {e}")
        sys.exit(1)