import pandas as pd
import numpy as np
import os
import re
import warnings
import gzip
import zipfile
import lzma
from datetime import datetime

warnings.filterwarnings('ignore')

class DataProcess:
    '''
    Class for loading and preparing a dataframe to a prediction-ready format for the model.
    
    This class provides methods to:
    - Load data from various file formats (.gz, .zip, .xz, .csv)
    - Rename columns for consistency
    - Remove nan values in specific columns
    - Unify date formats and handle timezones
    - Filter data by channel, date range, and word count (optional)
    - Prepare the data structure required for making model predictions
    '''
    
    channel_extract_regex = re.compile(r'https?://(www\.)?([^/]+)')
    domain_regex = re.compile(r'\.(ru|com|org|net|info|edu|gov|io)$')
    
    def __init__(self, 
                 news_data_path=None):
        
        self.news_data_path = news_data_path
    
    def _column_exists(self, news_data, column):
        '''
        Function to check if a column exists in the dataframe.
        '''
        return column in news_data.columns
    
    def get_latest_modified_file(self, 
                                 directory, 
                                 file_extensions=[".gz", ".zip", ".xz", ".csv"]):
        '''
        Finds and returns the most recently modified file with one of the specified extensions in a given directory.
        
        directory: the path to the directory where files are located.
        file_extensions: list of acceptable file extensions
        '''
        try:
            # List all files in the directory with the specified extensions
            files = [f for f in os.listdir(directory) if any(f.endswith(names) for names in file_extensions)]
            
            # Check if any files match the extensions
            if not files:
                print(f" No files with the specified extensions {file_extensions} found in the directory: {directory}")
                return None
            
            # Get the full paths of files
            full_paths = [os.path.join(directory, f) for f in files]
            
            # Find the most recently modified file
            latest_file = max(full_paths, key=os.path.getmtime)
            print(f"✅ The latest file in {directory} is: {latest_file}")
            return latest_file
        
        except Exception as e:
            print(f"❗️ An error occurred while finding the latest file: {e}")
            return None
        
    def load_news_data(self, file_path):
        '''
        Loads the news data from various file formats (.gz, .zip, .xz, .csv).
        
        file_path: the path to the file with the news data.
        '''
        try:
            if file_path.endswith('.gz'):
                return self.load_gz_file(file_path)
            elif file_path.endswith('.zip'):
                return self.load_zip_file(file_path)
            elif file_path.endswith('.xz'):
                return self.load_xz_file(file_path)
            else:
                return self.load_csv_file(file_path)
        except Exception as e:
            raise ValueError(f"❗️ An error occurred while reading the file {file_path}: {e}")

    def load_gz_file(self, file_path):
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            news_data = pd.read_csv(file, sep=';', quotechar='"')
        print(f"✅ The data was successfully loaded from the .gz file: {file_path}")
        return news_data

    def load_zip_file(self, file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            csv_filename = next((name for name in zip_file.namelist() if name.endswith('.csv')), None)
            if not csv_filename:
                raise ValueError("❗️ The zip archive doesn`t contain any csv files")
            with zip_file.open(csv_filename) as file:
                news_data = pd.read_csv(file, encoding='utf-8', sep=';', quotechar='"')
        print(f"✅ The data was successfully loaded from the .zip file: {file_path}")
        return news_data

    def load_xz_file(self, file_path):
        with lzma.open(file_path, 'rt', encoding='utf-8') as file:
            news_data = pd.read_csv(file, sep='\t', quotechar='"', encoding='utf-8')
        print(f"✅ The data was successfully loaded from the .xz file: {file_path}")
        return news_data

    def load_csv_file(self, file_path):
        for sep in [';', ',']:
            try:
                news_data = pd.read_csv(file_path, encoding='utf-8', sep=sep, quotechar='"')
                print(f"✅ The data was successfully loaded from the file: {file_path} with'{sep}' separator.")
                return news_data
            except pd.errors.ParserError:
                print(f"❗️ Failed with '{sep}' separator, trying next...")
        raise ValueError(f"❗️ Failed to load the file with both ';' and ',' separators: {file_path}")
    
    def rename_columns(self, 
                       news_data, 
                       content_column='summary', 
                       date_column='date', 
                       link_column='link'):
        '''
        Renames specified columns in the dataframe to standard names ('content', 'date', 'link'), 
        only if those columns exist in the dataframe.
        
        news_data: dataframe containing the news data.
        text_column: name of the content column.
        date_column: name of the date column.
        link_column: name of the link column.
        '''
        rename_dict = {content_column: 'content', 
                       date_column: 'date', 
                       link_column: 'link'}
        
        columns_to_rename = {old: new for old, new in rename_dict.items() if old in news_data.columns}
        
        if columns_to_rename:
            news_data.rename(columns=columns_to_rename, inplace=True)
            print("✅ Columns were renamed successfully.")
        else:
            print(" No specified columns were found to rename.")
        
        return news_data
    
    def remove_nan_values(self, 
                          news_data, 
                          text_column='content', 
                          link_column='link',
                          date_column='date'):
        '''
        Removes rows with nan values in the specified text and date columns, if they exist.
        
        news_data: dataframe containing the news information.
        text_column: name of the column that contains the text information to be classified
        date_column: name of the column that contains the date of the news creation.
        '''
        # Remove nan values in the text column if it exists
        if self._column_exists(news_data, text_column):
            nan_news_count = news_data[text_column].isna().sum()
            news_data.dropna(subset=[text_column], inplace=True)
            print(f" The number of nans in the '{text_column}' column: {nan_news_count}")
        else:
            print(f" Column '{text_column}' was not found.")
            
        # Remove nan values in the link column if it exists
        if self._column_exists(news_data, link_column):
            nan_news_count = news_data[link_column].isna().sum()
            news_data.dropna(subset=[link_column], inplace=True)
            print(f" The number of nans in the '{link_column}' column: {nan_news_count}")
        else:
            print(f" Column '{text_column}' was not found.")

        # Remove nan values in the date column if it exists
        if self._column_exists(news_data, date_column):
            nan_date_count = news_data[date_column].isna().sum()
            news_data.dropna(subset=[date_column], inplace=True)
            print(f" The number of nans in the '{date_column}' column: {nan_date_count}")
        else:
            print(f" Column '{date_column}' was not found.")
        
        return news_data

    def extract_channel_name(self, link):
        '''
        Extracts the channel (source) name from a url.
        
        url: the url from which to extract the channel name.
        '''
        if 't.me' in link:
            path = link.split('t.me/')[-1]
            channel_name = path.split("/")[0]  
            return channel_name if channel_name else 'Unknown tg channel'
        else:
            match = self.channel_extract_regex.search(link)
            if match:
                domain = match.group(2)
                return self.domain_regex.sub('', domain)
            return 'Channel name was not found'
    
    def channel_from_url(self,
                         news_data,
                         link_column='link'):
        '''
        Applies the channel extraction to the specified link column and adds a new column 'channel' to the dataframe.
        
        news_data: dataframe containing the news data.
        link_column: the name of the column with urls.
        '''
        if not self._column_exists(news_data, 'channel'):
            news_data['channel'] = news_data[link_column].apply(self.extract_channel_name)
            print("✅ Channel information successfully extracted from urls")
        else:
            print(" 'channel' column already exists. No changes were made.")

        return news_data
         
    def has_timezone(self, 
                     date_str):
        '''
        Checks if the date string contains timezone information (+00:00)
        
        date_str: the date string to check for timezone information
        '''
        if isinstance(date_str, pd.Timestamp):
            return date_str.tzinfo is not None
        return bool(re.search(r'[\+\-]\d{2}:\d{2}', date_str))
    
    def convert_dates(self,
                      news_data,
                      date_column='date'):
        '''
        Converts the date column to a standard datetime format, removing timezones if necessary.
        
        news_data: dataframe containing the news data.
        date_column: name of the column with date information
        '''
        # Check if the specified date column exists
        if not self._column_exists(news_data, date_column) or news_data[date_column].dropna().empty:
            print(f" Column '{date_column}' is not found or contains only nan values")
            return news_data
        
        sample_date = news_data[date_column].dropna().iloc[0]
        
        if self.has_timezone(sample_date):
            print("✅ The time zone has been defined, utc=True is used for data conversion.")
            news_data[date_column] = pd.to_datetime(news_data[date_column], errors='coerce').dt.tz_localize(None)
        else:
            print(" The time zone has NOT been defined.")
            news_data[date_column] = pd.to_datetime(news_data[date_column], errors='coerce')
        
        return news_data
    
    def unify_date_format(self, 
                          news_data, 
                          date_column='date'):
        '''
        Unifies the date format in the specified column
        '''
        news_data = self.convert_dates(news_data, date_column)
        return news_data
    
    def filter_news_by_date(self, 
                            news_data, 
                            start_date=None, 
                            end_date=None, 
                            date_column='date'):
        '''
        Filters news data by date (start_date, end_date)
        
        news_data: df containing the news data
        start_date: the starting date for filtering (optional)
        end_date: the ending date for filtering (optional)
        date_column: name of the column that contains the date of the news` creation
        '''
        news_data = self.convert_dates(news_data, date_column)

        # Convert start_date and end_date to datetime if provided
        if start_date:
            start_date = pd.to_datetime(start_date, utc=True) if self.has_timezone(news_data[date_column].dropna().iloc[0]) else pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date, utc=True) if self.has_timezone(news_data[date_column].dropna().iloc[0]) else pd.to_datetime(end_date)
        
        # Filter by date range
        if start_date:
            news_data = news_data[news_data[date_column] >= start_date]
        if end_date:
            news_data = news_data[news_data[date_column] < end_date]
        
        print(f"✅ Filtering by date completed. Number of rows after filtering: {len(news_data)}")
        return news_data

    def filter_news_by_channel(self, 
                               news_data, 
                               channel_column='channel'):
        '''
        Filters the data based on the 'channel' column (dels ovd-info news)
    
        news_data: df containing the news data
        channel_column: the column that contains the channel names
        '''
        # Exclude ovdinfo news
        if channel_column in news_data.columns:
            filtered_news = news_data[~news_data[channel_column].str.contains('ovdinfo', case=False, na=False)]
            print(f"✅ Filtering by the 'channel' column is completed. The number of news lines after filtering is equal to: {len(filtered_news)}")
        else:
            filtered_news = news_data
            print(f" Column '{channel_column}' not found in the df.")
        return filtered_news
    
    def count_words(self, 
                    text):
        ''' 
        Counts the number of words in a text
        
        text: the text column in which we need to count the number of words
        '''
        if not isinstance(text, str):
            text = str(text)
            
        words = re.findall(r'\b\w+\b', text)
        return len(words)
            
    def filter_short_news(self, 
                          news_data, 
                          text_column='extracted_text', 
                          min_word_count=7):
        '''
        Filters news with word counts less than the minimum value (min_word_count)
        Prints the count of such short news rows
    
        news_data: df containing the news data
        text_column: the column that contains the news text info
        min_word_count: the minimum word count for filtering (default is 7)
        '''
        
        news_data['word_count'] = news_data[text_column].apply(self.count_words)
        short_news = news_data[news_data['word_count'] <= min_word_count]
        filtered_news = news_data[news_data['word_count'] >= min_word_count]
        
        print(f" The number of short news rows (less than {min_word_count} words): {len(short_news)}")
        
        news_data = news_data.drop(columns=['word_count'])
        return filtered_news, short_news
    
    def save_to_csv(self, 
                    df, 
                    output_dir, 
                    filename='processed_data', 
                    compressed=False, 
                    add_date=False):
        '''
        Saves the dataframe to a csv or csv.gz file with optional date info and compression.
        
        df: the dataframe to save.
        output_dir: the directory to save the file in.
        filename: base name for the output file
        compressed: if 'True', saves as a .csv.gz file; otherwise, saves as a .csv file.
        add_date: if 'True', appends the current date to the filename.
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Add a date to the filename if specified
        if add_date:
            current_date = datetime.now().strftime('%Y-%m-%d')
            filename = f"{filename}_{current_date}"
        
        # Determine the full file path with extension
        extension = '.csv.gz' if compressed else '.csv'
        file_path = os.path.join(output_dir, f"{filename}{extension}")
        
        # Save the dataframe to csv or csv.gz
        try:
            compression_options = {'method': 'gzip'} if compressed else None
            df.to_csv(file_path, index=False, encoding='utf-8', sep=';', quotechar='"', compression=compression_options)
            print(f"✅ Data successfully was saved to: {file_path}")
        except Exception as e:
            print(f"❗️ An error occurred while saving the file: {e}")
        
def process_data(path, 
                 apply_filtering=False,
                 start_date=None, 
                 end_date=None, 
                 train_set=False,
                 content_column='summary',
                 date_column='date',
                 link_column='link',
                 channel_column='channel'):
    '''
    Main function to process data, unify it, and prepare for classification.

    path: path to the data file or directory.
    apply_filtering: whether to apply date and word count filtering.
    start_date: start date for filtering (if applicable).
    end_date: end date for filtering (if applicable).
    train_set: for model training only.
    content_column, date_column, link_column: column names to be renamed.
    channel_column: column name for channel filtering.
    '''
    data_processor = DataProcess()
    
    # Check if path is a directory and get the latest file if so
    if os.path.isdir(path):
        file_path = data_processor.get_latest_modified_file(path)
        if not file_path:
            raise ValueError("❗️ No suitable file found in the specified directory.")
    else:
        file_path = path

    # Load the news data
    data = data_processor.load_news_data(file_path)
    
    # Check if the dataframe has only one column (rename that column as 'content')
    if isinstance(data, pd.DataFrame) and len(data.columns) == 1:
        data.columns = ['content']
        print("📝 The loaded data has only one column, which we assume is the 'content' column.")

    # Rename specified columns to standardized names
    data = data_processor.rename_columns(data, 
                                         content_column=content_column, 
                                         date_column=date_column, 
                                         link_column=link_column)

    # Remove rows with nan values in essential columns
    data = data_processor.remove_nan_values(data, text_column='content', link_column='link', date_column='date')
    
    if link_column in data.columns:
        # Extract 'channel' from urls if 'channel' column does not already exist
        data = data_processor.channel_from_url(data, link_column='link')
        
    if date_column in data.columns:
        # Standardize date format and remove timezones in the date column
        data = data_processor.convert_dates(data, date_column='date')

    # Filter by channel if the channel column exists (del ovd-info news)
    if train_set and data_processor._column_exists(data, channel_column):
        data = data_processor.filter_news_by_channel(data, channel_column=channel_column)

    # Optional filtering by date range and short content removal if specified
    if apply_filtering:
        data = data_processor.filter_news_by_date(data, start_date=start_date, end_date=end_date, date_column='date')
        data, _ = data_processor.filter_short_news(data, text_column='content', min_word_count=7)

    return data
    