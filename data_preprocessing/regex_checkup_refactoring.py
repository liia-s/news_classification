import re
import pandas as pd
import numpy as np
import os
import warnings
import json
import gzip
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')

load_dotenv()

MAIN_PATH = os.getenv('MAIN_PATH')

NEWS_INPUT_PATH = os.path.join(MAIN_PATH, 'data/processed/parsed_news_text_oisite.csv')
NEWS_GZ_INPUT_PATH = os.path.join(MAIN_PATH, 'data/raw/ovdinfo/channels_sources_texts.csv.gz')
REGEX_INPUT_PATH = os.path.join(MAIN_PATH, 'data/regexp_classifier/oi_db_regexp_v1.csv')

# the base path for storing classification results
BASE_OUTPUT_PATH = os.path.join(MAIN_PATH, 'data/classification_outputs/oi_regexp_classifier')

METRICS_OUTPUT_PATH = os.path.join(MAIN_PATH, 'data/regexp_classifier/metrics.json')


class DataProcess:
    def __init__(self, 
                 news_csv_path=None, 
                 gz_news_path=None):
        
        self.news_csv_path = news_csv_path
        self.gz_news_path = gz_news_path 
    
    def load_news_data(self, 
                       file_path):
        '''
        Loads the news data
        
        file_path: the path to the file with the news data
        '''
        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as file:
                    news_data = pd.read_csv(file)
                print(f"The data was successfully uploaded from .gz file {file_path}")
            else:
                news_data = pd.read_csv(file_path, sep=';') 
                print(f"The data was successfully uploaded from {file_path}")
            return news_data
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
         
    def has_timezone(self, 
                     date_str):
        '''
        Checks if the date string contains timezone information (+00:00)
        p.s. this function is for the negative df
        
        date_str: the date string to check for timezone information
        '''
        if isinstance(date_str, pd.Timestamp):
            return date_str.tzinfo is not None
        return bool(re.search(r'[\+\-]\d{2}:\d{2}', date_str))
    
    def unify_date_format(self, 
                          news_data, 
                          date_column='date'):
        '''
        Unifies the date format across both datasets, removing timezones
        '''
        sample_date = news_data[date_column].dropna().iloc[0]
        
        if self.has_timezone(sample_date):
            print("The time zone has been defined, utc=True is used for data conversion")
            news_data[date_column] = pd.to_datetime(news_data[date_column], errors='coerce').dt.tz_localize(None)
        else:
            print("The time zone has NOT been defined")
            news_data[date_column] = pd.to_datetime(news_data[date_column], errors='coerce')
        return news_data
    
    def rename_columns(self, 
                       news_data, 
                       text_column='content', 
                       date_column='date',
                       link_column='url'):
        '''
        Renames text and date columns
        '''
        news_data.rename(columns={text_column: 'content', 
                                  date_column: 'date',
                                  link_column: 'url'}, inplace=True)
        return news_data
    
    def remove_nan_values(self,
                          news_data, 
                          text_column='content', 
                          date_column='date'):
        '''
        Removes empty rows in the text and date columns
        
        news_data: df containing the news dataset
        text_column: name of the column that contains the text info to be classified
        date_column: name of the column that contains the date of the news` creation
        '''
        # removing nan values in the news text column
        nan_news_count = news_data[text_column].isna().sum()
        news_data.dropna(subset=[text_column], inplace=True)
        print(f"The number of nans by the news columns: {nan_news_count}")
        
        # removing nan values in the news date column
        nan_date_count = news_data[date_column].isna().sum()
        news_data.dropna(subset=[date_column], inplace=True)
        print(f"The number of nans by the date column: {nan_date_count}")
        
        return news_data
    
    
    def filter_news_by_channel(self, 
                               news_data, 
                               channel_column='channel'):
        '''
        Filters the data based on the 'channel' column
        p.s. this function is necessary for the dataset provided by the volunteer (the negative df)
    
        news_data: df containing the news data
        channel_column: the column that contains the channel names
        '''
        # we need to filter the df containing negatives, so we exclude ovdinfo news
        if channel_column in news_data.columns:
            filtered_news = news_data[~news_data[channel_column].str.contains('ovdinfo', case=False, na=False)]
            print(f"Filtering by the 'channel' column is completed. The number of news lines after filtering is equal to: {len(filtered_news)}")
        else:
            filtered_news = news_data
            print(f"Column '{channel_column}' not found in the df.")
        return filtered_news
    
    def filter_news_by_date(self, 
                            news_data, 
                            start_date=None, 
                            end_date=None, 
                            date_column='created_at'):
        '''
        Filters news data by date (start_date, end_date)
        
        news_data: df containing the news data
        start_date: the starting date for filtering (optional)
        end_date: the ending date for filtering (optional)
        date_column: name of the column that contains the date of the news` creation
        '''
        try:
            sample_date = news_data[date_column].dropna().iloc[0]
            
            # convert dates depending on the presence of timezones
            if self.has_timezone(sample_date):
                print("The time zone has been defined, utc=True is used for data conversion")
                news_data[date_column] = pd.to_datetime(news_data[date_column], errors='coerce', utc=True)

                # convert start_date and end_date to utc if provided
                if start_date:
                    start_date = pd.to_datetime(start_date).tz_localize('UTC')
                if end_date:
                    end_date = pd.to_datetime(end_date).tz_localize('UTC')

            else:
                print("The time zone has NOT been defined")
                news_data[date_column] = pd.to_datetime(news_data[date_column], errors='coerce')
                if start_date:
                    start_date = pd.to_datetime(start_date)
                if end_date:
                    end_date = pd.to_datetime(end_date)

            # filter by date
            if start_date:
                news_data = news_data[news_data[date_column] >= start_date]

            if end_date:
                news_data = news_data[news_data[date_column] < end_date]

            print(f"Filtering by date is completed. The number of news rows after filtering: {len(news_data)}")
            return news_data

        except Exception as e:
            raise ValueError(f"Error filtering news by date: {e}")
    
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
        
        print(f"The number of short news rows (less than {min_word_count} words): {len(short_news)}")
        
        news_data = news_data.drop(columns=['word_count'])
        return filtered_news, short_news
    
    def process_data(self, 
                     file_path, 
                     filtered=False, 
                     start_date=None, 
                     end_date=None,
                     text_column='content',
                     date_column='created_at',
                     channel_column='channel'):
        '''
        Main function to process data, unify it and prepare for classification
        '''
        # load the news data
        news_data = self.load_news_data(file_path)
        
        #for negative only
        if channel_column in news_data.columns:
            news_data = self.filter_news_by_channel(news_data,
                                                    channel_column=channel_column)
        
        
        # unify date formats
        news_data = self.unify_date_format(news_data, 
                                           date_column=date_column)
        
        # remove nans
        news_data = self.remove_nan_values(news_data, 
                                           text_column=text_column, 
                                           date_column=date_column)

        # filter if necessary
        if filtered:
            news_data = self.filter_news_by_date(news_data, 
                                                 date_column=date_column, 
                                                 start_date=start_date, 
                                                 end_date=end_date)
            
            filtered_news_data, _ = self.filter_short_news(news_data, 
                                                           text_column=text_column, 
                                                           min_word_count=7)
            news_data = filtered_news_data

        return news_data


class MetricsEvaluation:
    
    def __init__(self, 
                 metrics_output_path=METRICS_OUTPUT_PATH):
        self.metrics_output_path = metrics_output_path

    def evaluate_classification(self, 
                                news_data, 
                                df_name,
                                true_label='true_label'):
        '''
        Evaluates the classification quality based on true_labels and saves the metrics to metrics.json

        news_data: df containing the labeled news data
        df_name: the name of the file to save the results
        '''
        if true_label not in news_data.columns:
            raise ValueError(f"The column '{true_label}' was NOT found in the df")
        
        true_labels = news_data[true_label].tolist()
        predicted_labels = news_data['predicted_label'].tolist()
        total_data = len(news_data)
        
        true_positive_news = sum(news_data[true_label] == 1)
        true_negative_news = sum(news_data[true_label] == 0)

        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        # roc_auc = roc_auc_score(true_labels, predicted_labels)

        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        true_predicted_negative, false_predicted_positive, false_predicted_negative, true_predicted_positive = conf_matrix.ravel()

        
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        # print(f"ROC AUC Score: {roc_auc:.2f}")
        # print(f"Confusion Matrix: \n{conf_matrix}")

        metrics = {
            'df_name': df_name,
            'total_data': total_data,
            'true_positive_news': true_positive_news,
            'true_negative_news': true_negative_news,
            'true_predicted_negative': true_predicted_negative,
            'false_predicted_positive': false_predicted_positive,
            'false_predicted_negative': false_predicted_negative,
            'true_predicted_positive': true_predicted_positive,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            # 'roc_auc_score': roc_auc
        }

        
        self.save_metrics(metrics)

    def json_converter(self, 
                       val):
        '''
        Converts various data types into JSON formats
        
        val: the input value to be converted
        '''
        if isinstance(val, np.int64):
            return int(val)
        if isinstance(val, np.float64):
            return round(float(val), 2)
        if isinstance(val, dict):
            return {k: self.json_converter(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self.json_converter(i) for i in val]
        return val

    def save_metrics(self, 
                     metrics, 
                     append=True):
        '''
        Saves the metrics to the metrics.json
    
        metrics: dictionary containing the evaluation metrics
        append: whether to append to the existing file or overwrite
        '''
        try:
            if not append and os.path.exists(METRICS_OUTPUT_PATH):
                with open(METRICS_OUTPUT_PATH, 'w', encoding='utf-8') as json_file:
                    json.dump([], json_file, indent=4, ensure_ascii=False)
                
            if append and os.path.exists(METRICS_OUTPUT_PATH):
                if os.stat(METRICS_OUTPUT_PATH).st_size > 0:
                    with open(METRICS_OUTPUT_PATH, 'r', encoding='utf-8') as json_file:
                        all_metrics = json.load(json_file)
                else:
                    all_metrics = []
            else:
                all_metrics = []
            
            metrics_converted = self.json_converter(metrics)
            all_metrics.append(metrics_converted)

            with open(METRICS_OUTPUT_PATH, 'w', encoding='utf-8') as json_file:
                json.dump(all_metrics, json_file, indent=4, ensure_ascii=False)
            
            print(f"Metrics have been successfully saved in {METRICS_OUTPUT_PATH}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
                

class RegexpClassifier:
    
    def __init__(self, 
                 regex_csv_path):
        
        self.regex_csv_path = regex_csv_path
        self.regex_patterns = self.load_regex()

    def load_regex(self):
        '''
        Loads regex from the csv file
        '''
        try:
            data = pd.read_csv(self.regex_csv_path)
            regex_patterns = data['regexp'].dropna().tolist()
            print("The regex file has been loaded successfully")
            return regex_patterns
        except Exception as e:
            raise ValueError(f"Error reading the file {self.regex_csv_path}: {e}")

    def classify_text(self, 
                      text, 
                      label_type='positive'):
        '''
        Classifies text based on regex
        
        text: the column in the df with news text that needs to be classified
        label_type: this label allows to classify news into positive and negative 
        '''
        for pattern in self.regex_patterns:
            if re.search(pattern, text):
                return 1 if label_type == 'positive' else 0  
        return 0 if label_type == 'positive' else 1 
    
    def classify_news(self, 
                      news_data, 
                      text_column='extracted_text', 
                      date_column='created_at',
                      label_type='positive'):
        ''' 
        Applies classification to the news dataset based on regex
        
        news_data: df containing the news dataset
        text_column: name of the column that contains the text info to be classified
        date_column: name of the column that contains the date of the news` creation
        label_type: specifies the type of classification ('positive' or 'negative')
        '''
        if text_column not in news_data.columns:
            raise ValueError(f"The column '{text_column}' was NOT found in the df")
        
        # assign true labels before classification 
        label = 1 if label_type == 'positive' else 0
        news_data['true_label'] = label
        
        news_data['predicted_label'] = news_data[text_column].apply(lambda text: self.classify_text(text, label_type))
        return news_data
    
    def generate_output_filename(self, 
                                 df_name, 
                                 filtered=False):
        '''
        Creates the output filename based on the dataset and filtering conditions
    
        df_name: the name of the dataset being processed
        filtered: indicates whether the data has been filtered
        '''
        file_type = 'filtered' if filtered else 'full'
        file_extension = '.csv'
        return os.path.join(BASE_OUTPUT_PATH, 
                            f"{file_type}_{df_name}_classification{file_extension}")

def run_classification_pipeline(data_processor, 
                                regexp_classifier, 
                                metrics_evaluator, 
                                file_path,
                                df_name='news_data',
                                text_column='content',
                                date_column='date',
                                channel_column='channel',
                                label_type='positive',
                                filtered=False,
                                start_date=None,
                                end_date=None):
    # load and process data
    data = data_processor.process_data(file_path=file_path,
                                       filtered=filtered,
                                       start_date=start_date,
                                       end_date=end_date,
                                       text_column=text_column,
                                       date_column=date_column,
                                       channel_column=channel_column)
    
    # classify data
    classified_data = regexp_classifier.classify_news(data,
                                                      text_column=text_column,  
                                                      label_type=label_type)
    
    # evaluate metrics
    metrics_evaluator.evaluate_classification(classified_data, 
                                              df_name=df_name,
                                              true_label='true_label')
    
    # save results
    output_filename = regexp_classifier.generate_output_filename(df_name=df_name,
                                                                 filtered=filtered)
    
    classified_data.to_csv(output_filename, index=False)
    
    print(f"The classification results have been successfully saved in: {output_filename}")
    
    
data_processor = DataProcess()
metrics_evaluator = MetricsEvaluation(metrics_output_path=METRICS_OUTPUT_PATH)
regexp_classifier = RegexpClassifier(regex_csv_path=REGEX_INPUT_PATH)

# original news_data_oisite data
run_classification_pipeline(data_processor,
                            regexp_classifier,
                            metrics_evaluator,
                            file_path=NEWS_INPUT_PATH,
                            df_name='news_data_oisite',
                            text_column='extracted_text',
                            date_column='created_at',
                            label_type='positive',
                            filtered=False)


run_classification_pipeline(data_processor,
                            regexp_classifier,
                            metrics_evaluator,
                            file_path=NEWS_INPUT_PATH,
                            df_name='news_data_oisite_from_june',
                            text_column='extracted_text',
                            date_column='created_at',
                            label_type='positive',
                            filtered=True,
                            start_date='2024-06-01')

# channels_sources_texts (negative, original data)
run_classification_pipeline(data_processor,
                            regexp_classifier,
                            metrics_evaluator,
                            file_path=NEWS_GZ_INPUT_PATH,
                            df_name='channels_sources_texts',
                            text_column='content',
                            date_column='date',
                            label_type='negative',
                            filtered=False)


run_classification_pipeline(data_processor,
                            regexp_classifier,
                            metrics_evaluator,
                            file_path=NEWS_GZ_INPUT_PATH,
                            df_name='channels_sources_texts_to_june',
                            text_column='content',
                            date_column='date',
                            label_type='negative',
                            filtered=True,
                            start_date='2024-06-01')
