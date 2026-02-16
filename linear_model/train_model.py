import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

tqdm.pandas() 
from data_loader import DataProcess
from data_preprocessing import full_text_processing
from evaluator import (MetricsEvaluation, remove_duplicates, split_data)

from dotenv import load_dotenv
load_dotenv()
MAIN_PATH = os.getenv('MAIN_PATH')

PARSED_NEWS_OISITE = os.path.join(MAIN_PATH, 'data/parsed_news_text_oisite.csv')
OI_CHANNELS_LINKS = os.path.join(MAIN_PATH, 'data/raw/ovdinfo/ovd_info_channels_links.csv')
EXTRA_POSITIVES = os.path.join(MAIN_PATH, 'data/raw/parsed_sites_extra_positives.cvs.tar.xz')

EXTRACTED_LINKS_OISITE = os.path.join(MAIN_PATH, 'data/raw/extracted_links_oisite.csv')

OUTPUT_MERGED_DF_PATH = os.path.join(MAIN_PATH, 'data/processed')
METRICS_OUTPUT_PATH = os.path.join(MAIN_PATH, 'data/processed/model_metrics.json')

SAVE_PATH = os.path.join(MAIN_PATH, 'data/models_outputs_draft')

data_processor = DataProcess()

parsed_news_oisite_data = data_processor.load_news_data(PARSED_NEWS_OISITE)
oi_channels_links_data = data_processor.load_news_data(OI_CHANNELS_LINKS)
extra_positives_data = data_processor.load_news_data(EXTRA_POSITIVES)

extracted_links_oisite = data_processor.load_news_data(EXTRACTED_LINKS_OISITE)

# Clean and preprocess extra positives data
extra_positives_data.rename(columns={'parsed_sites.cvs':'nid'}, inplace=True)
extra_positives_data = extra_positives_data[extra_positives_data.summary.notna()]
extra_positives_data['nid'] = extra_positives_data['nid'].astype('int64')

# Map dates from extracted_links_oisite to extra_positives_data
date_dict = extracted_links_oisite.set_index('nid')['created_at'].to_dict()
extra_positives_data['date'] = extra_positives_data['nid'].map(date_dict)

parsed_news_oisite_data['true_label'] = 1
parsed_news_oisite_data['data_source'] = 'parsed_news_text_oisite'

oi_channels_links_data['true_label'] = 0
oi_channels_links_data['data_source'] = 'ovd_info_channels_links'

extra_positives_data['true_label'] = 1
extra_positives_data['data_source'] = 'extra_positives'

parsed_news_oisite_data.rename(columns={'created_at':'date',
                                        'extracted_text':'content'}, inplace=True)
oi_channels_links_data.rename(columns={'url':'link'}, inplace=True)
extra_positives_data.rename(columns={'summary':'content'}, inplace=True)
dataframes = [parsed_news_oisite_data, oi_channels_links_data, extra_positives_data]

# Apply processing steps to each dataframe
processed_dataframes = []
for df in dataframes:
    df = data_processor.channel_from_url(df, link_column='link')
    df = data_processor.unify_date_format(df, date_column='date')
    df = data_processor.remove_nan_values(df, text_column='content', link_column='link', date_column='date')
    
    df = data_processor.filter_news_by_channel(df, channel_column='channel')
    
    processed_dataframes.append(df)

merged_df = pd.concat(processed_dataframes, ignore_index=True)

merged_df = merged_df[['data_source', 'link', 'channel', 'date', 'content', 'true_label']]

excluded_ua_channels = ['u_now', 'truexanewsua']
merged_df = merged_df[~merged_df['channel'].isin(excluded_ua_channels)]

processed_data = remove_duplicates(merged_df)

# Save the merged dataset to a csv.gz file
data_processor.save_to_csv(processed_data,
                           output_dir=OUTPUT_MERGED_DF_PATH,
                           filename='prep_extended_news_data_upd', 
                           compressed=True, 
                           add_date=False)
                           
# Apply full text preprocessing to the data
news_data_model = full_text_processing(processed_data,
                                      text_column='content',
                                      channel_column='channel',
                                      is_training=True) 

news_data_model = news_data_model[(news_data_model['final_processed_text'] != '') & (news_data_model['final_processed_text'].notna())]

# Split the processed data into train, train_holdout, dev, and test sets
train_data, train_holdout_data, dev_data, test_data = split_data(news_data_model)
X_train = train_data['final_processed_text']
Y_train = train_data['true_label']

X_holdout = train_holdout_data['final_processed_text']
Y_holdout = train_holdout_data['true_label']

X_dev = dev_data['final_processed_text']
Y_dev = dev_data['true_label']

X_test = test_data['final_processed_text']
Y_test = test_data['true_label']

pipeline = Pipeline([('count', CountVectorizer(min_df=35)), 
                     ('tfidf', TfidfTransformer()), 
                     ('scaler', RobustScaler(with_centering=False)),
                     ('classifier', LogisticRegression(max_iter=1000,
                                                       penalty='l1',
                                                       C=2,
                                                       solver='liblinear'))])

pipeline.fit(X_train, Y_train)

# Save the model`s pipeline
joblib.dump(pipeline, f'{SAVE_PATH}/model.joblib')

# Define datasets for evaluation and initialize the metrics evaluator
datasets = {'train': (train_data, X_train, Y_train),
            'train_holdout': (train_holdout_data, X_holdout, Y_holdout),
            'dev': (dev_data, X_dev, Y_dev),
            'test': (test_data, X_test, Y_test)}

metrics_eval = MetricsEvaluation(metrics_output_path=METRICS_OUTPUT_PATH)

for data_type, (data, text, true_labels) in datasets.items():
    metrics_eval.evaluate_classification(data=data, 
                                        text=text, 
                                        true_labels=true_labels, 
                                        model=pipeline, 
                                        model_name='LogisticRegression_scaled_c_2', 
                                        data_type=data_type)
    # Predict labels and probabilities
    data['predicted_label'] = pipeline.predict(text)
    data['predicted_prob'] = pipeline.predict_proba(text)[:, 1]
    data['dataset_type'] = data_type
    
    # Save predictions and probabilities for each dataset type
    data_processor.save_to_csv(data, output_dir=SAVE_PATH, 
                               filename=f'{data_type}_predictions', 
                               compressed=True, add_date=True)
