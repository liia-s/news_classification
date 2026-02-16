import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
import warnings
from tqdm import tqdm

class MetricsEvaluation:
    
    def __init__(self, 
                 metrics_output_path):
        self.metrics_output_path = metrics_output_path
        
    @staticmethod   
    def sigmoid(output):
        return 1 / (1 + np.exp(-output))

    def evaluate_classification(self, 
                                data,
                                text, 
                                true_labels,
                                model,
                                model_name='default_model',  
                                data_type='train'):
        '''
        Evaluates the classification quality based on true_labels and saves the metrics to metrics.json
        '''
        total_data = len(data)
        true_positive_news = sum(data['true_label'] == 1)
        true_negative_news = sum(data['true_label'] == 0)
        
        predicted_labels = model.predict(text)
        
        try:
            positive_probabilities = model.predict_proba(text)[:, 1]
        except AttributeError:
            try:
                # for svm we use sigmoid func, logreg and catboost support predict_proba
                positive_probabilities = self.sigmoid(model.decision_function(text)) 
            except AttributeError:
                raise ValueError('The model doesn`t support predict_proba or decision_function')
        
        true_predicted_negative, false_predicted_positive, false_predicted_negative, true_predicted_positive = confusion_matrix(true_labels, predicted_labels).ravel()
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, predicted_labels)
        
        precision_, recall_, _ = precision_recall_curve(true_labels, positive_probabilities)
        pr_auc = auc(recall_, precision_)
        
        print(f'{data_type} metrics:')
        
        print(f'total_data: {total_data}')
        print(f'true_positive_news: {true_positive_news}')
        print(f'true_negative_news: {true_negative_news}')
        
        print(f'true_predicted_negative: {true_predicted_negative}')
        print(f'false_predicted_positive: {false_predicted_positive}')
        print(f'false_predicted_negative: {false_predicted_negative}')
        print(f'true_predicted_positive: {true_predicted_positive}')
        
        print(f'precision: {precision:.2f}')
        print(f'recall: {recall:.2f}')
        print(f'f1-score: {f1:.2f}')
        print(f"pr auc score: {pr_auc:.2f}")
        print('------------------------------------------------------')
        
        metrics = {
            data_type: {
                'total_data': total_data,
                'true_positive_news': true_positive_news,
                'true_negative_news': true_negative_news,
                'true_predicted_negative': true_predicted_negative,
                'false_predicted_positive': false_predicted_positive,
                'false_predicted_negative': false_predicted_negative,
                'true_predicted_positive': true_predicted_positive,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'pr_auc_score': pr_auc
            }
        }

        self.save_metrics(model_name, metrics)

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
                     model_name, 
                     metrics, 
                     append=True):
        '''
        Saves the metrics to the metrics.json
    
        metrics: dictionary containing the evaluation metrics
        append: whether to append to the existing file or overwrite
        '''
        try:
            if append and os.path.exists(self.metrics_output_path):
                if os.stat(self.metrics_output_path).st_size > 0:
                    with open(self.metrics_output_path, 'r', encoding='utf-8') as json_file:
                        all_metrics = json.load(json_file)
                else:
                    all_metrics = {}
            else:
                all_metrics = {}
            if model_name not in all_metrics:
                all_metrics[model_name] = {}
                
            all_metrics[model_name].update(metrics)

            all_metrics = self.json_converter(all_metrics)

            with open(self.metrics_output_path, 'w', encoding='utf-8') as json_file:
                json.dump(all_metrics, json_file, indent=4, ensure_ascii=False)
            
            print(f"Metrics have been successfully saved")
        except Exception as e:
            print(f"Error saving metrics: {e}")
        
            
def split_data(data,
            date_column='date', 
            split_date='2024-06-01', 
            random_seed=42):
    '''
    Splits data into 4 sets: train, train_holdout, dev, and test
    '''
    data[date_column] = pd.to_datetime(data[date_column])
    
    train_data = data[data[date_column] < split_date]
    val_data = data[data[date_column] >= split_date]
    
    train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    dev_data, test_data = train_test_split(val_data, test_size=0.5, random_state=random_seed)
    
    train_holdout = train_data.sample(frac=0.1, random_state=random_seed).reset_index(drop=True)
    
    train_data = train_data.drop(train_holdout.index).reset_index(drop=True)

    print(f'train size: {len(train_data)}')
    print(f'train_holdout size: {len(train_holdout)}')
    print(f'dev size: {len(dev_data)}')
    print(f'test size: {len(test_data)}')
    print(f'total: {len(train_data) + len(train_holdout) + len(dev_data) + len(test_data)}')
    
    return train_data, train_holdout, dev_data, test_data

def remove_duplicates(df):
    df_true_label_1 = df[df['true_label'] == 1]
    df_other = df[df['true_label'] != 1]
    df_true_label_1 = df_true_label_1.drop_duplicates(subset='link', keep='last')
    df_other = df_other.drop_duplicates(subset='link', keep='last')
    df_other = df_other[~df_other['link'].isin(df_true_label_1['link'])]
    df_combined = pd.concat([df_true_label_1, df_other])
    return df_combined

def plot_learning_curve(model, 
                        X, 
                        Y, 
                        title, 
                        train_sizes=None, 
                        scoring='neg_log_loss', 
                        save_path=None):
    
    if train_sizes is None:
        train_sizes = np.linspace(0.01, 1.0, 30)

    train_sizes, train_scores, valid_scores = learning_curve(model, 
                                                             X, 
                                                             Y, 
                                                             train_sizes=train_sizes, 
                                                             scoring=scoring, 
                                                             n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='gray')
    
    ax.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1, color='gray')
    
    ax.plot(train_sizes, train_scores_mean, 'o-', color='purple', label='training score', markersize=3)
    ax.plot(train_sizes, valid_scores_mean, 'o-', color='blue', label='validation score', markersize=3)

    ax.set_title(title)
    ax.set_xlabel('training set size')
    ax.set_ylabel(f'score ({scoring})')
    ax.legend(loc='best')
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return fig 


def evaluate_holdout(data, 
                    model, 
                    model_name, 
                    metrics_evaluator, 
                    date_column='date', 
                    split_date='2024-06-01',
                    min_df_values=np.arange(5, 400, 10),
                    train_sizes=None, 
                    scoring='neg_log_loss', 
                    save_path=None,
                    random_seed=42):
    
    train_data, train_holdout, dev_data, test_data = split_data(data, date_column, split_date)
    
    
    # training holdout
    X_train = train_data['final_processed_text']
    Y_train = train_data['true_label']
    
    # holdout
    X_holdout = train_holdout['final_processed_text']
    Y_holdout = train_holdout['true_label']

    # val
    X_dev, Y_dev = dev_data['final_processed_text'], dev_data['true_label']
    X_test, Y_test = test_data['final_processed_text'], test_data['true_label']
        
    vocab_sizes = []
    
    if save_path:
        pdf_path = f'{save_path}/all_graphs_holdouts.pdf'
        pdf_pages = PdfPages(pdf_path)
        
    for min_df in min_df_values:
        tfidf_vectorizer = TfidfVectorizer(min_df=min_df)  
        
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        
        X_holdout_tfidf = tfidf_vectorizer.transform(X_holdout)
         
        X_dev_tfidf = tfidf_vectorizer.transform(X_dev)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        vocab_size = len(tfidf_vectorizer.get_feature_names_out())
        vocab_sizes.append(vocab_size)
        
        
        model_full = f'{model_name}_min_df_{min_df}_vocab_{vocab_size}'
        

        model.fit(X_train_tfidf, Y_train)
        
    
        
        metrics_evaluator.evaluate_classification(data=train_data,
                                            text=X_train_tfidf,
                                            true_labels=Y_train,
                                            model=model,
                                            model_name=model_full,
                                            data_type='train')
        
        metrics_evaluator.evaluate_classification(data=train_holdout,
                                                text=X_holdout_tfidf,
                                                true_labels=Y_holdout,
                                                model=model,
                                                model_name=model_full,
                                                data_type='train_holdout')
        
        metrics_evaluator.evaluate_classification(data=dev_data,
                                                text=X_dev_tfidf,
                                                true_labels=Y_dev,
                                                model=model,
                                                model_name=model_full,
                                                data_type='dev')
        
        metrics_evaluator.evaluate_classification(data=test_data,
                                                text=X_test_tfidf,
                                                true_labels=Y_test,
                                                model=model,
                                                model_name=model_full,
                                                data_type='test')
        fig = plot_learning_curve(model, 
                    X_train_tfidf, 
                    Y_train, 
                    title=f'learning curve for {model_full}',  
                    train_sizes=train_sizes,
                    scoring=scoring, 
                    save_path=None)

        if save_path:
            pdf_pages.savefig(fig) 
        plt.close(fig)

    if save_path:
        pdf_pages.close()
    df_vocab_sizes = pd.DataFrame({'min_df': min_df_values,
                                   'vocab_size': vocab_sizes})
    if save_path:
        df_vocab_sizes.to_csv(f'{save_path}/vocab_size_vs_min_df.csv', index=False)
