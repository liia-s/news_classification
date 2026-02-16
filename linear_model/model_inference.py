import joblib
import pandas as pd
from datetime import datetime
import sklearn

from .data_loader import process_data, DataProcess
from .data_preprocessing import full_text_processing

class LinearModelInference:
    def __init__(self, 
                 model_path,
                 content_column='summary', 
                 channel_column='channel'):
        '''
        Initializes the class for linear model inference.
        
        tfidf_path: path to the saved TfidfVectorizer.
        scaler_path: path to the saved scaler.
        model_path: path to the saved classification model.
        stopwords_set: set of stop words for preprocessing.
        content_column: name of the column containing news content.
        date_column: name of the column containing dates.
        channel_column: name of the column containing channel information
        '''
        
        self.pipeline = joblib.load(model_path)
            
        self.content_column = content_column
        self.channel_column = channel_column
        self.data_processor = DataProcess()

    def _load_and_prepare_data(self, 
                              file_path):
        '''
        Loads and prepares data using DataProcess.
        
        file_path: path to the data file.
        '''
        data = process_data(file_path, 
                            content_column=self.content_column, 
                            link_column='link', 
                            train_set=False,
                            channel_column=self.channel_column)
        self.content_column = 'content'
        return data
    
    def _prepare_data_for_inference(self, data):
        
        '''
        Prepares input data for inference by converting it to dataframe if needed.
        
        data: input data, can be a string, list, or a df.
        '''
        if isinstance(data, str):
            data = pd.DataFrame([data], columns=[self.content_column])
        elif isinstance(data, list):
            data = pd.DataFrame(data, columns=[self.content_column])
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(" ❗️ The inputed data should be a string, list, or a df")
        return data

    def _preprocess_data(self, data):
        '''
        Performs text preprocessing using TextPreprocess.
        
        data: data file with news.
        '''
        # if isinstance(data, list):
        #     data = pd.DataFrame(data, columns=[self.content_column])
        # elif isinstance(data, str):
        #     data = pd.DataFrame([data], columns=[self.content_column])
        
        processed_data = full_text_processing(data,
                                              text_column=self.content_column, 
                                              channel_column=self.channel_column,
                                              is_training=False,
                                              disable_tqdm=True)
        
        return processed_data

    def _transform_and_predict(self,
                               data,
                               text_column='final_processed_text'):
        '''
        Transforms data and makes class predictions.
        
        data: dataFrame with preprocessed text
        text_column: name of the column with fully processed text.
        '''
        
        # Make predictions
        data['predicted_labels'] = self.pipeline.predict(data[text_column])
        data['predicted_probability'] = self.pipeline.predict_proba(data[text_column])[:, 1]
        
        if 'content' not in data.columns:
            data['content'] = data[text_column]
        
        result_columns = ['link', 'channel', 'content', 'final_processed_text', 'predicted_labels', 'predicted_probability']
        columns_to_return = [col for col in result_columns if col in data.columns and data[col].notna().any()]
    
        return data[columns_to_return]
    
    def _save_results(self, 
                     data, 
                     output_path, 
                     compressed=False, 
                     add_date=True):
        '''
        Saves the resulting dataframe to a csv file 
        
        data: the dataframe with prediction results.
        output_path: a directory path where the file will be saved.
        compressed: saves as .csv.gz if 'True', otherwise as .csv.
        add_date: adds current date to filename if 'True'.
        '''
        self.data_processor.save_to_csv(df=data, 
                                        output_dir=output_path, 
                                        filename='linear_predicted_results', 
                                        compressed=compressed, 
                                        add_date=add_date)
        
        print(f"✅ Results are saved to {output_path}")
        
    def run_inference(self, news_text):
        '''
        Runs the full model inference process on a single news text and returns the prediction result.
        
        news_text: a string with the news` text.
        Returns a list with prediction`s probabilities.
        '''
        data = self._prepare_data_for_inference(news_text)

        # Preprocess texts
        processed_data = self._preprocess_data(data)
        
        # Transform and predict
        prediction_results = self._transform_and_predict(processed_data)
        
        return prediction_results['predicted_probability'].tolist()

    def run_inference_file(self, file_path, output_path=None):
        '''
        Main method to run the full model inference process with the opportunity to save a resulting file.
        file_path: path to the file with news data.
        '''
        # Load and prepare data
        data = self._load_and_prepare_data(file_path)
        
        if (isinstance(data, pd.DataFrame) and len(data.columns) == 1 and 'content' in data.columns
            ) or isinstance(data, list):
            print("The df has only one column. The process will be executed by run_inference...")
            data_list = data['content'].tolist() if isinstance(data, pd.DataFrame) else data
            prediction_results = self.run_inference(data_list)
        else:
            # Preprocess text
            processed_data = self._preprocess_data(data)
            
            # Transform and predict
            prediction_results = self._transform_and_predict(processed_data)
            
            # Save the resulting dataframe
            if output_path:
                self._save_results(prediction_results, output_path, compressed=False, add_date=True)
        
        return prediction_results
    
