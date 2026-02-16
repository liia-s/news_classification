import os
from dotenv import load_dotenv
from model_inference import LinearModelInference

load_dotenv()

MAIN_PATH = os.getenv('MAIN_PATH')

# A dataframe with parsed news data
NEWS_DF_INPUT_PATH = os.path.join(MAIN_PATH, 'data')
# The saved linear model
PIPELINE_PATH = os.path.join(MAIN_PATH, 'linear_model/model.joblib')

# The folder for storing the outcomes
OUTPUT_PATH = os.path.join(MAIN_PATH, 'data/processed')


inference_pipeline = LinearModelInference(model_path=PIPELINE_PATH,
                                          content_column='summary',
                                          channel_column='channel')

# To process a text 
results = inference_pipeline.run_inference_file(file_path=NEWS_DF_INPUT_PATH,
                                                output_path=OUTPUT_PATH)
