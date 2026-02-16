import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

def checking_or_writing_status(name_values, 
                               value=True, 
                               mode='c', 
                               period=10, 
                               name_file='status.json'):
    """
    The function checks or updates the script execution status in the JSON file.

    name_values: The key (or list of keys) for which the value is checked or updated.
    value: The value to be written (used only in "w" mode).
    mode: "c" - checking the value, "w" - recording the value.
    period: The period in seconds to check (in "c" mode).
    name_file: The name of the JSON file to store the status.
    
    """
    
    PATH_INFO = os.getenv('PATH_INFO')

    if not PATH_INFO:
        raise EnvironmentError('he PATH_INFO environment variable is NOT set')

    path_file = os.path.join(PATH_INFO, name_file)

    # Value verification
    if mode == 'c':
        while True:
            if not os.path.exists(path_file):
                raise FileNotFoundError(f'The {path_file} file was not found')

            # Reading the JSON file
            with open(path_file, 'r', encoding='utf-8') as file_json:
                buf_json = json.load(file_json)

            # Checking the values (there may be one key or a list of keys)
            if isinstance(name_values, list):
                for_sum_val = [buf_json.get(val, False) for val in name_values]
                total_sum_name_values = len(name_values)
            else:
                for_sum_val = [buf_json.get(name_values, False)]
                total_sum_name_values = 1

            # If all keys have the required values (True), we continue execution
            if sum(for_sum_val) == total_sum_name_values:
                print(f'- The key(s) "{name_values}" has been updated')
                break
            else:
                print(f' - Waiting for the key(s) "{name_values}" to be updated. Checking every  {period} seconds')
                time.sleep(period)

    elif mode == 'w':
        if not os.path.exists(path_file):
            buf_json = {}
        else:
            with open(path_file, 'r', encoding='utf-8') as file_json:
                buf_json = json.load(file_json)

        print(f'- Updating the value of the key "{name_values}" to "{value}"')

        # Updating the value by key
        buf_json[name_values] = value

        # Writing the updated value to a file
        with open(path_file, 'w', encoding='utf-8') as file_json:
            json.dump(buf_json, file_json, indent=4, ensure_ascii=False)
