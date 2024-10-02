import os
import json
import pandas as pd

def load_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def load_data(config):
    directory = config['path']
    file_name = config['file_name']

    if not os.path.exists(directory):
        raise FileNotFoundError(f'Diretório não encontrado: {directory}')
    
    data_path = os.path.join(directory, file_name)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f'Arquivo não encontrado: {data_path}')

    file_extension = os.path.splitext(file_name)[1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(data_path)
    if file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)

    return df