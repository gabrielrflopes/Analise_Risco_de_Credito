import os
import json
import pandas as pd

class LoadData:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_data(self):
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        
        directory = config['data_path']
        file_name = config['file_name']

        if not os.path.exists(directory):
            raise FileNotFoundError(f'Diretório não encontrado: {directory}')
        
        data_path = os.path.join(directory, file_name)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f'Arquivo não encontrado: {data_path}')

        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(data_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f'Extensão de arquivo não suportada: {file_extension}')

        return df