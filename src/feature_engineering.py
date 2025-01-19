import pandas as pd

'''
Classe para ser usada após a limpeza dos dados
'''

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
   
    def create_lat_lon(self, verbose: bool = True) -> pd.DataFrame:
        """
        Cria colunas de latitude e longitude a partir da coluna 'lat_lon' no DataFrame.

        Args:
        df (pd.DataFrame): O DataFrame que contém a coluna 'lat_lon' a ser processada.
        verbose (bool): Se True, imprime mensagens sobre o progresso da função.

        Returns:
        pd.DataFrame: O DataFrame com as novas colunas 'latitude' e 'longitude' adicionadas.
        """

        # Criar os dois novos atributos
        self.df[['latitude', 'longitude']] = self.df['lat_lon'].str.strip('()').str.split(',', expand= True)

        # Transformando o tipo das colunas novas para float
        self.df['latitude'] = self.df['latitude'].astype(float)
        self.df['longitude'] = self.df['longitude'].astype(float)

        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            # Remover a coluna original e a última entrada criada
            self.df.drop(columns=['lat_lon'], axis=1, inplace=True)
            if verbose:
                print('Atributos "latitude" e "longitude" criados!')
        elif verbose:
            print('Erro: Variáveis não criadas.')

        return self


    def std_shipping_state(self) -> pd.DataFrame:
        """
        Padroniza as entradas da coluna 'shipping_state' do DataFrame.

        Args:
        df (pd.DataFrame): O DataFrame que contém a coluna 'shipping_state' a ser padronizada.

        Returns:
        pd.DataFrame: O DataFrame com a coluna 'shipping_state' padronizada.
        """
        # Padronizar as entradas em shipping state (BR-RS -> RS)
        self.df['shipping_state'] = self.df['shipping_state'].str.split('-', expand = True)[1]
        return self


    def apply_feature_engineering(self) -> pd.DataFrame:
        """
        Aplica engenharia de features ao DataFrame, criando as colunas 'latitude' e 'longitude'
        a partir da coluna 'lat_lon' e padronizando as entradas da coluna 'shipping_state'.
        
        Args:
        df (pd.DataFrame): O DataFrame a ser processado.
        
        Returns:
        pd.DataFrame: O DataFrame com as novas colunas de latitude e longitude e a coluna 
        'shipping_state' padronizada.
        """
        self.create_lat_lon()
        self.std_shipping_state()
        
        return self.df
    
    
    def get_feature_engineered_df(self) -> pd.DataFrame:
        return self.df