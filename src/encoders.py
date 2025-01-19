import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


class Encoders:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()


    def apply_label_encoder(self, verbose=True):
        """
        Aplica LabelEncoder à coluna 'target_default'.
        """
        le = LabelEncoder()
        le.fit(self.df['target_default'])

        self.df['target_default'] = le.transform(self.df['target_default'])
        
        if verbose and [0, 1] in self.df['target_default'].unique():
            print('LabelEncoder aplicado!')

        return self
    

    def elbow_method(self, 
                     cols: list = ['marketing_channel', 'score_1', 'score_2', 'shipping_state'], 
                     max_clusters: int = 10) -> None:
        """
        Aplica o método do cotovelo para determinar o número ideal de clusters.

        Parâmetros:
        df (pd.DataFrame): O DataFrame contendo os dados a serem transformados.
        cols (list): Lista de colunas a serem analisadas.
        max_clusters (int): Número máximo de clusters a serem testados.
        """
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        axs = ax.flatten() # Para que os índices dos axes sejam 

        for idx, col in enumerate(cols):
            # Convertendo os valores categóricos em numéricos
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col]) # e.g df['marketing_channel] = le.fit_transform(df['marketing_channel])
            
            wcss = []
            # Realizar os testes com KMeans para cada coluna, determinando ao final a inércia e anexando à lista wcss
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, random_state=123)
                kmeans.fit(self.df[[col]])
                wcss.append(kmeans.inertia_)
            
            # Plotando o gráfico do método do cotovelo
            axs[idx].plot(range(1, max_clusters + 1), wcss, marker='o')
            axs[idx].set_title(f'Método do cotovelo para {col}')
            axs[idx].set_xlabel('Número de clusters')
            axs[idx].set_ylabel('WCSS')

        fig.tight_layout()


    def apply_cluster_encoding(self, n_clusters: int = 3, verbose: bool = True) -> pd.DataFrame:
        """
        Aplica codificação de cluster às colunas categóricas especificadas usando KMeans.
        """
        cols_to_encode = ['score_1', 'score_2', 'shipping_state', 'marketing_channel']
        
        for col in cols_to_encode:
            # Converter os valores categóricos em numéricos
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])

            # Aplicar KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=123)
            self.df[col + '_cluster'] = kmeans.fit_predict(self.df[[col]])

            if verbose and (col + '_cluster') in self.df.columns:
                print(f'Coluna {col + "_cluster"} criada.')

        # Dropping the original columns after encoding
        self.df.drop(columns=cols_to_encode, inplace=True)

        return self
    
    def apply_encoders(self, n_clusters: int = 3, verbose: bool = True) -> pd.DataFrame:
        """
        Aplica codificadores ao DataFrame, incluindo a codificação de rótulos e a codificação de clusters.

        Parâmetros:
        n_clusters (int): O número de clusters a serem utilizados na codificação de clusters. O padrão é 3.

        Retorna:
        pd.DataFrame: O DataFrame com as colunas codificadas.
        """
        self.apply_label_encoder(verbose)
        self.apply_cluster_encoding(n_clusters, verbose)
        return self.df
    

    def get_encoded_df(self):
        return self.df