import numpy as np
import pandas as pd

class CleanData:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_target_nan(self, verbose: bool = True):
        """
        Remove as linhas onde a variável alvo 'target_default' é NaN.
        """
        nan_idx = self.df.loc[self.df['target_default'].isna()].index
        self.df.drop(index = nan_idx, axis = 0, inplace = True)

        if verbose:
            print('Valores nulos removidos!')

        return self


    def remove_cols(self, verbose: bool = True) -> pd.DataFrame:
        """
        Remove colunas especificadas do DataFrame e imprime as novas dimensões.
        """
        # Definir colunas que se manterão
        cols_to_keep = ['target_default', 'score_1', 'score_2', 'score_3', 'score_4',
                    'score_5', 'score_6', 'risk_rate', 'last_amount_borrowed',
                    'last_borrowed_in_months', 'credit_limit', 'income', 'ok_since', 
                    'n_bankruptcies', 'n_defaulted_loans', 'n_accounts', 'n_issues', 
                    'application_time_in_funnel','external_data_provider_email_seen_before', 
                    'lat_lon', 'marketing_channel','reported_income', 'shipping_state']


        # Definir colunas que serão removidas
        cols_to_drop = [col for col in self.df.columns if col not in cols_to_keep]

        # Remover colunas com valores ausentes
        self.df = self.df.drop(columns=cols_to_drop, axis=1)

        if verbose:
            # Imprimir dimensões do conjunto limpo
            print(
                'Dimensão do conjunto limpo:',
                f'\nEntradas: {self.df.shape[0]}',
                f'\nVariáveis: {self.df.shape[1]}'
            )
            
            # Print das colunas removidas
            print(f'\nColunas removidas: {", ".join(cols_to_drop)}')
        
        return self


    def remove_inf(self, verbose: bool = True) -> pd.DataFrame:
        '''
        Remove exemplos com valores infinitos
        '''
        initial_len = len(self.df)

        self.df = self.df[~self.df.isin([np.inf, -np.inf]).any(axis=1)].reset_index(drop=True)
        
        if verbose and initial_len != len(self.df):
            print('Valores infinitos removidos.')
        return self


    def remove_neg_vals(self, verbose: bool = True) -> pd.DataFrame:
        '''
        Remove exemplos com valores negativos para external_data_provider_email_seen_before.
        '''
        initial_len = len(self.df)
        self.df = self.df[self.df['external_data_provider_email_seen_before'] >= 0].reset_index(drop=True)
        
        if verbose and initial_len != len(self.df):
            print('Valores negativos removidos.')
        return self


    def remove_nan_lat_lon(self, verbose: bool = True) -> pd.DataFrame:
        """
        Remove entradas ausentes na coluna lat_lon.
        """
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['lat_lon']).reset_index(drop=True)

        if verbose and initial_len != len(self.df):
            print('Valores NaN na coluna lat_lon removidos')
        return self


    def remove_outliers(
            self,
            columns: list,
            percentile: float = .8,
            verbose: bool = True
            ) -> pd.DataFrame:
        """
        Remove outliers das colunas especificadas com base em um percentil especificado.
        """
        initial_len = len(self.df)
        for column in columns:
            # Calcula o valor limite para cada coluna
            threshold = self.df[column].quantile(percentile)
            # Remove outliers para cada coluna
            self.df = self.df[self.df[column] <= threshold]
        
        # Reinicia o índice
        self.df.reset_index(drop=True, inplace=True)
        
        if verbose:
            removed = initial_len - len(self.df)
            print(f"Removidos {removed} outliers acima do percentil {percentile:.2f}.")
            print(f"Nova forma do DataFrame: {self.df.shape}")        
        return self
    

    def apply_all_clean_functions(self, outlier_cols: list = ['income', 'reported_income'], outlier_percentile: float = 0.95, verbose: bool = True):
        """
        Aplica todas as transformações de limpeza em uma sequência predefinida.

        Args:
            outlier_columns (list): Lista de colunas para remover outliers. Se None, não remove outliers.
            outlier_percentile (float): Percentil para remoção de outliers. Padrão é 0.8.
            verbose (bool): Se True, imprime mensagens de status. Padrão é True.

        Returns:
            pd.DataFrame: DataFrame limpo após todas as transformações.
        """
        self.remove_target_nan(verbose)
        self.remove_cols(verbose)
        self.remove_inf(verbose)
        self.remove_neg_vals(verbose)
        self.remove_nan_lat_lon(verbose)

        if verbose:
            print("Todas as transformações de limpeza foram aplicadas.")
            print(f"Forma final do DataFrame: {self.df.shape}")
        
        return self.df

    def get_clean_data(self) -> pd.DataFrame:
        return self.df