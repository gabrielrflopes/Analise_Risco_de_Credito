import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class Imputer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
            
    def numerical_imputer(self, verbose: bool = True) -> pd.DataFrame:
        # Instanciar o objeto para imputar valores da mediana
        num_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

        # Selecionando as colunas para preencher com zero
        cols_to_fill_0 = ['n_defaulted_loans', 'n_bankruptcies', 'last_amount_borrowed', 'last_borrowed_in_months']
        self.df.loc[:, cols_to_fill_0] = self.df.loc[:, cols_to_fill_0].fillna(value = 0)

        # Filtrando os atributos numéricos
        num_cols = self.df.select_dtypes(exclude = 'O').columns
        # Filtrar colunas numéricas com valores NaN
        nan_cols = [col for col in num_cols if self.df[col].isna().any()]
        
        if nan_cols:
            # Ajustar e transformar apenas colunas com valores NaN
            num_imputer = num_imputer.fit(self.df.loc[:, nan_cols])
            self.df.loc[:, nan_cols] = num_imputer.transform(self.df.loc[:, nan_cols])
            if verbose:
                print(f"Colunas imputadas: {', '.join(nan_cols)}")
        elif verbose:
            print("Nenhuma coluna numérica com valores NaN encontrada.")
        
        return self


    def cat_imputer(self, verbose: bool = True) -> pd.DataFrame:
        # Filtrar colunas categóricas
        cat_cols = self.df.select_dtypes(include='O').columns
        
        # Find columns with NaN values
        nan_cols = [col for col in cat_cols if self.df[col].isna().any()]
        
        if verbose:
            print(f'Colunas categóricas: {cat_cols}')
            print(f'COlunas com NaN: {nan_cols}')
        
        # Impute missing values for each categorical column
        for col in nan_cols:
            # Instanciate imputer for each column
            cat_imputer = SimpleImputer(missing_values=None, strategy='most_frequent')
            
            # Reshape the column to 2D array for fit_transform
            self.df[col] = cat_imputer.fit_transform(self.df[col].values.reshape(-1, 1)).ravel()
        
        return self


    def apply_imputers(self, verbose: bool = True) -> pd.DataFrame:
        
        self.numerical_imputer(verbose)
        self.cat_imputer(verbose)

        if verbose:
            print('Imputers aplicados com sucesso.')
        return self.df
    
    def get_imputed_df(self) -> pd.DataFrame:
        return self.df