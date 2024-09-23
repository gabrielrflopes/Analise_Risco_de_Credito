# Importações básicas
import pandas as pd
import numpy as np

# Importações para processamento de dados
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Importação para manipulação de tempo (usada em apply_feature_engineering)
import time

# Configuração para evitar avisos
import warnings
warnings.filterwarnings('ignore')

# Configuração para reprodutibilidade
np.random.seed(42)


def remove_target_nan(df: pd.DataFrame, verbose=True):
    """
    Remove as linhas onde a variável alvo 'target_default' é NaN.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        verbose (bool): Se True, imprime mensagem de status. Padrão é True.
    
    Returns:
        pd.DataFrame: DataFrame com as linhas de alvo NaN removidas.
    """
    df_clean = df.copy()

    nan_idx = df_clean.loc[df_clean['target_default'].isna()].index

    df_clean.drop(index = nan_idx, axis = 0, inplace = True)
    if verbose:
        print('Valores nulos removidos!')

    return df_clean

def apply_label_encoder(df: pd.DataFrame, verbose=True):
    """
    Aplica LabelEncoder à coluna 'target_default'.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        verbose (bool): Se True, imprime mensagem de status. Padrão é True.
    
    Returns:
        pd.DataFrame: DataFrame com a variável alvo codificada.
    """
    df_clean = df.copy()

    le = LabelEncoder()
    le.fit(df_clean['target_default'])

    df_clean['target_default'] = le.transform(df_clean['target_default'])
    
    if verbose and [0, 1] in df_clean['target_default'].unique():
        print('LabelEncoder aplicado!')

    return df_clean

def remove_cols(df: pd.DataFrame, cols_to_drop: list, verbose=True) -> pd.DataFrame:
    """
    Remove colunas especificadas do DataFrame e imprime as novas dimensões.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        cols_to_drop (list): Lista de nomes de colunas a serem removidas.
        verbose (bool): Se True, imprime mensagens de status. Padrão é True.
    
    Returns:
        pd.DataFrame: DataFrame com as colunas especificadas removidas.
    """
    # Criar cópia para backup
    df_cleaned = df.copy()

    # Remover colunas com valores ausentes
    df_cleaned = df.drop(columns=cols_to_drop, axis=1)
    # Remover entradas ausentes na coluna lat_lon
    idx = df_cleaned.loc[df_cleaned['lat_lon'].isna()].index
    df_cleaned = df_cleaned.drop(index = idx, axis = 0)
    df_cleaned = df_cleaned.reset_index(drop = True)
    
    if verbose:
        # Imprimir dimensões do conjunto limpo
        print(
            'Dimensão do conjunto limpo:',
            f'\nEntradas: {df_cleaned.shape[0]}',
            f'\nVariáveis: {df_cleaned.shape[1]}'
        )
        
        # Print das colunas removidas
        print(f'\nColunas removidas: {", ".join(cols_to_drop)}')
    
    return df_cleaned

def numerical_imputer(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Imputa valores ausentes em colunas numéricas usando a estratégia da mediana.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        verbose (bool): Se True, imprime mensagens de status. Padrão é True.
    
    Returns:
        pd.DataFrame: DataFrame com valores numéricos imputados.
    """
    df_num = df.copy()
    # Instanciar o objeto para imputar valores da mediana
    num_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

    # Selecionando as colunas para preencher com zero
    cols_to_fill_0 = ['last_amount_borrowed', 'last_borrowed_in_months', 'n_defaulted_loans', 'n_bankruptcies']
    df_num.loc[:, cols_to_fill_0] = df_num.loc[:, cols_to_fill_0].fillna(value = 0)

    # Filtrando os atributos numéricos
    num_cols = df_num.select_dtypes(exclude = 'O').columns
    # Filtrar colunas numéricas com valores NaN
    nan_cols = [col for col in num_cols if df_num[col].isna().any()]
    
    if nan_cols:
        # Ajustar e transformar apenas colunas com valores NaN
        num_imputer = num_imputer.fit(df_num.loc[:, nan_cols])
        df_num.loc[:, nan_cols] = num_imputer.transform(df_num.loc[:, nan_cols])
        if verbose:
            print(f"Colunas imputadas: {', '.join(nan_cols)}")
    elif verbose:
        print("Nenhuma coluna numérica com valores NaN encontrada.")
    
    return df_num

def cat_imputer(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Imputa valores ausentes em colunas categóricas usando a estratégia do valor mais frequente.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        verbose (bool): Se True, imprime mensagens de status. Padrão é True.
    
    Returns:
        pd.DataFrame: DataFrame com valores categóricos imputados.
    """
    df_cat = df.copy()
    # Instanciando o objeto para imputar valores baseados na moda
    cat_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

    # Filtro para variáveis categóricas
    cat_cols = df_cat.select_dtypes(include = 'O').columns

    if verbose:
        print(f'Colunas categóricas: {cat_cols}')
    nan_cols = [col for col in cat_cols if df_cat[col].isna().any()]

    if nan_cols:
        # Ajustar e transformar apenas colunas com valores NaN
        cat_imputer = cat_imputer.fit(df_cat.loc[:, nan_cols])
        df_cat.loc[:, nan_cols] = cat_imputer.transform(df_cat.loc[:, nan_cols])
        if verbose:
            print(f"Colunas imputadas: {', '.join(nan_cols)}")
    elif verbose:
        print("Nenhuma coluna categórica com valores NaN encontrada.")

    return df_cat

def impute_nans(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Imputa valores ausentes em colunas numéricas e categóricas.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        verbose (bool): Se True, imprime mensagens de status. Padrão é True.
    
    Returns:
        pd.DataFrame: DataFrame com todos os valores ausentes imputados.
    """
    df_proc = df.copy()
    # Imputar atributos numéricos
    df_proc = numerical_imputer(df_proc, verbose=verbose)
    # Imputar atributos categóricos
    df_proc = cat_imputer(df_proc, verbose=verbose)

    return df_proc

def apply_feature_engineering(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Aplica engenharia de features para criar novos atributos e modificar os existentes.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        verbose (bool): Se True, imprime mensagens de status. Padrão é True.
    
    Returns:
        pd.DataFrame: DataFrame com novos atributos engenheirados.
    """
    import time
    df_fe = df.copy()

    # Criar os dois novos atributos
    df_fe[['latitude', 'longitude']] = df_fe['lat_lon'].str.strip('()').str.split(',', expand= True)

    # Transformando o tipo das colunas novas para float
    df_fe['latitude'] = df_fe['latitude'].astype(float)
    df_fe['longitude'] = df_fe['longitude'].astype(float)

    time.sleep(2)
    if 'latitude' in df_fe.columns and 'longitude' in df_fe.columns:
        # Remover a coluna original e a última entrada criada
        df_fe.drop(columns=['lat_lon'], axis=1, inplace=True)
        if verbose:
            print('Atributos "latitude" e "longitude" criados!')
    elif verbose:
        print('Erro: Variáveis não criadas.')

    # Padronizar as entradas em shipping state (BR-RS -> RS)
    df_fe['shipping_state'] = df_fe['shipping_state'].str.split('-', expand = True)[1]

    return df_fe

def process_data(df):
    """
    Agrega todas as etapas de limpeza e processamento de dados.
    
    Args:
    df (pd.DataFrame): O DataFrame de entrada bruto.
    
    Returns:
    pd.DataFrame: O DataFrame processado e limpo com colunas de engenharia de features.
    """
    # Remove valores NaN da variável alvo
    df = remove_target_nan(df, verbose=False)
    
    # Aplica codificação de rótulos à variável alvo
    df = apply_label_encoder(df, verbose=False)
    
    # Remove colunas desnecessárias
    cols_to_drop = [
        'target_fraud', 'external_data_provider_fraud_score',
        'ids', 'email', 'external_data_provider_first_name',
        'profile_phone_number', 'profile_tags', 'facebook_profile',
        'user_agent', 'state', 'zip', 'job_name', 'channel',
        'shipping_zip_code', 
        'external_data_provider_credit_checks_last_month',
        'application_time_applied', 'reason', 'real_state', 
        'external_data_provider_credit_checks_last_year',
        'external_data_provider_credit_checks_last_2_year'
        ]
    df = remove_cols(df, cols_to_drop, verbose=False)
    
    # Imputa valores ausentes
    df = impute_nans(df, verbose=False)
    
    # Aplica feature engineering
    df = apply_feature_engineering(df, verbose=False)
        
    return df