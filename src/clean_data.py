# Importações básicas
import pandas as pd
import numpy as np
# Importações para processamento de dados
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# Funções do projeto
from imputers import Imputer

# Configuração para evitar avisos
import warnings
warnings.filterwarnings('ignore')

# Configuração para reprodutibilidade
np.random.seed(42)



def process_data(df):
    """
    Agrega todas as etapas de limpeza e processamento de dados.
    
    Args:
    df (pd.DataFrame): O DataFrame de entrada bruto.
    
    Returns:
    pd.DataFrame: O DataFrame processado e limpo com colunas de engenharia de features.
    """
    # Remove valores NaN da variável alvo
    df_tgt = remove_target_nan(df, verbose=False)
    
    # Aplica codificação de rótulos à variável alvo
    df_le = apply_label_encoder(df_tgt, verbose=False)

    # Imputa valores ausentes
    df_imputed = impute_nans(df_le, verbose=False)

    # Remove colunas desnecessárias
    df_cols = remove_cols(df_imputed)

    # Remove valores nulos na coluna lat_lon
    df_latlon = remove_nan_lat_lon(df_cols)

    # Remove valores negativos em external_data_provider_email_seen_before
    df_neg = remove_neg_vals(df_latlon)

    # Remove valores infinitos
    df_inf = remove_inf(df_neg)

    # Aplica feature engineering
    df_fe = apply_feature_engineering(df_inf)

    return df_fe