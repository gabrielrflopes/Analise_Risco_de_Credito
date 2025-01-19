from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(yt: np.array, yp: np.array):
    """
    Plota a matriz de confusão para avaliar o desempenho de um classificador.

    Parâmetros:
    yt (np.array): Os rótulos verdadeiros (ground truth) das classes.
    yp (np.array): Os rótulos previstos pelo classificador.

    Retorna:
    None: A função exibe a matriz de confusão em um gráfico.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    cf = confusion_matrix(yt, yp, normalize='all')

    cf_display = ConfusionMatrixDisplay(cf)

    cf_display.plot(ax=ax)

    ax.set_xlabel('Previsão', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title('Matriz de confusão - Baseline (RF)', fontsize=14)

    fig.tight_layout()