import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class CreateTrainTest:
    def __init__(self, df: pd.DataFrame, test_size: float = 0.3):
        self.df = df
        self.test_size = test_size

    def train_test_df(self) -> tuple:
        '''
        Separa o conjunto de dados original entre treino e teste.
        '''
        test_df = self.df.sample(frac=self.test_size, random_state=123)
        train_df = self.df.drop(labels=test_df.index)
        return train_df, test_df


class ModelTraining:
    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df
        self.X, self.y = None, None


    def create_X_y(self):
        """
        Esta função separa a matriz de atributos (X) da variável-alvo (y) em um DataFrame.
        """
        self.X = self.train_df.drop(['target_default'], axis=1)
        self.y = self.train_df['target_default']
        return self.X, self.y


    def train_val_split(self, train_size: float = 0.7) -> tuple:
        """
        Esta função divide os dados em conjuntos de treino e validação.
        """
        if self.X is None or self.y is None:
            self.X, self.y = self.create_X_y()
        return train_test_split(self.X, self.y, train_size=train_size, random_state=123)


    def val_model(self, clf, quite: bool = False):
        """
        Avalia o desempenho de um classificador utilizando validação cruzada.
        """

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        recall_scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='recall')

        if not quite:
            print(f'Recall: {recall_scores.mean():.3f} +/- {recall_scores.std():.3f}')

        return recall_scores.mean()


class CreateBaseline:
    def __init__(self, Xtr, Xts, ytr):
        self.Xtr = Xtr
        self.Xts = Xts
        self.ytr = ytr


    def baseline_model(self):
        """
        Instanciar e ajustar o modelo.
        """
        rf = RandomForestClassifier()
        rf.fit(np.array(self.Xtr), np.array(self.ytr))
        return rf.predict(np.array(self.Xts))


    def test_class_balance_methods(self, balance_methods: dict, quite: bool = False):
        """
        Treina e avalia um classificador utilizando métodos de balanceamento de classes.
        """
        from imblearn.pipeline import Pipeline

        rf = RandomForestClassifier()
        scores = []

        for name, bal in balance_methods.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('balance', bal),
                ('baseline_clf', rf)
            ])

            score = cross_val_score(pipeline, np.array(self.Xtr), np.array(self.ytr), cv=3, scoring='recall')
            scores.append(score.mean())

            if not quite:
                print(f'Recall w/ {name}: {round(score.mean(), 3)}')

        return scores