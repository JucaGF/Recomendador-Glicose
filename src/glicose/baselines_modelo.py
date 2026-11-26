"""Modelos de comparação para o simulador.

Cada baseline recebe as partições de treino e teste e devolve a iAUC prevista do
teste.

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from glicose.esquema import DIAS_HABITO

ALVO = "iauc_dexcom"

XGBOOST_ARVORES = 300
XGBOOST_PROFUNDIDADE = 3
XGBOOST_TAXA_APRENDIZADO = 0.05


@dataclass(frozen=True)
class Particao:
    """Refeições e atributos de um mesmo recorte do estudo, com as linhas alinhadas."""
    refeicoes: pd.DataFrame
    atributos: pd.DataFrame

    @property
    def iauc(self) -> np.ndarray:
        """iAUC medida de cada refeição."""
        return self.refeicoes[ALVO].to_numpy()

    def recortar(self, linhas: np.ndarray) -> "Particao":
        """Partição só com as linhas da posição indicada."""
        return Particao(self.refeicoes.iloc[linhas], self.atributos.iloc[linhas])


def media_participante(treino: Particao, teste: Particao) -> np.ndarray:
    """Média da iAUC de D1 a D6 de cada participante, repetida para todas as suas refeições."""
    habito = teste.refeicoes[teste.refeicoes["dia"].isin(DIAS_HABITO)]
    media = habito.groupby("participante")[ALVO].mean()
    return media.reindex(teste.refeicoes["participante"]).to_numpy()


def linear_carbo(treino: Particao, teste: Particao) -> np.ndarray:
    """Regressão linear da iAUC no carbo, sem mais atributos."""
    modelo = LinearRegression().fit(treino.atributos[["carbo"]], treino.iauc)
    return modelo.predict(teste.atributos[["carbo"]])


def xgboost(treino: Particao, teste: Particao) -> np.ndarray:
    """XGBoost com perda absoluta, treinado em todos os atributos."""
    modelo = XGBRegressor(
        n_estimators=XGBOOST_ARVORES,
        max_depth=XGBOOST_PROFUNDIDADE,
        learning_rate=XGBOOST_TAXA_APRENDIZADO,
        objective="reg:absoluteerror",
    )
    return modelo.fit(treino.atributos, treino.iauc).predict(teste.atributos)


BASELINES = {"media": media_participante, "linear": linear_carbo, "xgboost": xgboost}
