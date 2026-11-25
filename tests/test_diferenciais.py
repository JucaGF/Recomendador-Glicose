import numpy as np
import pandas as pd
import pytest

from glicose import ag, diferenciais
from glicose.ag import ConfigAG, Lambdas
from glicose.esquema import MACROS, dados_sinteticos

N_PARTICIPANTES = 3
SEMENTE = 0
# Cada pessoa sente um macro diferente. Assim o melhor plano de uma não é o melhor da outra.
MACRO_SENTIDO = {1: "carbo", 2: "gordura", 3: "fibra"}
SENSIBILIDADE = 30.0
CONFIG_PEQUENA = ConfigAG(
    tamanho_populacao=20, max_geracoes=40, paciencia=40, lambdas=Lambdas(0.0, 0.0, 0.0)
)


@pytest.fixture(scope="module")
def sinteticos():
    dados = dados_sinteticos(N_PARTICIPANTES, semente=SEMENTE)
    dados["perfis"] = dados["perfis"].set_index("participante", drop=False)
    return dados


def simulador_de(perfil: pd.Series):
    macro = MACRO_SENTIDO[perfil["participante"]]

    def simular(refeicoes: pd.DataFrame) -> np.ndarray:
        return SENSIBILIDADE * refeicoes[macro].to_numpy()

    return simular


@pytest.fixture(scope="module")
def matriz(sinteticos):
    perfis, cardapio = sinteticos["perfis"], sinteticos["cardapio"]
    planos = {
        p: ag.rodar(CONFIG_PEQUENA, simulador_de(perfil), perfil, cardapio, SEMENTE).melhor
        for p, perfil in perfis.iterrows()
    }
    return diferenciais.matriz_personalizacao(planos, perfis, cardapio, simulador_de)


def test_proprio_plano_e_o_melhor_em_cada_perfil(matriz):
    diagonal = np.diag(matriz)
    assert (diagonal <= matriz.min(axis=0).to_numpy()).all()
    assert diferenciais.diagonal_minima(matriz).all()
    assert (matriz.to_numpy() > diagonal).any(), "os planos precisam diferir entre as pessoas"


def test_plano_do_grupo_nunca_ganha_do_proprio(matriz):
    um_grupo_so = pd.Series("saudavel", index=matriz.index)
    perdas = diferenciais.perda_plano_do_grupo(matriz, um_grupo_so)
    assert (perdas["perda"] >= 0).all()
    assert perdas["perda"].min() == 0, "o dono do plano do grupo não perde nada"


def test_sem_travas_o_ag_sai_do_cardapio_onde_a_rede_erra(sinteticos):
    perfil = sinteticos["perfis"].iloc[0]
    cardapio = sinteticos["cardapio"]
    maximos = cardapio[list(MACROS)].max().to_numpy()
    limites = (np.zeros(len(diferenciais.GENES_LIVRES)), np.append(maximos, 30.0))

    def simular(refeicoes):
        return SENSIBILIDADE * refeicoes["carbo"].to_numpy()

    genes = diferenciais.ag_sem_travas(CONFIG_PEQUENA, simular, perfil, cardapio, limites, SEMENTE)
    carbo = genes[:, diferenciais.GENES_LIVRES.index("carbo")]

    assert (genes >= limites[0]).all() and (genes <= limites[1]).all()
    assert (carbo < cardapio["carbo"].min()).all()
