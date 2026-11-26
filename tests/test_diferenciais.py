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


def simulador_cego(perfil: pd.Series):
    return ag.simulador_falso


@pytest.fixture(scope="module")
def iauc(sinteticos):
    perfis, cardapio = sinteticos["perfis"], sinteticos["cardapio"]
    dias = diferenciais.dias_viaveis(cardapio, perfis)
    return {
        nome: diferenciais.iauc_por_pessoa(dias, perfis, cardapio, de)
        for nome, de in (("perfil", simulador_de), ("cego", simulador_cego))
    }


def test_proprio_plano_e_o_melhor_em_cada_perfil(iauc):
    matriz = diferenciais.matriz_personalizacao(iauc["perfil"])
    assert diferenciais.diagonal_estrita(matriz).all()
    assert (diferenciais.perda_de_outro_plano(matriz) > 0).all()
    assert diferenciais.concordancia(iauc["perfil"]) < 1


def test_simulador_que_ignora_o_perfil_nao_personaliza(iauc):
    matriz = diferenciais.matriz_personalizacao(iauc["cego"])
    um_grupo_so = pd.Series("saudavel", index=matriz.index)
    assert not diferenciais.diagonal_estrita(matriz).any()
    assert (diferenciais.perda_de_outro_plano(matriz) == 0).all()
    assert diferenciais.concordancia(iauc["cego"]) == pytest.approx(1.0)
    assert (diferenciais.perda_plano_do_grupo(iauc["cego"], um_grupo_so)["perda"] == 0).all()


def test_plano_feito_para_os_outros_custa_a_quem_sente_outro_macro(iauc):
    um_grupo_so = pd.Series("saudavel", index=iauc["perfil"].index)
    perdas = diferenciais.perda_plano_do_grupo(iauc["perfil"], um_grupo_so)
    assert (perdas["perda"] > 0).all()


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
