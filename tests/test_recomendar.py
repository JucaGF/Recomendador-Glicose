import json

import numpy as np
import pandas as pd
import pytest

from glicose import ag, plano, recomendar
from glicose.ag import ConfigAG, Lambdas
from glicose.esquema import DIAS_PLANO, MACROS, dados_sinteticos
from glicose.simulador import ConfigRede, treinar_final

N_PARTICIPANTES = 6
CONFIG_PEQUENA_COM_OUTROS_PESOS = ConfigAG(
    tamanho_populacao=20,
    max_geracoes=5,
    lambdas=Lambdas(kcal=50_000.0, proteina=20_000.0, habito=250.0),
)
SEMENTE = 3
PLANOS_POR_ESTRATEGIA = 4
# A rede calcula em float32, e lotes de tamanhos diferentes podem mudar o último dígito.
TOLERANCIA_FLOAT32 = 1e-5
REFEICOES_COM_UMA_SUJA = 4


@pytest.fixture(scope="module")
def sinteticos():
    return dados_sinteticos(N_PARTICIPANTES, semente=0)


@pytest.fixture(scope="module")
def simulador(sinteticos):
    return treinar_final(
        sinteticos["refeicoes"], sinteticos["perfis"], "log1p", ConfigRede(epocas=5)
    )


@pytest.fixture(scope="module")
def recomendado(sinteticos, simulador, tmp_path_factory):
    perfil = sinteticos["perfis"].iloc[0]
    pasta = tmp_path_factory.mktemp("planos")
    impresso = recomendar.recomendar(
        perfil,
        sinteticos["cardapio"],
        lambda r: simulador.prever(perfil, r),
        SEMENTE,
        pasta,
        CONFIG_PEQUENA_COM_OUTROS_PESOS,
    )
    return perfil, impresso, pasta


def test_plano_lido_do_json_reproduz_a_aptidao_impressa(sinteticos, simulador, recomendado):
    perfil, impresso, pasta = recomendado
    cardapio = sinteticos["cardapio"]
    gravado = json.loads((pasta / "001.json").read_text())
    assert (pasta / "001.png").is_file()

    def simular(refeicoes):
        return simulador.prever(perfil, refeicoes)

    lido = recomendar.json_para_plano(gravado)
    aptidao_lida = ag.aptidao(
        lido, simular, perfil, cardapio, CONFIG_PEQUENA_COM_OUTROS_PESOS.lambdas
    )[0]
    do_ag = ag.rodar(CONFIG_PEQUENA_COM_OUTROS_PESOS, simular, perfil, cardapio, SEMENTE)

    assert aptidao_lida == pytest.approx(impresso["aptidao"], rel=TOLERANCIA_FLOAT32)
    assert do_ag.aptidao == pytest.approx(impresso["aptidao"], rel=TOLERANCIA_FLOAT32)


def test_d7_e_d8_recomendam_horario_e_caminhada_do_habito(sinteticos, recomendado):
    perfil, impresso, _ = recomendado
    habito = plano.habito(perfil, sinteticos["cardapio"])
    refeicoes = pd.DataFrame(impresso["refeicoes"])
    d7_d8 = (refeicoes["dia"] < DIAS_PLANO[-1]).to_numpy()

    np.testing.assert_array_equal(refeicoes["intervalo"][d7_d8], habito.intervalo[0, d7_d8])
    np.testing.assert_array_equal(refeicoes["caminhada"][d7_d8], habito.caminhada[0, d7_d8])


def test_todos_os_pratos_do_plano_existem_no_cardapio(sinteticos, recomendado):
    _, impresso, pasta = recomendado
    cardapio = sinteticos["cardapio"].set_index("prato")
    refeicoes = json.loads((pasta / "001.json").read_text())["refeicoes"]

    assert len(refeicoes) == plano.N_REFEICOES
    for refeicao in refeicoes:
        prato = cardapio.loc[refeicao["prato"]]
        assert prato["tipo"] == refeicao["tipo"]
        assert [prato[m] for m in MACROS] == [refeicao[m] for m in MACROS]
