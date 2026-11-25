import numpy as np
import pandas as pd
import pytest

from glicose.esquema import DIAS_HABITO, ENTRADA_SIMULADOR, conformar, dados_sinteticos
from glicose.simulador import (
    ConfigRede,
    carregar,
    dividir,
    escolher_alvo,
    montar_atributos,
    resumir,
    salvar,
    separar_validacao,
    treinar_final,
    validar,
)

N_PARTICIPANTES = 15
IAUC_POR_GRAMA_CARBO = 1.5
N_REFEICOES_PLANO = 300
CAMINHADA_MAXIMA = 30
HORARIO_MAXIMO = 23.0
CONFIG_RAPIDA = ConfigRede(epocas=5)


@pytest.fixture(scope="module")
def sinteticos():
    return dados_sinteticos(N_PARTICIPANTES, semente=0)


@pytest.fixture(scope="module")
def simulador(sinteticos):
    return treinar_final(sinteticos["refeicoes"], sinteticos["perfis"], "iauc", CONFIG_RAPIDA)


@pytest.fixture
def plano(sinteticos):
    rng = np.random.default_rng(1)
    perfil = sinteticos["perfis"].iloc[0]
    refeicoes = sinteticos["refeicoes"]
    pratos = refeicoes[refeicoes["participante"] == perfil["participante"]]
    plano = pratos.sample(N_REFEICOES_PLANO, replace=True, random_state=rng).assign(
        horario=rng.uniform(0, HORARIO_MAXIMO, N_REFEICOES_PLANO),
        glicose_pre=perfil["glicose_jejum_cgm"],
        minutos_ativos_60=rng.uniform(0, CAMINHADA_MAXIMA, N_REFEICOES_PLANO),
    )
    return perfil, conformar(plano, ENTRADA_SIMULADOR)


def test_nenhum_participante_fica_no_treino_e_no_teste_do_mesmo_fold(sinteticos):
    testados = []
    for treino, teste in dividir(sinteticos["refeicoes"], sinteticos["perfis"]):
        participantes_treino = set(treino.refeicoes["participante"])
        participantes_teste = set(teste.refeicoes["participante"])
        assert participantes_treino.isdisjoint(participantes_teste)
        assert treino.refeicoes["dia"].isin(DIAS_HABITO).all()
        testados.extend(participantes_teste)

        ajuste, validacao = separar_validacao(treino, semente=0)
        participantes = treino.refeicoes["participante"].to_numpy()
        assert set(participantes[ajuste]).isdisjoint(participantes[validacao])
        assert len(ajuste) + len(validacao) == len(treino.refeicoes)
    assert sorted(testados) == sorted(sinteticos["perfis"]["participante"])


def test_rede_erra_menos_que_a_media_quando_a_iauc_vem_do_carbo(sinteticos):
    refeicoes = sinteticos["refeicoes"].assign(
        iauc_dexcom=lambda r: IAUC_POR_GRAMA_CARBO * r["carbo"]
    )
    sem_atividade = refeicoes["participante"] % 3 == 0
    refeicoes.loc[sem_atividade, "minutos_ativos_60"] = np.nan

    previsoes = validar(refeicoes, sinteticos["perfis"])
    tabela = resumir(previsoes)
    rede = f"rede_{escolher_alvo(tabela)}"

    assert np.isfinite(previsoes[rede]).all()
    assert tabela.loc[rede, "mae_d7_d9"] < tabela.loc["media", "mae_d7_d9"]


def test_prever_devolve_um_valor_por_refeicao_na_ordem_de_entrada(simulador, plano):
    perfil, refeicoes = plano
    previsto = simulador.prever(perfil, refeicoes)

    entradas = simulador.entradas(montar_atributos(refeicoes, pd.DataFrame([perfil])))
    referencia = simulador.para_iauc(simulador.modelo.predict(entradas, verbose=0))
    np.testing.assert_allclose(previsto, referencia, atol=1e-4)

    embaralhada = np.random.default_rng(2).permutation(N_REFEICOES_PLANO)
    previsto_embaralhado = simulador.prever(perfil, refeicoes.iloc[embaralhada])
    np.testing.assert_allclose(previsto_embaralhado, previsto[embaralhada], atol=1e-4)


def test_simulador_gravado_e_lido_preve_o_mesmo(simulador, plano, tmp_path):
    perfil, refeicoes = plano
    salvar(simulador, tmp_path)
    np.testing.assert_allclose(
        carregar(tmp_path).prever(perfil, refeicoes), simulador.prever(perfil, refeicoes)
    )
