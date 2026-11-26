"""Testes da validação metodológica (`validacao`).

Autores: Miguel Mochizuki Silva
"""

import numpy as np
import pandas as pd
import pytest

from glicose import simulador, validacao
from glicose.baselines_modelo import ALVO
from glicose.esquema import DIAS_HABITO, DIAS_PLANO, dados_sinteticos
from glicose.simulador import ConfigRede
from glicose.validacao import (
    analisar,
    com_perda,
    dobras,
    holm,
    ic_mediana,
    metricas_por_participante,
    p_wilcoxon,
    particao_conhecida,
    pearson_com_ic,
    rodar_protocolo,
)

N_PARTICIPANTES = 15
CONFIG_RAPIDA = ConfigRede(epocas=5, n_redes=2)
BOOTSTRAP_PEQUENO = 200


@pytest.fixture(scope="module")
def sinteticos():
    return dados_sinteticos(N_PARTICIPANTES, semente=0)


@pytest.fixture(scope="module")
def estudo(sinteticos):
    return simulador.particao_do_estudo(sinteticos["refeicoes"], sinteticos["perfis"])


def test_holm_corrige_na_ordem_certa_e_nunca_passa_de_um():
    ajustado = holm(np.array([0.01, 0.04, 0.03]))
    assert ajustado == pytest.approx([0.03, 0.06, 0.06])
    assert holm(np.array([0.6, 0.9])).max() <= 1.0


def test_ic_da_mediana_cobre_o_valor_verdadeiro_e_colapsa_em_constante():
    rng = np.random.default_rng(0)
    valores = np.random.default_rng(1).normal(5, 1, 200)
    mediana, inferior, superior = ic_mediana(valores, BOOTSTRAP_PEQUENO, rng)
    assert inferior < 5 < superior
    assert inferior <= mediana <= superior
    assert ic_mediana(np.full(30, 7.0), BOOTSTRAP_PEQUENO, rng) == (7.0, 7.0, 7.0)


def test_wilcoxon_pareado_aceita_zeros_e_detecta_efeito_consistente():
    assert p_wilcoxon(np.zeros(10)) == 1.0
    assert p_wilcoxon(np.arange(1.0, 21.0)) < 0.001


def test_metricas_por_participante_batem_com_a_conta_a_mao():
    # Participante 1, D9 completo: erros +2, -4, +6. Participante 2, D9 incompleto: erro +1.
    previsoes = pd.DataFrame(
        {
            "repeticao": 0,
            "participante": [1, 1, 1, 2],
            "dia": [9, 9, 9, 9],
            ALVO: [10.0, 10.0, 10.0, 10.0],
            "rede": [12.0, 6.0, 16.0, 11.0],
        }
    )
    metricas = metricas_por_participante(previsoes, "rede")
    assert metricas.loc[1, "mae"] == pytest.approx(4.0)
    assert metricas.loc[1, "rmse"] == pytest.approx(np.sqrt((4 + 16 + 36) / 3))
    assert metricas.loc[1, "vies"] == pytest.approx(4 / 3)
    assert metricas.loc[1, "erro_soma_d9"] == pytest.approx(4.0)
    assert np.isnan(metricas.loc[2, "erro_soma_d9"])


def test_pearson_de_serie_perfeita_vale_um_com_intervalo_estreito():
    previsao = np.arange(60.0)
    linhas = pd.DataFrame(
        {"participante": np.repeat(np.arange(6), 10), "previsao": previsao, "alvo": 2 * previsao}
    )
    r, inferior, superior = pearson_com_ic(linhas, BOOTSTRAP_PEQUENO, np.random.default_rng(0))
    assert r == pytest.approx(1.0)
    assert inferior > 0.99
    assert superior == pytest.approx(1.0)


def test_cenario_ii_nunca_poe_a_mesma_pessoa_no_treino_e_no_teste(estudo):
    particoes = set()
    for semente in range(4):
        testados = []
        for treino, teste in dobras(estudo, semente):
            participantes_teste = set(teste.refeicoes["participante"])
            assert set(treino.refeicoes["participante"]).isdisjoint(participantes_teste)
            assert treino.refeicoes["dia"].isin(DIAS_HABITO).all()
            testados.extend(participantes_teste)
        assert sorted(testados) == sorted(estudo.refeicoes["participante"].unique())
        particoes.add(tuple(testados))
    assert len(particoes) > 1


def test_cenario_i_treina_so_em_d1_a_d6_e_mantem_todos_os_participantes(estudo):
    treino, teste = particao_conhecida(estudo)
    assert treino.refeicoes["dia"].isin(DIAS_HABITO).all()
    assert not treino.refeicoes["dia"].isin(DIAS_PLANO).any()
    assert set(treino.refeicoes["participante"]) == set(teste.refeicoes["participante"])
    assert len(teste.refeicoes) == len(estudo.refeicoes)


def test_perda_trocada_so_vale_dentro_do_bloco():
    original = simulador.construir_modelo
    with com_perda("mse"):
        modelo = simulador.construir_modelo(CONFIG_RAPIDA)
        assert "mean_squared_error" in str(modelo.loss) or modelo.loss == "mse"
    assert simulador.construir_modelo is original


def test_protocolo_completo_gera_as_tabelas_no_formato_esperado(sinteticos):
    previsoes = rodar_protocolo(
        sinteticos["refeicoes"],
        sinteticos["perfis"],
        CONFIG_RAPIDA,
        repeticoes=2,
        semente=0,
        perdas=("mae", "mse"),
    )
    assert previsoes["dia"].isin(DIAS_PLANO).all()
    assert set(previsoes["cenario"]) == {"ii", "i"}
    assert set(previsoes.loc[previsoes["cenario"] == "i", "perda"]) == {"mae"}
    assert set(previsoes.loc[previsoes["perda"] == "mse", "cenario"]) == {"ii"}
    assert (previsoes["desvio"] >= 0).all()

    tabelas = analisar(previsoes, bootstrap=BOOTSTRAP_PEQUENO, semente=0)
    contrastes = tabelas["contrastes"]
    assert set(contrastes["metrica"]) == set(validacao.METRICAS)
    # 3 baselines em cada cenário, i contra ii e só a perda mse na ablação.
    assert (contrastes.groupby("metrica").size() == 8).all()
    assert (contrastes["p_holm"] >= contrastes["p"] - 1e-12).all()
    assert (contrastes["ic_inferior"] <= contrastes["ic_superior"]).all()
    assert set(tabelas["calibracao"]["cenario"]) == {"ii", "i"}
    assert not tabelas["decis"].empty
