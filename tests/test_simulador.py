import numpy as np
import pandas as pd
import pytest

from glicose.baselines_modelo import Particao, media_participante
from glicose.esquema import DIAS_HABITO, ENTRADA_SIMULADOR, conformar, dados_sinteticos
from glicose.simulador import (
    MODELOS,
    N_FOLDS,
    ConfigRede,
    carregar,
    dividir,
    montar_atributos,
    particao_do_estudo,
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
EPOCAS_RAPIDAS = 5
CONFIG_RAPIDA = ConfigRede(epocas=EPOCAS_RAPIDAS, n_redes=2)
CONFIG_DUAS_REDES = ConfigRede(n_redes=2)
IAUC_QUE_NAO_PODE_VAZAR = 1e6
FIBRA_IMPOSSIVEL_G = 2000.0
TOLERANCIA_IAUC = 1e-4
# A rede faz as contas em float32, que guarda uns 7 dígitos da iAUC em mg/dL·min.
TOLERANCIA_FLOAT32 = 1e-5


@pytest.fixture(scope="module")
def sinteticos():
    return dados_sinteticos(N_PARTICIPANTES, semente=0)


@pytest.fixture(scope="module")
def simulador(sinteticos):
    return treinar_final(
        sinteticos["refeicoes"], sinteticos["perfis"], EPOCAS_RAPIDAS, CONFIG_RAPIDA
    )


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

    previsoes, melhores_epocas = validar(refeicoes, sinteticos["perfis"], CONFIG_DUAS_REDES)
    tabela = resumir(previsoes)

    assert tabela.loc["rede", "mae_d7_d9"] < tabela.loc["media", "mae_d7_d9"]
    assert len(melhores_epocas) == N_FOLDS * CONFIG_DUAS_REDES.n_redes
    assert all(1 <= epoca <= CONFIG_DUAS_REDES.epocas for epoca in melhores_epocas)


def test_tabela_separa_os_dias_de_teste_dos_de_habito():
    real = [100.0, 200.0, 100.0, 300.0]
    erro = [10.0, -10.0, 100.0, -100.0]
    previsoes = pd.DataFrame(
        {
            "dia": [7, 9, 1, 6],
            "fold": [0, 1, 0, 1],
            "iauc_dexcom": real,
            **{nome: real for nome in MODELOS},
            "rede": np.add(real, erro),
        }
    )
    tabela = resumir(previsoes)

    assert tabela.loc["rede", "mae_d7_d9"] == 10.0
    assert tabela.loc["rede", "mae_d1_d6"] == 100.0


def test_media_do_participante_usa_so_d1_a_d6():
    dias = np.arange(1, 10)
    iauc = np.where(np.isin(dias, DIAS_HABITO), dias * 100.0, IAUC_QUE_NAO_PODE_VAZAR)
    refeicoes = pd.DataFrame({"participante": 1, "dia": dias, "iauc_dexcom": iauc})
    teste = Particao(refeicoes, pd.DataFrame(index=refeicoes.index))

    previsto = media_participante(teste, teste)

    np.testing.assert_allclose(previsto, np.mean([100, 200, 300, 400, 500, 600]))


def test_prever_devolve_um_valor_por_refeicao_na_ordem_de_entrada(simulador, plano):
    perfil, refeicoes = plano
    previsto = simulador.prever(perfil, refeicoes)
    assert previsto.shape == (N_REFEICOES_PLANO,)

    entradas = simulador.entradas(montar_atributos(refeicoes, pd.DataFrame([perfil])))
    referencia = simulador.para_iauc(simulador.modelo.predict(entradas, verbose=0))
    np.testing.assert_allclose(previsto, referencia, atol=TOLERANCIA_IAUC, rtol=TOLERANCIA_FLOAT32)

    embaralhada = np.random.default_rng(2).permutation(N_REFEICOES_PLANO)
    previsto_embaralhado = simulador.prever(perfil, refeicoes.iloc[embaralhada])
    np.testing.assert_allclose(
        previsto_embaralhado, previsto[embaralhada], atol=TOLERANCIA_IAUC, rtol=TOLERANCIA_FLOAT32
    )


def test_prever_monta_as_mesmas_entradas_que_o_treino(simulador, sinteticos):
    estudo = particao_do_estudo(sinteticos["refeicoes"], sinteticos["perfis"])
    perfil = sinteticos["perfis"].iloc[1]
    pessoa = estudo.recortar(
        (estudo.refeicoes["participante"] == perfil["participante"]).to_numpy()
    )

    np.testing.assert_allclose(
        simulador.prever(perfil, conformar(pessoa.refeicoes, ENTRADA_SIMULADOR)),
        simulador.prever_atributos(pessoa.atributos),
    )


def test_fibra_acima_do_carbo_vale_como_ausente_no_estudo(sinteticos):
    refeicoes = sinteticos["refeicoes"].copy()
    refeicoes.loc[refeicoes.index[:3], "fibra"] = FIBRA_IMPOSSIVEL_G

    estudo = particao_do_estudo(refeicoes, sinteticos["perfis"])

    impossivel = estudo.refeicoes.index.isin(refeicoes.index[:3])
    assert estudo.atributos.loc[impossivel, "fibra"].isna().all()
    assert estudo.atributos.loc[~impossivel, "fibra"].notna().all()


def test_simulador_gravado_e_lido_preve_o_mesmo(simulador, plano, tmp_path):
    perfil, refeicoes = plano
    salvar(simulador, tmp_path)
    np.testing.assert_allclose(
        carregar(tmp_path).prever(perfil, refeicoes), simulador.prever(perfil, refeicoes)
    )
