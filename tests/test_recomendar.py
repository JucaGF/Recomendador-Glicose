"""Testes da recomendação (`recomendar`) e das camadas de avaliação (`experimentos` e
`figuras`).

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from glicose import ag, experimentos, figuras, plano, recomendar
from glicose.ag import ConfigAG, Lambdas
from glicose.esquema import DIAS_PLANO, MACROS, TIPOS_REFEICAO, dados_sinteticos
from glicose.simulador import ConfigRede, treinar_final

N_PARTICIPANTES = 6
CONFIG_PEQUENA_COM_OUTROS_PESOS = ConfigAG(
    tamanho_populacao=20,
    max_geracoes=5,
    lambdas=Lambdas(kcal=50_000.0, proteina=20_000.0, habito=250.0),
)
SEMENTE = 3
SEMENTES_PEQUENAS = 2
# A rede calcula em float32, e lotes de tamanhos diferentes podem mudar o último dígito.
TOLERANCIA_FLOAT32 = 1e-5
REFEICOES_COM_UMA_SUJA = 4
PRATOS_POR_TIPO_NA_BUSCA_EXAUSTIVA = 3


@pytest.fixture(scope="module")
def sinteticos():
    return dados_sinteticos(N_PARTICIPANTES, semente=0)


@pytest.fixture(scope="module")
def simulador(sinteticos):
    return treinar_final(sinteticos["refeicoes"], sinteticos["perfis"], ConfigRede(epocas=5))


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


def test_otimo_exato_empata_com_a_busca_exaustiva_nos_tres_dias(sinteticos):
    perfil = sinteticos["perfis"].iloc[0]
    cardapio = (
        sinteticos["cardapio"]
        .groupby("tipo", sort=False)
        .head(PRATOS_POR_TIPO_NA_BUSCA_EXAUSTIVA)
        .reset_index(drop=True)
        .assign(prato=lambda c: np.arange(len(c)))
    )
    lambdas = CONFIG_PEQUENA_COM_OUTROS_PESOS.lambdas
    dias = list(itertools.product(range(PRATOS_POR_TIPO_NA_BUSCA_EXAUSTIVA), repeat=3))
    todos = plano.habito(perfil, cardapio)[np.zeros(len(dias) ** len(DIAS_PLANO), dtype=int)]
    todos.prato_no_tipo = np.array(
        [np.concatenate(tres) for tres in itertools.product(dias, repeat=len(DIAS_PLANO))]
    )
    todos.caminhada[:, ag.POSICOES_D9] = experimentos.CAMINHADA_MAXIMA

    otimo = experimentos.otimo_exato(
        perfil, cardapio, ag.simulador_falso, CONFIG_PEQUENA_COM_OUTROS_PESOS
    )

    exaustiva = ag.aptidao(todos, ag.simulador_falso, perfil, cardapio, lambdas).min()
    assert ag.aptidao(otimo, ag.simulador_falso, perfil, cardapio, lambdas)[0] == pytest.approx(
        exaustiva
    )


def linha_feita_a_mao(participante, estrategia, semente, iauc_d9, aptidao):
    return {
        "participante": participante,
        "grupo": "dm2",
        "estrategia": estrategia,
        "semente": semente,
        "plano": 0,
        "iauc_d9": iauc_d9,
        "iauc_d9_sem_caminhada": iauc_d9,
        "aptidao": aptidao,
        "dias_fora_kcal": 0,
        "dias_fora_proteina": 0,
        "distancia_habito": 1.0,
        "avaliacoes": 100,
    }


def test_tabela_e_vitorias_fazem_a_media_das_sementes_antes_das_pessoas():
    resultados = pd.DataFrame(
        [
            linha_feita_a_mao(1, "ag", 0, 100.0, 1.0),
            linha_feita_a_mao(1, "ag", 1, 300.0, 5.0),
            linha_feita_a_mao(2, "ag", 0, 400.0, 1.0),
            linha_feita_a_mao(1, "aleatoria", 0, 200.0, 2.0),
            linha_feita_a_mao(1, "aleatoria", 1, 200.0, 3.0),
            linha_feita_a_mao(2, "aleatoria", 0, 500.0, 2.0),
        ]
    )

    tabela = figuras.tabela_por_grupo(resultados)
    vitorias = figuras.vitorias_ag(resultados)

    assert tabela.loc[("dm2", "ag"), "iauc_d9"] == 300.0
    assert tabela.loc[("dm2", "ag"), "participantes"] == 2
    assert vitorias.loc[("aptidao", "por_execucao"), "vence"] == pytest.approx(2 / 3)
    assert vitorias.loc[("aptidao", "por_participante"), "vence"] == 0.5


def test_empate_no_piso_nao_conta_como_derrota_do_ag():
    resultados = pd.DataFrame(
        [
            linha_feita_a_mao(1, "ag", 0, 0.0, 1.0),
            linha_feita_a_mao(1, "aleatoria", 0, 0.0, 2.0),
            linha_feita_a_mao(1, "ag", 1, 100.0, 1.0),
            linha_feita_a_mao(1, "aleatoria", 1, 200.0, 2.0),
            linha_feita_a_mao(2, "ag", 0, 300.0, 1.0),
            linha_feita_a_mao(2, "aleatoria", 0, 200.0, 2.0),
        ]
    )

    placar = figuras.vitorias_ag(resultados).loc[("iauc_d9", "por_execucao")]

    assert placar.to_dict() == pytest.approx({"vence": 1 / 3, "empata": 1 / 3, "perde": 1 / 3})


def test_caminhada_depois_da_troca_desconta_a_do_habito_e_mostra_quem_fica_negativo():
    resultados = pd.DataFrame(
        [
            linha_feita_a_mao(1, "habito", None, 1000.0, 1.0) | {"iauc_d9_sem_caminhada": 1100.0},
            linha_feita_a_mao(1, "ag", 0, 600.0, 1.0) | {"iauc_d9_sem_caminhada": 900.0},
            linha_feita_a_mao(2, "habito", None, 1000.0, 1.0) | {"iauc_d9_sem_caminhada": 1250.0},
            linha_feita_a_mao(2, "ag", 0, 600.0, 1.0) | {"iauc_d9_sem_caminhada": 700.0},
        ]
    )

    parte = figuras.caminhada_depois_da_troca(resultados).loc["dm2"]

    assert parte["parte"] == pytest.approx(((300.0 - 100.0) + (100.0 - 250.0)) / (400.0 + 400.0))
    assert parte["pessoas"] == 2
    assert parte["pessoas_com_parte_negativa"] == 1


def test_vitrine_roda_as_mesmas_sementes_que_os_demais():
    grupos = pd.Series({1: "dm2", 2: "dm2", 3: "saudavel"}, name="grupo").rename_axis(
        "participante"
    )

    lista = experimentos.tarefas(grupos, SEMENTES_PEQUENAS)

    assert lista == [(p, s) for p in (1, 2, 3) for s in (None, 0, 1)]


def test_d9_real_entra_com_a_glicose_de_jejum_e_so_quando_esta_completo(sinteticos):
    perfil = sinteticos["perfis"].iloc[0]
    refeicoes, dias = sinteticos["refeicoes"], sinteticos["dias"]
    da_pessoa_em_d9 = (refeicoes["participante"] == perfil["participante"]) & (
        refeicoes["dia"] == experimentos.DIA_ALVO
    )

    def devolve_glicose_pre(entrada):
        return entrada["glicose_pre"].to_numpy()

    real = experimentos.comportamento_real(perfil, refeicoes, dias, devolve_glicose_pre)
    sem_jantar = refeicoes[~(da_pessoa_em_d9 & (refeicoes["tipo"] == "jantar"))]
    sem_relogio = refeicoes.assign(
        minutos_ativos_60=refeicoes["minutos_ativos_60"].mask(da_pessoa_em_d9)
    )

    assert len(real) == len(TIPOS_REFEICAO)
    assert (real["iauc"] == perfil["glicose_jejum_cgm"]).all()
    assert real["iauc_d9"].iloc[0] == pytest.approx(
        len(TIPOS_REFEICAO) * perfil["glicose_jejum_cgm"]
    )
    assert experimentos.comportamento_real(perfil, sem_jantar, dias, devolve_glicose_pre) is None
    assert experimentos.comportamento_real(perfil, sem_relogio, dias, devolve_glicose_pre) is None
    quarta_refeicao_suja = dias.assign(
        n_refeicoes=dias["n_refeicoes"].mask(
            (dias["participante"] == perfil["participante"])
            & (dias["dia"] == experimentos.DIA_ALVO),
            REFEICOES_COM_UMA_SUJA,
        )
    )
    assert (
        experimentos.comportamento_real(
            perfil, refeicoes, quarta_refeicao_suja, devolve_glicose_pre
        )
        is None
    )


def test_d9_real_chega_ao_simulador_sem_a_fibra_maior_que_o_carbo(sinteticos):
    perfil = sinteticos["perfis"].iloc[0]
    refeicoes = sinteticos["refeicoes"]
    da_pessoa_em_d9 = (refeicoes["participante"] == perfil["participante"]) & (
        refeicoes["dia"] == experimentos.DIA_ALVO
    )
    fibra_impossivel = refeicoes.assign(
        fibra=refeicoes["fibra"].mask(da_pessoa_em_d9, refeicoes["carbo"] + 1)
    )
    fibras_vistas = []

    def guarda_a_fibra(entrada):
        fibras_vistas.append(entrada["fibra"])
        return np.zeros(len(entrada))

    experimentos.comportamento_real(perfil, fibra_impossivel, sinteticos["dias"], guarda_a_fibra)

    assert fibras_vistas
    assert all(fibra.isna().all() for fibra in fibras_vistas)


def test_distancia_do_otimo_compara_cada_execucao_com_o_otimo_da_pessoa():
    resultados = pd.DataFrame(
        [
            linha_feita_a_mao(1, "otimo", None, 50.0, 100.0),
            linha_feita_a_mao(1, "ag", 0, 60.0, 100.0),
            linha_feita_a_mao(1, "ag", 1, 90.0, 150.0),
            linha_feita_a_mao(2, "otimo", None, 50.0, 200.0),
            linha_feita_a_mao(2, "ag", 0, 70.0, 300.0),
        ]
    )

    distancia = figuras.distancia_do_otimo(resultados).loc["ag"]

    assert distancia["excesso_mediano"] == pytest.approx(0.5)
    assert distancia["excesso_maximo"] == pytest.approx(0.5)
    assert distancia["execucoes_no_otimo"] == pytest.approx(1 / 3)


@pytest.fixture(scope="module")
def resultados_pequenos(sinteticos) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabelas = {nome: sinteticos[nome] for nome in ("cardapio", "refeicoes", "dias")}
    partes, curvas = [], []
    for _, perfil in sinteticos["perfis"].iterrows():
        partes += experimentos.rodar_deterministicas(
            perfil, tabelas, ag.simulador_falso, CONFIG_PEQUENA_COM_OUTROS_PESOS
        )
        for semente in range(SEMENTES_PEQUENAS):
            resultados, historicos = experimentos.rodar_semente(
                perfil, tabelas, ag.simulador_falso, semente, CONFIG_PEQUENA_COM_OUTROS_PESOS
            )
            partes += resultados
            curvas += historicos
    return (
        experimentos.conformar_com_ausentes(pd.concat(partes), experimentos.RESULTADOS),
        experimentos.conformar_com_ausentes(pd.concat(curvas), experimentos.CURVAS),
    )


def test_nenhum_plano_fica_abaixo_do_otimo_medido_com_os_mesmos_pesos(resultados_pequenos):
    resultados, _ = resultados_pequenos
    todos = figuras.planos(resultados).dropna(subset=["aptidao"])
    otimo = todos[todos["estrategia"] == "otimo"].set_index("participante")["aptidao"]

    do_otimo = todos["participante"].map(otimo)
    piso = do_otimo - TOLERANCIA_FLOAT32 * do_otimo.abs()

    assert len(otimo) == N_PARTICIPANTES
    assert (todos["aptidao"] >= piso).all()


def test_figuras_saem_de_uma_tabela_pequena_sem_rodar_o_ag(
    resultados_pequenos, tmp_path, monkeypatch
):
    resultados, curvas = resultados_pequenos

    def ag_proibido(*args, **kwargs):
        raise AssertionError("figuras.py não pode rodar o AG")

    monkeypatch.setattr(ag, "rodar", ag_proibido)
    caminhos = figuras.gerar(resultados, curvas, tmp_path)

    esperadas = {
        "curvas_reais",
        "tabela_grupos",
        "ag_vs_aleatoria_real",
        "plausibilidade_macros",
        "caminhada_refeicao",
        "d7_d8_vs_habito",
        "frente_real",
        "planos_tres_grupos",
    }
    assert set(caminhos) == esperadas
    for caminho in caminhos.values():
        assert caminho.stat().st_size > 0
