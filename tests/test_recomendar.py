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
PLANOS_POR_ESTRATEGIA = 4
# A rede calcula em float32, e lotes de tamanhos diferentes podem mudar o último dígito.
TOLERANCIA_FLOAT32 = 1e-5
REFEICOES_COM_UMA_SUJA = 4


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
    assert vitorias["aptidao_por_execucao"] == pytest.approx(2 / 3)
    assert vitorias["aptidao_por_participante"] == 0.5


def test_parte_da_caminhada_divide_o_ganho_sem_caminhada_pelo_ganho_total():
    ag_caminhando = linha_feita_a_mao(1, "ag", 0, 600.0, 1.0) | {"iauc_d9_sem_caminhada": 900.0}
    resultados = pd.DataFrame([linha_feita_a_mao(1, "habito", None, 1000.0, 1.0), ag_caminhando])

    parte = figuras.parte_da_caminhada(figuras.tabela_por_grupo(resultados))

    assert parte["dm2"] == pytest.approx(0.75)


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


def refeicao_de_d9(semente, tipo, carbo, caminhada):
    return {
        "participante": 1,
        "grupo": "dm2",
        "estrategia": "ag",
        "semente": semente,
        "dia": 9,
        "tipo": tipo,
        "carbo": carbo,
        "caminhada": caminhada,
    }


def test_maior_caminhada_so_conta_quando_e_unica():
    resultados = pd.DataFrame(
        [
            refeicao_de_d9(0, "cafe", 10.0, 30.0),
            refeicao_de_d9(0, "almoco", 90.0, 30.0),
            refeicao_de_d9(0, "jantar", 20.0, 30.0),
            refeicao_de_d9(1, "cafe", 10.0, 5.0),
            refeicao_de_d9(1, "almoco", 90.0, 25.0),
            refeicao_de_d9(1, "jantar", 20.0, 0.0),
            refeicao_de_d9(2, "cafe", 10.0, 25.0),
            refeicao_de_d9(2, "almoco", 90.0, 5.0),
            refeicao_de_d9(2, "jantar", 20.0, 0.0),
        ]
    )

    coincidencias = figuras.caminhada_e_carbo(resultados)

    assert coincidencias["caminhada_unica"].tolist() == [False, True, True]
    assert figuras.fracao_no_mais_carbo(coincidencias)["dm2"] == 0.5


def resultados_sem_ag(sinteticos) -> tuple[pd.DataFrame, pd.DataFrame]:
    cardapio = sinteticos["cardapio"]
    rng = np.random.default_rng(0)
    partes, curvas = [], []
    for _, perfil in sinteticos["perfis"].iterrows():
        partes.append(
            experimentos.comportamento_real(
                perfil, sinteticos["refeicoes"], sinteticos["dias"], ag.simulador_falso
            )
        )
        for nome, montar in experimentos.DETERMINISTICAS.items():
            partes.append(
                experimentos.linhas_dos_planos(
                    montar(perfil, cardapio), nome, None, perfil, ag.simulador_falso, cardapio
                )
            )
        for semente in range(2):
            for estrategia in ("ag", "aleatoria", "nsga2_joelho", experimentos.FRENTE):
                aleatorios = plano.aleatoria(PLANOS_POR_ESTRATEGIA, cardapio, rng)
                if estrategia != experimentos.FRENTE:
                    aleatorios = aleatorios[0]
                partes.append(
                    experimentos.linhas_dos_planos(
                        aleatorios, estrategia, semente, perfil, ag.simulador_falso, cardapio
                    )
                )
            for estrategia in ("ag", "aleatoria"):
                curvas.append(
                    pd.DataFrame(
                        {
                            "participante": perfil["participante"],
                            "grupo": perfil["grupo"],
                            "estrategia": estrategia,
                            "semente": semente,
                            "geracao": [0, 1, 2],
                            "avaliacoes": [100, 200, 300],
                            "melhor": [30.0, 20.0, 10.0],
                            "media": [60.0, 40.0, 20.0],
                        }
                    )
                )
    resultados = experimentos.conformar_com_ausentes(
        pd.concat([p for p in partes if p is not None]), experimentos.RESULTADOS
    )
    return resultados, pd.concat(curvas, ignore_index=True)


def test_figuras_saem_de_uma_tabela_pequena_sem_rodar_o_ag(sinteticos, tmp_path, monkeypatch):
    resultados, curvas = resultados_sem_ag(sinteticos)

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
