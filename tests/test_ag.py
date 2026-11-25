import itertools

import numpy as np
import pandas as pd
import pytest

from glicose import ag, plano
from glicose.esquema import CARDAPIO, ENTRADA_SIMULADOR, TIPOS_REFEICAO, conformar, dados_sinteticos

PRATOS_PEQUENOS = {
    "cafe": [(20, 10, 5, 2), (45, 20, 10, 3), (70, 25, 15, 4)],
    "almoco": [(25, 15, 10, 4), (60, 30, 20, 6), (90, 40, 25, 8)],
    "jantar": [(20, 20, 10, 3), (55, 30, 20, 5), (85, 35, 30, 6)],
}


def cardapio_pequeno() -> pd.DataFrame:
    linhas = [
        {
            "tipo": tipo,
            "carbo": carbo,
            "proteina": proteina,
            "gordura": gordura,
            "fibra": fibra,
            "kcal": 4 * carbo + 4 * proteina + 9 * gordura,
            "foto": "",
            "participante_origem": 1,
        }
        for tipo in TIPOS_REFEICAO
        for carbo, proteina, gordura, fibra in PRATOS_PEQUENOS[tipo]
    ]
    cardapio = pd.DataFrame(linhas)
    cardapio["prato"] = range(len(cardapio))
    return conformar(cardapio, CARDAPIO)


def perfil_pequeno() -> pd.Series:
    habito_por_tipo = {}
    for tipo, horario in zip(TIPOS_REFEICAO, (8.0, 12.5, 19.0), strict=True):
        carbo, proteina, gordura, fibra = PRATOS_PEQUENOS[tipo][2]
        habito_por_tipo |= {
            f"carbo_{tipo}": carbo,
            f"proteina_{tipo}": proteina,
            f"gordura_{tipo}": gordura,
            f"fibra_{tipo}": fibra,
            f"horario_{tipo}": horario,
        }
    return pd.Series(
        {
            "participante": 1,
            "glicose_jejum_cgm": 100.0,
            "kcal_dia": 1400.0,
            "proteina_dia": 60.0,
            **habito_por_tipo,
        }
    )


# O simulador falso ignora o horário e premia a caminhada, então basta enumerar os pratos com o
# horário fixo e a caminhada no máximo.
def forca_bruta(perfil, cardapio, lambdas) -> tuple[np.ndarray, float]:
    combinacoes = np.array(list(itertools.product(range(3), repeat=plano.N_REFEICOES)))
    minimos, _ = plano.limites_intervalo()
    todos = plano.Populacao(
        combinacoes,
        np.tile(minimos, (len(combinacoes), 1)),
        np.full(combinacoes.shape, plano.LIMITES_CAMINHADA[1]),
    )
    aptidoes = ag.aptidao(todos, ag.simulador_falso, perfil, cardapio, lambdas)
    melhor = np.argmin(aptidoes)
    return combinacoes[melhor], aptidoes[melhor]


def pratos_d9(pop: plano.Populacao) -> list[int]:
    return pop.prato_no_tipo[0, ag.POSICOES_D9].tolist()


def test_ag_acha_o_otimo_da_forca_bruta_em_10_sementes():
    cardapio, perfil = cardapio_pequeno(), perfil_pequeno()
    config = ag.ConfigAG()
    otimo, aptidao_otima = forca_bruta(perfil, cardapio, config.lambdas)

    for semente in range(10):
        resultado = ag.rodar(config, ag.simulador_falso, perfil, cardapio, semente)
        assert pratos_d9(resultado.melhor) == otimo[ag.POSICOES_D9].tolist(), semente
        assert resultado.aptidao == pytest.approx(aptidao_otima, abs=1.0), semente


def test_nenhum_individuo_avaliado_tem_menos_de_3_horas_entre_refeicoes_nem_jantar_apos_23h():
    dados = dados_sinteticos(3, semente=1)
    config_mutacao_maxima = ag.ConfigAG(taxa_mutacao_gaussiana=1.0, sigma_relativo_inicial=1.0)

    for _, perfil in dados["perfis"].iterrows():
        avaliados = []

        def simulador_espiao(refeicoes, avaliados=avaliados):
            avaliados.append(refeicoes)
            return ag.simulador_falso(refeicoes)

        resultado = ag.rodar(
            config_mutacao_maxima, simulador_espiao, perfil, dados["cardapio"], semente=7
        )

        refeicoes = pd.concat(avaliados)
        relogio = refeicoes["dia"].to_numpy() * 24 + refeicoes["horario"].to_numpy()
        intervalos = np.diff(relogio.reshape(-1, plano.N_REFEICOES), axis=1)
        assert intervalos.min() >= 3.0 - 1e-9
        jantares = refeicoes.loc[refeicoes["tipo"] == "jantar", "horario"]
        assert jantares.max() <= plano.HORA_MAXIMA_JANTAR
        assert len(avaliados) == len(resultado.historico)
        assert len(refeicoes) == resultado.avaliacoes * plano.N_REFEICOES


def test_cantos_dos_limites_decodificam_jantar_ate_23h_e_intervalos_de_3_horas_ou_mais():
    minimos, maximos = plano.limites_intervalo()
    cantos = np.array(list(itertools.product((0, 1), repeat=plano.N_REFEICOES)))
    intervalo = np.where(cantos == 1, maximos, minimos)
    pop = plano.Populacao(np.zeros_like(cantos), intervalo, np.zeros(cantos.shape))

    por_dia_e_tipo = plano.horarios(pop).reshape(len(pop), -1, len(TIPOS_REFEICAO))

    jantares = por_dia_e_tipo[:, :, TIPOS_REFEICAO.index("jantar")]
    assert jantares.max() == plano.HORA_MAXIMA_JANTAR
    assert np.diff(por_dia_e_tipo, axis=2).min() >= 3.0


def test_penalidade_de_kcal_mantem_d9_perto_do_habito():
    cardapio, perfil = cardapio_pequeno(), perfil_pequeno()
    menos_kcal = [
        int(cardapio.loc[cardapio["tipo"] == tipo, "kcal"].to_numpy().argmin())
        for tipo in TIPOS_REFEICAO
    ]

    for semente in range(3):
        sem_penalidade = ag.rodar(
            ag.ConfigAG(lambdas=ag.Lambdas(0.0, 0.0, 0.0)),
            ag.simulador_falso,
            perfil,
            cardapio,
            semente,
        )
        com_penalidade = ag.rodar(
            ag.ConfigAG(lambdas=ag.Lambdas(kcal=ag.Lambdas().kcal, proteina=0.0, habito=0.0)),
            ag.simulador_falso,
            perfil,
            cardapio,
            semente,
        )

        assert pratos_d9(sem_penalidade.melhor) == menos_kcal
        kcal_d9 = ag.total_por_dia(com_penalidade.melhor, cardapio, "kcal")[0, ag.DIA_9]
        assert 0.8 * perfil["kcal_dia"] <= kcal_d9 <= 1.2 * perfil["kcal_dia"]


def test_para_refeicoes_segue_o_contrato_do_simulador():
    dados = dados_sinteticos(1, semente=2)
    perfil = dados["perfis"].iloc[0]
    pop = plano.aleatoria(4, dados["cardapio"], np.random.default_rng(0))

    refeicoes = plano.para_refeicoes(pop, dados["cardapio"], perfil)

    assert refeicoes.dtypes.astype(str).to_dict() == ENTRADA_SIMULADOR
    assert len(refeicoes) == 4 * plano.N_REFEICOES
    assert (refeicoes["glicose_pre"] == perfil["glicose_jejum_cgm"]).all()
    assert refeicoes["minutos_ativos_60"].tolist() == pop.caminhada.ravel().tolist()
    assert set(refeicoes["dia"]) == {7, 8, 9}


def test_habito_representa_os_horarios_do_participante():
    dados = dados_sinteticos(5, semente=3)
    for _, perfil in dados["perfis"].iterrows():
        horarios = plano.horarios(plano.habito(perfil, dados["cardapio"]))[0, :3]
        esperado = [perfil[f"horario_{tipo}"] for tipo in TIPOS_REFEICAO]
        assert horarios == pytest.approx(esperado)
