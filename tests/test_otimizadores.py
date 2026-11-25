import itertools

import numpy as np
import pandas as pd
import pytest

from glicose import ag, nsga2, otimizadores, plano
from glicose.esquema import CARDAPIO, MACROS, TIPOS_REFEICAO, conformar, dados_sinteticos


def montar_cardapio(pratos_por_tipo: dict[str, list[dict]]) -> pd.DataFrame:
    linhas = [
        {"tipo": tipo, "fibra": 0.0, "foto": "", "participante_origem": 1, **prato}
        for tipo in TIPOS_REFEICAO
        for prato in pratos_por_tipo[tipo]
    ]
    cardapio = pd.DataFrame(linhas)
    cardapio["prato"] = range(len(cardapio))
    return conformar(cardapio, CARDAPIO)


def montar_perfil(kcal_dia: float, proteina_dia: float, habito_por_tipo: dict) -> pd.Series:
    colunas = {}
    for tipo, horario in zip(TIPOS_REFEICAO, (8.0, 12.5, 19.0), strict=True):
        colunas |= {f"{m}_{tipo}": valor for m, valor in habito_por_tipo[tipo].items()}
        colunas[f"horario_{tipo}"] = horario
    return pd.Series(
        {
            "participante": 1,
            "glicose_jejum_cgm": 100.0,
            "kcal_dia": kcal_dia,
            "proteina_dia": proteina_dia,
            **colunas,
        }
    )


def prato(carbo, proteina, gordura, fibra=0.0) -> dict:
    return {
        "carbo": carbo,
        "proteina": proteina,
        "gordura": gordura,
        "fibra": fibra,
        "kcal": 4 * carbo + 4 * proteina + 9 * gordura,
    }


PRATOS_PEQUENOS = {
    "cafe": [prato(20, 10, 5, 2), prato(45, 20, 10, 3), prato(70, 25, 15, 4)],
    "almoco": [prato(25, 15, 10, 4), prato(60, 30, 20, 6), prato(90, 40, 25, 8)],
    "jantar": [prato(20, 20, 10, 3), prato(55, 30, 20, 5), prato(85, 35, 30, 6)],
}


def perfil_pequeno() -> pd.Series:
    habito = {tipo: {m: PRATOS_PEQUENOS[tipo][2][m] for m in MACROS} for tipo in TIPOS_REFEICAO}
    return montar_perfil(kcal_dia=1400.0, proteina_dia=60.0, habito_por_tipo=habito)


# Pontos 0 a 5 em (f1, f2), os dois a minimizar.
# Frente 1: (1, 5), (2, 3), (4, 1). Nenhum é melhor nos dois.
# Frente 2: (3, 4) é dominado só por (2, 3), e (5, 2) só por (4, 1).
# Frente 3: (5, 5) é dominado por todos.
SEIS_PONTOS = np.array([[3, 4], [1, 5], [5, 5], [4, 1], [5, 2], [2, 3]], dtype=float)
FRENTES_SEIS_PONTOS = [{1, 3, 5}, {0, 4}, {2}]


def test_ordenar_nao_dominados_reproduz_as_frentes_calculadas_a_mao():
    frentes = nsga2.ordenar_nao_dominados(SEIS_PONTOS)

    assert [set(frente.tolist()) for frente in frentes] == FRENTES_SEIS_PONTOS


def test_distancia_aglomeracao_da_primeira_frente_calculada_a_mao():
    frente = SEIS_PONTOS[[1, 5, 3]]

    distancia = nsga2.distancia_aglomeracao(frente)

    # O ponto do meio, (2, 3), tem vizinhos (1, 5) e (4, 1). Em f1 a amplitude é 3 e o vão é 3.
    # Em f2 a amplitude é 4 e o vão é 4. A soma dá 1 + 1 = 2. As pontas ficam infinitas.
    assert distancia.tolist() == [np.inf, 2.0, np.inf]


def test_plano_viavel_vem_antes_de_qualquer_inviavel():
    objetivos = np.array([[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    violacao = np.array([0.0, 0.5, 0.2])

    frentes = nsga2.ordenar_nao_dominados(objetivos, violacao)

    assert [frente.tolist() for frente in frentes] == [[0], [2], [1]]


# O simulador falso ignora o horário e premia a caminhada, então a força bruta enumera os 3^9
# planos de pratos com a caminhada no máximo. penalidade_kcal e penalidade_proteina valem
# também em D7 e D8, por isso só D9 livre não basta.
def frente_forca_bruta(perfil, cardapio) -> np.ndarray:
    combinacoes = np.array(list(itertools.product(range(3), repeat=plano.N_REFEICOES)))
    minimos, _ = plano.limites_intervalo()
    todos = plano.Populacao(
        combinacoes,
        np.tile(minimos, (len(combinacoes), 1)),
        np.full(combinacoes.shape, plano.LIMITES_CAMINHADA[1]),
    )
    objetivos = nsga2.calcular_objetivos(todos, ag.simulador_falso, perfil, cardapio)
    viaveis = objetivos[nsga2.calcular_violacao(todos, perfil, cardapio) == 0]
    ordenados = viaveis[np.lexsort((viaveis[:, 1], viaveis[:, 0]))]
    frente, menor_distancia = [], np.inf
    for iauc, distancia in ordenados:
        if distancia < menor_distancia:
            frente.append((iauc, distancia))
            menor_distancia = distancia
    return np.array(frente)


def test_frente_do_nsga2_igual_a_da_forca_bruta():
    cardapio, perfil = montar_cardapio(PRATOS_PEQUENOS), perfil_pequeno()
    esperada = frente_forca_bruta(perfil, cardapio)
    assert len(esperada) > 1

    for semente in range(3):
        resultado = nsga2.rodar(ag.ConfigAG(), ag.simulador_falso, perfil, cardapio, semente)
        assert resultado.objetivos == pytest.approx(esperada), semente


def test_busca_aleatoria_usa_exatamente_o_orcamento_do_ag():
    dados = dados_sinteticos(1, semente=4)
    perfil, cardapio = dados["perfis"].iloc[0], dados["cardapio"]
    resultado_ag = ag.rodar(ag.ConfigAG(max_geracoes=4), ag.simulador_falso, perfil, cardapio, 0)

    for n_avaliacoes in (resultado_ag.avaliacoes, 250):
        linhas_simuladas = []

        def simulador_espiao(refeicoes, linhas_simuladas=linhas_simuladas):
            linhas_simuladas.append(len(refeicoes))
            return ag.simulador_falso(refeicoes)

        resultado = otimizadores.busca_aleatoria(
            n_avaliacoes, simulador_espiao, perfil, cardapio, ag.Lambdas(), semente=0
        )

        assert sum(linhas_simuladas) == n_avaliacoes * plano.N_REFEICOES
        assert resultado.avaliacoes == n_avaliacoes
        refeita = ag.aptidao(resultado.melhor, ag.simulador_falso, perfil, cardapio, ag.Lambdas())
        assert refeita[0] == pytest.approx(resultado.aptidao)


CARDAPIO_GULOSO = {
    "cafe": [prato(10, 5, 2, 1) | {"kcal": 100.0}, prato(30, 20, 8, 3) | {"kcal": 400.0}],
    "almoco": [prato(20, 10, 4, 2) | {"kcal": 200.0}, prato(60, 30, 15, 5) | {"kcal": 600.0}],
    "jantar": [prato(15, 8, 3, 1) | {"kcal": 150.0}, prato(50, 25, 12, 4) | {"kcal": 500.0}],
}


def perfil_guloso(kcal_dia: float) -> pd.Series:
    habito = {tipo: {m: 1.0 for m in MACROS} for tipo in TIPOS_REFEICAO}
    return montar_perfil(kcal_dia=kcal_dia, proteina_dia=30.0, habito_por_tipo=habito)


def pratos_por_dia(pop: plano.Populacao) -> list[list[int]]:
    return pop.prato_no_tipo.reshape(-1, len(TIPOS_REFEICAO)).tolist()


def test_menos_carbo_escolhe_o_prato_de_menos_carbo_que_respeita_as_calorias():
    cardapio = montar_cardapio(CARDAPIO_GULOSO)

    escolhido = otimizadores.menos_carbo(perfil_guloso(kcal_dia=1000.0), cardapio)

    # Faixa de 800 a 1200 kcal. Pegar o prato de menos carbo em todas as refeições daria
    # 100 + 200 + 150 = 450 kcal. No jantar só o prato de 500 kcal fecha o dia dentro da faixa.
    assert pratos_por_dia(escolhido) == [[0, 0, 1]] * 3
    kcal_dias = ag.total_por_dia(escolhido, cardapio, "kcal")[0]
    assert kcal_dias.tolist() == [800.0] * 3


def test_menos_carbo_sem_prato_que_caiba_fica_o_mais_perto_da_faixa():
    escolhido = otimizadores.menos_carbo(
        perfil_guloso(kcal_dia=3000.0), montar_cardapio(CARDAPIO_GULOSO)
    )

    assert pratos_por_dia(escolhido) == [[1, 1, 1]] * 3


def test_menos_carbo_mantem_horario_e_caminhada_do_habito():
    dados = dados_sinteticos(3, semente=5)
    for _, perfil in dados["perfis"].iterrows():
        escolhido = otimizadores.menos_carbo(perfil, dados["cardapio"])
        habito = otimizadores.habito(perfil, dados["cardapio"])

        assert escolhido.intervalo.tolist() == habito.intervalo.tolist()
        assert escolhido.caminhada.tolist() == habito.caminhada.tolist()
