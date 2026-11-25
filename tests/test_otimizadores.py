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
        assert resultado.viavel
        assert resultado.objetivos == pytest.approx(esperada), semente


def test_frente_sem_plano_viavel_diz_que_nao_e_viavel():
    cardapio = montar_cardapio(PRATOS_PEQUENOS)
    # O maior dia possível tem 515 + 745 + 750 = 2010 kcal, longe da faixa de 8000 a 12000.
    perfil = perfil_pequeno()
    perfil["kcal_dia"] = 10_000.0
    maior_dia = plano.habito(perfil, cardapio)
    maior_dia.prato_no_tipo[:] = 2

    resultado = nsga2.rodar(
        ag.ConfigAG(max_geracoes=50), ag.simulador_falso, perfil, cardapio, semente=0
    )

    assert not resultado.viavel
    menor_violacao = nsga2.calcular_violacao(maior_dia, perfil, cardapio)[0]
    violacao_frente = nsga2.calcular_violacao(resultado.frente, perfil, cardapio)
    assert violacao_frente == pytest.approx(np.full(len(resultado.frente), menor_violacao))


# Pontas em (0, 10) e (10, 0). Na escala de 0 a 1, (2, 2) vira (0,2, 0,2), o mais longe da reta
# f1 + f2 = 1 entre as pontas.
FRENTE_COM_JOELHO = np.array([[0, 10], [1, 5], [2, 2], [6, 1], [10, 0]], dtype=float)


def test_ponto_de_joelho_de_uma_frente_calculada_a_mao():
    assert nsga2.ponto_de_joelho(FRENTE_COM_JOELHO) == 2


def test_ponto_de_joelho_nao_depende_da_unidade_dos_objetivos():
    em_outra_unidade = FRENTE_COM_JOELHO * np.array([1000.0, 0.01])

    assert nsga2.ponto_de_joelho(em_outra_unidade) == 2


def test_ponto_de_joelho_de_frente_com_um_ponto_so():
    assert nsga2.ponto_de_joelho(np.array([[3.0, 4.0]])) == 0


# O jantar 9 fica longe de todos os outros, então não é um dos 3 vizinhos de nenhum prato. A
# mutação só chega nele quando sorteia qualquer prato do tipo, e o cruzamento só troca pratos que
# já estão na população. O carbo de 5 g o identifica na entrada do simulador.
PRATOS_COM_ISOLADO = {
    "cafe": [prato(30, 15, 10, 3), prato(35, 15, 10, 3)],
    "almoco": [prato(50, 25, 15, 5), prato(55, 25, 15, 5)],
    "jantar": [prato(50 + k, 30, 15, 4) for k in range(9)] + [prato(5, 60, 2, 1)],
}


def test_nsga2_chega_a_um_prato_que_nao_e_vizinho_de_ninguem():
    cardapio = montar_cardapio(PRATOS_COM_ISOLADO)
    habito = {tipo: {m: PRATOS_COM_ISOLADO[tipo][0][m] for m in MACROS} for tipo in TIPOS_REFEICAO}
    perfil = montar_perfil(kcal_dia=1100.0, proteina_dia=60.0, habito_por_tipo=habito)
    assert 9 not in ag.vizinhos_prato(cardapio)["jantar"]

    config = ag.ConfigAG(tamanho_populacao=4, max_geracoes=100, taxa_mutacao_prato=0.5)
    sementes_sem_o_prato_no_inicio = 0
    for semente in range(10):
        tem_o_prato = []

        def simulador_espiao(refeicoes, tem_o_prato=tem_o_prato):
            tem_o_prato.append((refeicoes["carbo"] == 5).any())
            return ag.simulador_falso(refeicoes)

        nsga2.rodar(config, simulador_espiao, perfil, cardapio, semente)

        if not tem_o_prato[0]:
            sementes_sem_o_prato_no_inicio += 1
            assert any(tem_o_prato[1:]), semente
    assert sementes_sem_o_prato_no_inicio > 0


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


def cardapio_por_kcal(pratos_por_tipo: dict[str, list[tuple[float, float, float]]]):
    """Cada prato é (kcal, carbo, proteina)."""
    return montar_cardapio(
        {
            tipo: [
                {"carbo": c, "proteina": p, "gordura": k / 100, "fibra": c / 10, "kcal": k}
                for k, c, p in pratos
            ]
            for tipo, pratos in pratos_por_tipo.items()
        }
    )


def perfil_por_kcal(kcal_dia: float, proteina_dia: float = 30.0) -> pd.Series:
    habito = {tipo: {m: 1.0 for m in MACROS} for tipo in TIPOS_REFEICAO}
    return montar_perfil(kcal_dia=kcal_dia, proteina_dia=proteina_dia, habito_por_tipo=habito)


def pratos_por_dia(pop: plano.Populacao) -> list[list[int]]:
    return pop.prato_no_tipo.reshape(-1, len(TIPOS_REFEICAO)).tolist()


def test_menos_carbo_fica_na_faixa_quando_uma_combinacao_cabe():
    # Faixa de 800 a 1200 kcal. O único dia que cabe é 800 + 100 + 100. Começar pelo café de
    # 500 kcal, que tem menos carbo, deixa o dia em 700 ou 1600.
    cardapio = cardapio_por_kcal(
        {
            "cafe": [(500, 10, 20), (800, 20, 25)],
            "almoco": [(100, 5, 30)],
            "jantar": [(100, 5, 20), (1000, 50, 25)],
        }
    )

    escolhido = otimizadores.menos_carbo(perfil_por_kcal(kcal_dia=1000.0), cardapio)

    assert pratos_por_dia(escolhido) == [[1, 0, 0]] * 3
    assert ag.total_por_dia(escolhido, cardapio, "kcal")[0].tolist() == [1000.0] * 3


def test_menos_carbo_minimiza_o_carbo_do_dia_e_nao_o_de_cada_refeicao():
    # Faixa de 800 a 1200 kcal. Os dias que cabem e o carbo de cada um:
    # café 0, almoço 0, jantar 1: 800 kcal, 90 g    café 1, almoço 0, jantar 0: 800 kcal, 25 g
    # café 0, almoço 1, jantar 0: 800 kcal, 75 g    café 1, almoço 0, jantar 1: 1100 kcal, 95 g
    # café 0, almoço 1, jantar 1: 1100 kcal, 145 g  café 1, almoço 1, jantar 0: 1100 kcal, 80 g
    # O café de 5 g obriga um almoço ou jantar caro. O dia de menos carbo começa pelo de 10 g.
    cardapio = cardapio_por_kcal(
        {
            "cafe": [(100, 5, 20), (400, 10, 25)],
            "almoco": [(100, 5, 30), (400, 60, 25)],
            "jantar": [(300, 10, 20), (600, 80, 25)],
        }
    )

    escolhido = otimizadores.menos_carbo(perfil_por_kcal(kcal_dia=1000.0), cardapio)

    assert pratos_por_dia(escolhido) == [[1, 0, 0]] * 3
    assert ag.total_por_dia(escolhido, cardapio, "carbo")[0].tolist() == [25.0] * 3


def test_menos_carbo_respeita_a_proteina_minima():
    # Os dois cafés têm as mesmas kcal. O de menos carbo deixa o dia com 25 g de proteína,
    # abaixo dos 80% de 60 g.
    cardapio = cardapio_por_kcal(
        {
            "cafe": [(300, 10, 5), (300, 20, 40)],
            "almoco": [(400, 30, 10)],
            "jantar": [(300, 20, 10)],
        }
    )

    escolhido = otimizadores.menos_carbo(perfil_por_kcal(1000.0, proteina_dia=60.0), cardapio)

    assert pratos_por_dia(escolhido) == [[1, 0, 0]] * 3


def test_menos_carbo_sem_dia_que_caiba_fica_o_de_menor_penalidade_e_menos_carbo():
    # Faixa de 2400 a 3600 kcal e o maior dia tem 1500. Os cafés 0 e 1 deixam o dia igualmente
    # longe da faixa, e o 1 tem menos carbo.
    cardapio = cardapio_por_kcal(
        {
            "cafe": [(400, 30, 20), (400, 20, 25), (100, 5, 30)],
            "almoco": [(600, 60, 20), (200, 20, 25)],
            "jantar": [(500, 50, 20)],
        }
    )

    escolhido = otimizadores.menos_carbo(perfil_por_kcal(kcal_dia=3000.0), cardapio)

    assert pratos_por_dia(escolhido) == [[1, 0, 0]] * 3


def test_menos_carbo_mantem_o_horario_do_habito_e_nao_caminha():
    dados = dados_sinteticos(3, semente=5)
    for _, perfil in dados["perfis"].iterrows():
        escolhido = otimizadores.menos_carbo(perfil, dados["cardapio"])
        habito = otimizadores.habito(perfil, dados["cardapio"])

        assert escolhido.intervalo.tolist() == habito.intervalo.tolist()
        assert (escolhido.caminhada == 0).all()
