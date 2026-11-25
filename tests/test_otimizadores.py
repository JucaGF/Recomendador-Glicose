import pandas as pd
import pytest

from glicose import ag, otimizadores, plano
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
        refeita = ag.fitness(resultado.melhor, ag.simulador_falso, perfil, cardapio, ag.Lambdas())
        assert refeita[0] == pytest.approx(resultado.fitness)


CARDAPIO_GULOSO = {
    "cafe": [prato(10, 5, 2, 1) | {"kcal": 100.0}, prato(30, 20, 8, 3) | {"kcal": 400.0}],
    "almoco": [prato(20, 10, 4, 2) | {"kcal": 200.0}, prato(60, 30, 15, 5) | {"kcal": 600.0}],
    "jantar": [prato(15, 8, 3, 1) | {"kcal": 150.0}, prato(50, 25, 12, 4) | {"kcal": 500.0}],
}


def perfil_guloso(kcal_dia: float) -> pd.Series:
    habito = {tipo: {m: 1.0 for m in MACROS} for tipo in TIPOS_REFEICAO}
    return montar_perfil(kcal_dia=kcal_dia, proteina_dia=30.0, habito_por_tipo=habito)


def pratos_por_dia(pop: plano.Populacao) -> list[list[int]]:
    return pop.prato.reshape(-1, len(TIPOS_REFEICAO)).tolist()


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
