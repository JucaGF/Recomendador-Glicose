import itertools

import numpy as np
import pandas as pd
import pytest

from glicose import ag, plano
from glicose.esquema import (
    CARDAPIO,
    ENTRADA_SIMULADOR,
    HORAS_DESDE_REFEICAO_ANTERIOR,
    MACROS,
    TIPOS_REFEICAO,
    conformar,
    dados_sinteticos,
)

MUTACAO_GAUSSIANA_MAXIMA = {"taxa_mutacao_gaussiana": 1.0, "sigma_relativo_inicial": 1.0}

PRATOS_PEQUENOS = {
    "cafe": [(20, 10, 5, 2), (45, 20, 10, 3), (70, 25, 15, 4)],
    "almoco": [(25, 15, 10, 4), (60, 30, 20, 6), (90, 40, 25, 8)],
    "jantar": [(20, 20, 10, 3), (55, 30, 20, 5), (85, 35, 30, 6)],
}


def cardapio_pequeno(pratos: dict = PRATOS_PEQUENOS) -> pd.DataFrame:
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
        for carbo, proteina, gordura, fibra in pratos[tipo]
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


def perfil_sem_jantar() -> pd.Series:
    perfil = perfil_pequeno()
    perfil[[f"{coluna}_jantar" for coluna in (*MACROS, "horario")]] = np.nan
    return perfil


def planos(*pratos_por_plano: list[int]) -> plano.Populacao:
    minimos, _ = plano.limites_intervalo()
    forma = (len(pratos_por_plano), plano.N_REFEICOES)
    return plano.Populacao(
        np.array(pratos_por_plano), np.broadcast_to(minimos, forma).copy(), np.zeros(forma)
    )


def avaliados_pelo_ag(config: ag.ConfigAG, perfil, cardapio) -> tuple[pd.DataFrame, ag.Resultado]:
    avaliados = []

    def simulador_espiao(refeicoes):
        avaliados.append(refeicoes)
        return ag.simulador_falso(refeicoes)

    resultado = ag.rodar(config, simulador_espiao, perfil, cardapio, semente=7)
    assert len(avaliados) == len(resultado.historico)
    return pd.concat(avaliados), resultado


def horarios_por_plano(refeicoes: pd.DataFrame) -> np.ndarray:
    return refeicoes["horario"].to_numpy().reshape(-1, plano.N_REFEICOES)


def dias_viaveis(perfil, cardapio) -> plano.Populacao:
    todos = planos(*[list(dia) * 3 for dia in itertools.product(range(3), repeat=3)])
    viavel = (ag.penalidade_kcal(todos, perfil, cardapio) == 0) & (
        ag.penalidade_proteina(todos, perfil, cardapio) == 0
    )
    return todos[np.nonzero(viavel)[0]]


def distancia_em_desvios(tipo: str, de: int, para: int) -> float:
    todos_os_pratos = np.array([prato for pratos in PRATOS_PEQUENOS.values() for prato in pratos])
    desvio = todos_os_pratos.std(axis=0, ddof=1)
    diferenca = np.subtract(PRATOS_PEQUENOS[tipo][de], PRATOS_PEQUENOS[tipo][para])
    return float(np.linalg.norm(diferenca / desvio))


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
    config = ag.ConfigAG(horario_livre=True, **MUTACAO_GAUSSIANA_MAXIMA)

    for _, perfil in dados["perfis"].iterrows():
        refeicoes, resultado = avaliados_pelo_ag(config, perfil, dados["cardapio"])

        relogio = refeicoes["dia"].to_numpy() * 24 + refeicoes["horario"].to_numpy()
        intervalos = np.diff(relogio.reshape(-1, plano.N_REFEICOES), axis=1)
        assert intervalos.min() >= 3.0 - 1e-9
        jantares = refeicoes.loc[refeicoes["tipo"] == "jantar", "horario"]
        assert jantares.max() <= plano.HORA_MAXIMA_JANTAR
        assert len(refeicoes) == resultado.avaliacoes * plano.N_REFEICOES


def test_por_padrao_todo_plano_avaliado_tem_os_horarios_do_habito():
    dados = dados_sinteticos(3, semente=1)
    config = ag.ConfigAG(max_geracoes=20, **MUTACAO_GAUSSIANA_MAXIMA)

    for _, perfil in dados["perfis"].iterrows():
        refeicoes, _ = avaliados_pelo_ag(config, perfil, dados["cardapio"])

        do_habito = plano.horarios(plano.habito(perfil, dados["cardapio"]))
        assert (horarios_por_plano(refeicoes) == do_habito).all()


def test_com_horario_livre_cada_refeicao_e_avaliada_em_varios_horarios():
    dados = dados_sinteticos(1, semente=1)
    config = ag.ConfigAG(max_geracoes=20, horario_livre=True)

    refeicoes, _ = avaliados_pelo_ag(config, dados["perfis"].iloc[0], dados["cardapio"])

    por_plano = horarios_por_plano(refeicoes)
    for j in range(plano.N_REFEICOES):
        assert len(np.unique(por_plano[:, j])) > 1


def test_cantos_dos_limites_decodificam_jantar_ate_23h_e_intervalos_de_3_horas_ou_mais():
    minimos, maximos = plano.limites_intervalo()
    cantos = np.array(list(itertools.product((0, 1), repeat=plano.N_REFEICOES)))
    intervalo = np.where(cantos == 1, maximos, minimos)
    pop = plano.Populacao(np.zeros_like(cantos), intervalo, np.zeros(cantos.shape))

    por_dia_e_tipo = plano.horarios(pop).reshape(len(pop), -1, len(TIPOS_REFEICAO))

    jantares = por_dia_e_tipo[:, :, TIPOS_REFEICAO.index("jantar")]
    assert jantares.max() == plano.HORA_MAXIMA_JANTAR
    assert np.diff(por_dia_e_tipo, axis=2).min() >= 3.0


def test_sem_penalidade_d9_tem_o_menor_carbo_e_com_p_kcal_fica_perto_do_habito():
    cardapio, perfil = cardapio_pequeno(), perfil_pequeno()
    menos_carbo = [
        int(cardapio.loc[cardapio["tipo"] == tipo, "carbo"].to_numpy().argmin())
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

        assert pratos_d9(sem_penalidade.melhor) == menos_carbo
        kcal_d9 = ag.total_por_dia(com_penalidade.melhor, cardapio, "kcal")[0, ag.DIA_9]
        assert 0.8 * perfil["kcal_dia"] <= kcal_d9 <= 1.2 * perfil["kcal_dia"]


def test_penalidade_de_proteina_soma_o_que_falta_para_80_por_cento_em_cada_dia():
    cardapio, perfil = cardapio_pequeno(), perfil_pequeno()
    com_45_g_por_dia = [0] * plano.N_REFEICOES
    com_100_g_por_dia = [2] * plano.N_REFEICOES
    com_45_g_so_em_d9 = [2, 2, 2, 2, 2, 2, 0, 0, 0]
    faltam_3_de_60_g = 0.05

    penalidades = ag.penalidade_proteina(
        planos(com_45_g_por_dia, com_100_g_por_dia, com_45_g_so_em_d9), perfil, cardapio
    )

    assert penalidades == pytest.approx([3 * faltam_3_de_60_g, 0.0, faltam_3_de_60_g])


def test_distancia_habito_e_a_media_das_9_refeicoes_em_desvios_padrao_do_cardapio():
    igual_ao_habito = [2] * plano.N_REFEICOES
    jantar_d9_trocado = [2, 2, 2, 2, 2, 2, 2, 2, 1]

    distancias = ag.distancia_habito(
        planos(igual_ao_habito, jantar_d9_trocado), perfil_pequeno(), cardapio_pequeno()
    )

    assert distancias == pytest.approx([0.0, distancia_em_desvios("jantar", 2, 1) / 9])


def test_simulador_que_inverte_a_saida_troca_a_iauc_entre_os_planos():
    cardapio, perfil = cardapio_pequeno(), perfil_pequeno()
    pouco_carbo, muito_carbo = [0] * plano.N_REFEICOES, [2] * plano.N_REFEICOES
    pop = planos(pouco_carbo, muito_carbo)
    iauc_pouco_carbo = 1.5 * (20 + 25 + 20)
    iauc_muito_carbo = 1.5 * (70 + 90 + 85)

    def simulador_invertido(refeicoes: pd.DataFrame) -> np.ndarray:
        return ag.simulador_falso(refeicoes)[::-1]

    na_ordem = ag.iauc_d9(pop, ag.simulador_falso, perfil, cardapio)
    invertida = ag.iauc_d9(pop, simulador_invertido, perfil, cardapio)

    assert na_ordem == pytest.approx([iauc_pouco_carbo, iauc_muito_carbo])
    assert invertida == pytest.approx([iauc_muito_carbo, iauc_pouco_carbo])


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


def test_distancia_habito_ignora_o_tipo_sem_habito():
    habito_de_cafe_e_almoco = [2] * plano.N_REFEICOES
    jantar_d9_trocado = [2, 2, 2, 2, 2, 2, 2, 2, 1]
    cafe_d9_trocado = [2, 2, 2, 2, 2, 2, 1, 2, 2]

    distancias = ag.distancia_habito(
        planos(habito_de_cafe_e_almoco, jantar_d9_trocado, cafe_d9_trocado),
        perfil_sem_jantar(),
        cardapio_pequeno(),
    )

    assert distancias == pytest.approx([0.0, 0.0, distancia_em_desvios("cafe", 2, 1) / 6])


def test_habito_sem_jantar_usa_o_prato_mediano_e_o_intervalo_mediano():
    prato_mediano_do_jantar = 1
    perfil = perfil_sem_jantar()
    perfil["kcal_dia"] = 515.0 + 745.0 + 520.0

    inicial = plano.habito(perfil, cardapio_pequeno())

    assert inicial.prato_no_tipo[0].tolist() == [2, 2, prato_mediano_do_jantar] * 3
    intervalo_mediano = HORAS_DESDE_REFEICAO_ANTERIOR["jantar"][0]
    assert plano.horarios(inicial)[0, :3] == pytest.approx([8.0, 12.5, 12.5 + intervalo_mediano])


def test_ag_termina_com_aptidao_finita_para_perfil_sem_jantar():
    resultado = ag.rodar(
        ag.ConfigAG(), ag.simulador_falso, perfil_sem_jantar(), cardapio_pequeno(), semente=0
    )

    assert np.isfinite(resultado.aptidao)
    assert np.isfinite(resultado.historico[["melhor", "media"]].to_numpy()).all()


def test_mutacao_de_prato_alcanca_o_prato_que_nao_e_vizinho_de_nenhum_outro():
    jantares = [(10, 30, 10, 3), (11, 30, 10, 3), (12, 30, 10, 3), (13, 30, 10, 3), (95, 30, 10, 3)]
    cardapio = cardapio_pequeno(PRATOS_PEQUENOS | {"jantar": jantares})
    vizinhos = ag.vizinhos_prato(cardapio)
    isolado = 4
    assert isolado not in vizinhos["jantar"]
    todos_no_primeiro_prato = planos(*[[0] * plano.N_REFEICOES] * 200)

    mutados = ag.mutacao_prato(todos_no_primeiro_prato, vizinhos, 1.0, np.random.default_rng(0))

    posicoes_jantar = [j for j, (_, tipo) in enumerate(plano.REFEICOES_PLANO) if tipo == "jantar"]
    assert isolado in mutados.prato_no_tipo[:, posicoes_jantar]


def test_habito_e_o_dia_viavel_mais_perto_quando_o_prato_mais_perto_estoura_as_kcal():
    cardapio, perfil = cardapio_pequeno(), perfil_pequeno()
    pratos_mais_perto_em_macros = planos([2] * plano.N_REFEICOES)
    assert ag.penalidade_kcal(pratos_mais_perto_em_macros, perfil, cardapio)[0] > 0

    escolhido = plano.habito(perfil, cardapio)

    assert ag.penalidade_kcal(escolhido, perfil, cardapio)[0] == 0
    assert ag.penalidade_proteina(escolhido, perfil, cardapio)[0] == 0
    menor_distancia_viavel = ag.distancia_habito(dias_viaveis(perfil, cardapio), perfil, cardapio)
    assert ag.distancia_habito(escolhido, perfil, cardapio)[0] == pytest.approx(
        menor_distancia_viavel.min()
    )


def test_sem_dia_viavel_o_habito_e_o_dia_que_menos_sai_da_faixa():
    perfil = perfil_pequeno()
    perfil["kcal_dia"] = 10_000.0
    dia_de_mais_kcal = [2] * plano.N_REFEICOES

    escolhido = plano.habito(perfil, cardapio_pequeno())

    assert escolhido.prato_no_tipo[0].tolist() == dia_de_mais_kcal
