import numpy as np
import pandas as pd
import pytest

from glicose import leitura, refeicoes


def interpolar(leituras: pd.Series) -> pd.Series:
    grade = pd.date_range(leituras.index[0], leituras.index[-1], freq="min")
    return leituras.reindex(grade).interpolate(method="time", limit_area="inside")


def leituras_a_cada(inicio: str, passo: int, valores) -> pd.Series:
    instantes = pd.date_range(inicio, periods=len(valores), freq=f"{passo}min")
    return pd.Series(np.asarray(valores, dtype=float), index=instantes)


def leituras_aleatorias(datas: list[str], semente: int = 1) -> pd.Series:
    gerador = np.random.default_rng(semente)
    dias = [leituras_a_cada(f"{data} 00:00", 5, gerador.integers(90, 180, 288)) for data in datas]
    return pd.concat(dias)


def trocar_leituras(leituras: pd.Series, inicio: str, valores: list[float]) -> pd.Series:
    trocadas = leituras.copy()
    trocadas[pd.date_range(inicio, periods=len(valores), freq="5min")] = valores
    return trocadas


def tabela_com(leituras: pd.Series, refeicoes_registradas: dict[str, str]) -> pd.DataFrame:
    primeiro_dia = leituras.index[0].normalize()
    depois_do_ultimo_dia = leituras.index[-1].normalize() + pd.Timedelta(days=1)
    grade = pd.date_range(primeiro_dia, depois_do_ultimo_dia, freq="min", inclusive="left")
    tabela = pd.DataFrame(index=grade)
    tabela["dexcom"] = interpolar(leituras).reindex(grade)
    tabela["libre"] = tabela["dexcom"]
    tabela["mets"] = 1.0
    tabela["refeicao"] = None
    for instante, tipo in refeicoes_registradas.items():
        tabela.loc[pd.Timestamp(instante), "refeicao"] = tipo
    for macro in refeicoes.MACROS:
        tabela[macro] = 30.0
    tabela["foto"] = None
    return tabela


def falha_do_dexcom(tabela: pd.DataFrame, ultima_leitura: str, proxima_leitura: str) -> None:
    sem_sinal = slice(
        pd.Timestamp(ultima_leitura) + pd.Timedelta(minutes=1),
        pd.Timestamp(proxima_leitura) - pd.Timedelta(minutes=1),
    )
    tabela.loc[sem_sinal, "dexcom"] = np.nan


def refeicoes_da(tabela: pd.DataFrame) -> pd.DataFrame:
    return refeicoes.refeicoes_participante(1, tabela, leitura.dias_validos(tabela))


def almoco_e_jantar(leituras: pd.Series, almoco: str = "12:00") -> pd.DataFrame:
    return tabela_com(leituras, {f"2024-01-01 {almoco}": "almoco", "2024-01-01 19:00": "jantar"})


def test_iauc_de_um_triangulo():
    minutos = np.arange(121)
    glicose = 160 - np.abs(minutos - 60)
    assert refeicoes.calcular_iauc(minutos, glicose, base=100) == pytest.approx(3600)


def test_leituras_nativas_recupera_os_instantes_originais():
    gerador = np.random.default_rng(0)
    antes = leituras_a_cada("2024-01-01 00:00", 5, gerador.integers(80, 200, 40))
    plano = leituras_a_cada("2024-01-01 03:20", 5, [120, 120, 120, 120, 125, 130, 135])
    depois_da_falha = leituras_a_cada("2024-01-01 05:02", 5, gerador.integers(80, 200, 40))
    serie = pd.concat([interpolar(pd.concat([antes, plano])), interpolar(depois_da_falha)])
    serie = serie.asfreq("min")

    nativas = leitura.leituras_nativas(serie)

    originais = pd.concat([antes, plano, depois_da_falha])
    pd.testing.assert_index_equal(nativas.index, originais.index, check_names=False)
    np.testing.assert_array_equal(nativas.to_numpy(), originais.to_numpy())


def test_refeicao_seguida_de_lanche_em_90_minutos_fica_fora():
    tabela = tabela_com(
        leituras_aleatorias(["2024-01-01"]),
        {"2024-01-01 12:00": "almoco", "2024-01-01 13:30": "lanche", "2024-01-01 19:00": "jantar"},
    )

    assert refeicoes_da(tabela)["tipo"].tolist() == ["jantar"]


def test_dias_validos_sao_numerados_a_partir_do_primeiro_dia_completo():
    leituras = leituras_aleatorias(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    chegada_ao_meio_dia = leituras[leituras.index >= "2024-01-01 12:00"]
    tabela = tabela_com(
        chegada_ao_meio_dia,
        {
            "2024-01-01 13:00": "almoco",
            "2024-01-01 19:00": "jantar",
            "2024-01-02 08:00": "cafe",
            "2024-01-02 19:00": "jantar",
            "2024-01-03 12:00": "almoco",
            "2024-01-04 08:00": "cafe",
            "2024-01-04 12:00": "almoco",
        },
    )

    dias = leitura.dias_validos(tabela)
    tabela_refeicoes = refeicoes_da(tabela)

    assert dias["dia"].tolist() == [1, 2]
    assert dias["data"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")]
    assert list(zip(tabela_refeicoes["tipo"], tabela_refeicoes["dia"], strict=True)) == [
        ("almoco", 0),
        ("jantar", 0),
        ("cafe", 1),
        ("jantar", 1),
        ("cafe", 2),
        ("almoco", 2),
    ]


def test_dia_coberto_so_pelo_libre_nao_e_valido():
    tabela = almoco_e_jantar(leituras_aleatorias(["2024-01-01"]))
    falha_do_dexcom(tabela, "2024-01-01 06:00", "2024-01-02 00:00")

    assert tabela["libre"].notna().mean() > leitura.COBERTURA_MINIMA_DIA
    assert leitura.dias_validos(tabela).empty


def test_glicose_pre_e_a_ultima_leitura_ate_a_refeicao():
    leituras = trocar_leituras(leituras_aleatorias(["2024-01-01"]), "2024-01-01 12:00", [111, 150])
    tabela = almoco_e_jantar(leituras, almoco="12:03")

    almoco = refeicoes_da(tabela).iloc[0]

    assert almoco["tipo"] == "almoco"
    assert almoco["glicose_pre"] == 111


@pytest.mark.parametrize(
    ("ultima_leitura", "fica_na_tabela"),
    [("2024-01-01 11:45", True), ("2024-01-01 11:35", False)],
)
def test_leitura_com_mais_de_20_minutos_nao_serve_de_base(ultima_leitura, fica_na_tabela):
    tabela = almoco_e_jantar(leituras_aleatorias(["2024-01-01"]))
    falha_do_dexcom(tabela, ultima_leitura, "2024-01-01 12:05")

    tabela_refeicoes = refeicoes_da(tabela)

    assert ("almoco" in tabela_refeicoes["tipo"].tolist()) == fica_na_tabela
    if fica_na_tabela:
        assert tabela_refeicoes["glicose_pre"].iloc[0] == tabela.loc[ultima_leitura, "dexcom"]


@pytest.mark.parametrize(
    ("proxima_leitura", "fica_na_tabela"),
    [("2024-01-01 12:35", True), ("2024-01-01 12:40", False)],
)
def test_janela_com_menos_de_90_por_cento_de_dexcom_tira_a_refeicao(
    proxima_leitura, fica_na_tabela
):
    tabela = almoco_e_jantar(leituras_aleatorias(["2024-01-01"]))
    falha_do_dexcom(tabela, "2024-01-01 12:25", proxima_leitura)

    tabela_refeicoes = refeicoes_da(tabela)

    assert ("almoco" in tabela_refeicoes["tipo"].tolist()) == fica_na_tabela


def test_curva_ainda_alta_depois_de_2_horas_nao_soma_area():
    subida = [100 + 5 * passo for passo in range(12)]
    descida = [160 - 5 * passo for passo in range(13)]
    alta_depois_das_2_horas = [220] * 6
    leituras = trocar_leituras(
        leituras_aleatorias(["2024-01-01"]),
        "2024-01-01 12:00",
        subida + descida + alta_depois_das_2_horas,
    )

    almoco = refeicoes_da(almoco_e_jantar(leituras)).iloc[0]

    assert almoco["iauc_dexcom"] == pytest.approx(3600)
    assert almoco["pico"] == 160


def test_curva_vai_do_minuto_0_ao_minuto_120_entre_leituras():
    patamar = [130] * 24
    leituras = trocar_leituras(
        leituras_aleatorias(["2024-01-01"]), "2024-01-01 12:00", [100, *patamar, 180]
    )

    almoco = refeicoes_da(almoco_e_jantar(leituras, almoco="12:02")).iloc[0]

    rampa_do_minuto_0_ao_3 = 3 * 30 / 2
    patamar_do_minuto_3_ao_118 = 115 * 30
    do_minuto_118_ao_120_subindo_para_150 = 2 * (30 + 50) / 2
    assert almoco["iauc_dexcom"] == pytest.approx(
        rampa_do_minuto_0_ao_3 + patamar_do_minuto_3_ao_118 + do_minuto_118_ao_120_subindo_para_150
    )


def test_curva_que_desce_abaixo_da_base_soma_so_a_area_acima():
    desce_ate_76 = [100, 96, 92, 88, 84, 80]
    sobe_ate_136 = [76, 86, 96, 106, 116, 126]
    volta_a_100 = [136, 130, 124, 118, 112, 106]
    fica_em_100 = [100] * 7
    leituras = trocar_leituras(
        leituras_aleatorias(["2024-01-01"]),
        "2024-01-01 12:00",
        desce_ate_76 + sobe_ate_136 + volta_a_100 + fica_em_100,
    )

    almoco = refeicoes_da(almoco_e_jantar(leituras)).iloc[0]

    cruza_a_base_no_minuto_42_e_chega_a_36_no_minuto_60 = 18 * 36 / 2
    desce_de_36_a_0_do_minuto_60_ao_90 = 30 * 36 / 2
    assert almoco["iauc_dexcom"] == pytest.approx(
        cruza_a_base_no_minuto_42_e_chega_a_36_no_minuto_60 + desce_de_36_a_0_do_minuto_60_ao_90
    )


def test_glicose_jejum_cgm_usa_so_as_leituras_das_5h_as_7h():
    gerador = np.random.default_rng(2)
    leituras = leituras_a_cada("2024-01-01 00:00", 5, gerador.integers(150, 250, 288))
    leituras = trocar_leituras(leituras, "2024-01-01 05:00", [100] * 24)
    tabela = almoco_e_jantar(leituras)

    dias = refeicoes.resumo_dias(1, tabela, leitura.dias_validos(tabela))

    assert dias["glicose_jejum_cgm"].tolist() == [100]
