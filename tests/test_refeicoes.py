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


def test_iauc_de_um_triangulo():
    minutos = np.arange(121)
    glicose = 160 - np.abs(minutos - 60)
    assert refeicoes.calcular_iauc(minutos, glicose, base=100) == pytest.approx(3600)


def test_leituras_nativas_recupera_os_instantes_originais():
    rng = np.random.default_rng(0)
    antes = leituras_a_cada("2024-01-01 00:00", 5, rng.integers(80, 200, 40))
    plano = leituras_a_cada("2024-01-01 03:20", 5, [120, 120, 120, 120, 125, 130, 135])
    depois_da_falha = leituras_a_cada("2024-01-01 05:02", 5, rng.integers(80, 200, 40))
    serie = pd.concat([interpolar(pd.concat([antes, plano])), interpolar(depois_da_falha)])
    serie = serie.asfreq("min")

    nativas = leitura.leituras_nativas(serie)

    originais = pd.concat([antes, plano, depois_da_falha])
    pd.testing.assert_index_equal(nativas.index, originais.index, check_names=False)
    np.testing.assert_array_equal(nativas.to_numpy(), originais.to_numpy())


def participante_com(refeicoes_do_dia: dict[str, str]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dexcom = interpolar(leituras_a_cada("2024-01-01 00:00", 5, rng.integers(90, 180, 288)))
    tabela = pd.DataFrame(index=pd.date_range("2024-01-01", periods=24 * 60, freq="min"))
    tabela["dexcom"] = dexcom
    tabela["libre"] = dexcom
    tabela["mets"] = 1.0
    tabela["intensidade"] = np.nan
    tabela["refeicao"] = None
    for horario, tipo in refeicoes_do_dia.items():
        tabela.loc[pd.Timestamp(f"2024-01-01 {horario}"), "refeicao"] = tipo
    for macro in refeicoes.MACROS:
        tabela[macro] = 30.0
    tabela["foto"] = None
    return tabela


def test_refeicao_seguida_de_lanche_em_90_minutos_fica_fora():
    tabela = participante_com({"12:00": "almoco", "13:30": "lanche", "19:00": "jantar"})

    tabela_refeicoes = refeicoes.refeicoes_participante(1, tabela, leitura.dias_validos(tabela))

    assert tabela_refeicoes["tipo"].tolist() == ["jantar"]
