import numpy as np
import pandas as pd

from glicose import leitura


def interpolar(leituras: pd.Series) -> pd.Series:
    grade = pd.date_range(leituras.index[0], leituras.index[-1], freq="min")
    return leituras.reindex(grade).interpolate(method="time", limit_area="inside")


def leituras_a_cada(inicio: str, passo: int, valores) -> pd.Series:
    instantes = pd.date_range(inicio, periods=len(valores), freq=f"{passo}min")
    return pd.Series(np.asarray(valores, dtype=float), index=instantes)


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
