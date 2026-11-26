import numpy as np
import pandas as pd

from glicose.esquema import CARDAPIO, MACROS, TIPOS_REFEICAO, conformar

KCAL_POR_GRAMA = {"carbo": 4.0, "proteina": 4.0, "gordura": 9.0}
# No CGMacros real o desvio entre kcal e os macros vai até 27% e depois salta para 32% a 500%.
DESVIO_MAXIMO_KCAL_DOS_MACROS = 0.3
# Abaixo disso o registro não é uma refeição, como um jantar de 1 g de fibra e 10 kcal.
KCAL_MINIMA_PRATO = 50.0
# Os gêmeos de digitação de um mesmo prato diferem cerca de 1 g em um macro.
TOLERANCIA_GEMEOS_G = 2.0


def kcal_dos_macros(pratos: pd.DataFrame) -> pd.Series:
    return sum(pratos[macro] * kcal for macro, kcal in KCAL_POR_GRAMA.items())


def kcal_confere_com_macros(pratos: pd.DataFrame) -> pd.Series:
    estimada = kcal_dos_macros(pratos)
    return (pratos["kcal"] - estimada).abs() <= DESVIO_MAXIMO_KCAL_DOS_MACROS * estimada


def fibra_cabe_no_carbo(pratos: pd.DataFrame) -> pd.Series:
    # A fibra é parte do carboidrato total, então fibra acima do carbo é erro de registro.
    return pratos["fibra"] <= pratos["carbo"]


def representantes(macros: np.ndarray) -> list[int]:
    escolhidos: list[int] = []
    for linha, valores in enumerate(macros):
        distancias = np.abs(macros[escolhidos] - valores).max(axis=1)
        if distancias.min(initial=np.inf) > TOLERANCIA_GEMEOS_G:
            escolhidos.append(linha)
    return escolhidos


def unir_gemeos(variantes: pd.DataFrame) -> pd.DataFrame:
    return variantes.iloc[representantes(variantes[list(MACROS)].to_numpy())]


def montar_cardapio(refeicoes: pd.DataFrame) -> pd.DataFrame:
    macros = list(MACROS)
    com_macros = refeicoes.dropna(subset=macros)
    # O CGMacros não traz nome de prato. Sem foto, não há como mostrar o que comer.
    com_foto = com_macros[com_macros["foto"].notna()]
    nao_zerados = com_foto[com_foto[macros].ne(0).any(axis=1)]
    validos = nao_zerados[
        kcal_confere_com_macros(nao_zerados)
        & (nao_zerados["kcal"] >= KCAL_MINIMA_PRATO)
        & fibra_cabe_no_carbo(nao_zerados)
    ]

    variantes = (
        validos.groupby(["tipo", *macros, "kcal"], sort=False)
        .agg(
            foto=("foto", "first"),
            participante_origem=("participante", "first"),
            vezes=("foto", "size"),
        )
        .reset_index()
        .sort_values("vezes", ascending=False, kind="stable")
    )
    pratos = variantes.groupby("tipo", group_keys=False, sort=False)[variantes.columns].apply(
        unir_gemeos
    )
    pratos["tipo"] = pd.Categorical(pratos["tipo"], categories=TIPOS_REFEICAO, ordered=True)
    pratos = pratos.sort_values(["tipo", *macros]).reset_index(drop=True)
    pratos["prato"] = pratos.index
    return conformar(pratos, CARDAPIO)
