import numpy as np
import pandas as pd

from glicose.cardapio import fibra_cabe_no_carbo
from glicose.esquema import DIAS_HABITO, MACROS, PERFIS, TIPOS_REFEICAO, conformar

COLUNAS_BIO = {
    "subject": "participante",
    "Age": "idade",
    "Gender": "sexo",
    "BMI": "imc",
    "A1c PDL (Lab)": "hba1c",
    "Fasting GLU - PDL (Lab)": "glicose_jejum_lab",
    "Insulin": "insulina_jejum",
}

PONTOS_DE_UMA_RETA_EXATA = 2
MINIMO_REFEICOES_RETA = PONTOS_DE_UMA_RETA_EXATA + 1
CAMINHADA = [f"caminhada_{tipo}" for tipo in TIPOS_REFEICAO]


def sensibilidade_carbo(refeicoes_pessoa: pd.DataFrame) -> float:
    if len(refeicoes_pessoa) < MINIMO_REFEICOES_RETA:
        return np.nan
    inclinacao, _ = np.polyfit(refeicoes_pessoa["carbo"], refeicoes_pessoa["iauc_dexcom"], 1)
    return float(inclinacao)


def montar_perfis(refeicoes: pd.DataFrame, dias: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    habito = refeicoes[refeicoes["dia"].isin(DIAS_HABITO)]
    habito = habito.assign(
        fibra=habito["fibra"].where(fibra_cabe_no_carbo(habito)),
        caminhada=habito["minutos_ativos_60"],
    )
    dias_habito = dias[dias["dia"].isin(DIAS_HABITO)]

    # O cabeçalho do bio.csv traz "Insulin " com um espaço no fim.
    bio = bio.rename(columns=str.strip).rename(columns=COLUNAS_BIO)[list(COLUNAS_BIO.values())]
    grupo = refeicoes.groupby("participante")["grupo"].first()
    resumo_dias = dias_habito.groupby("participante").agg(
        glicose_media=("glicose_media", "mean"),
        glicose_jejum_cgm=("glicose_jejum_cgm", "mean"),
        kcal_dia=("kcal", "mean"),
        proteina_dia=("proteina", "mean"),
    )
    sensibilidade = (
        habito.groupby("participante")[["carbo", "iauc_dexcom"]]
        .apply(sensibilidade_carbo)
        .rename("sensibilidade_carbo")
    )
    por_tipo = habito.pivot_table(
        index="participante",
        columns="tipo",
        values=[*MACROS, "horario", "caminhada"],
        aggfunc="mean",
    )
    por_tipo.columns = [f"{coluna}_{tipo}" for coluna, tipo in por_tipo.columns]

    perfis = (
        bio[bio["participante"].isin(dias_habito["participante"])]
        .join(grupo, on="participante")
        .join(resumo_dias, on="participante")
        .join(sensibilidade, on="participante")
        .join(por_tipo, on="participante")
        .reindex(columns=list(PERFIS))
    )
    mediana_do_grupo = perfis.groupby("grupo")[CAMINHADA].transform("median")
    perfis[CAMINHADA] = (
        perfis[CAMINHADA].fillna(mediana_do_grupo).fillna(perfis[CAMINHADA].median())
    )
    return conformar(perfis, PERFIS)
