import os
from pathlib import Path

import numpy as np
import pandas as pd

TIPOS_REFEICAO = ("cafe", "almoco", "jantar")
GRUPOS = ("saudavel", "pre_diabetes", "dm2")
MACROS = ("carbo", "proteina", "gordura", "fibra")

# Os dtypes não guardam unidades. horario em horas desde 0h, iauc em mg/dL·min nas 2 horas
# depois da refeição, pico e glicose em mg/dL, cobertura_cgm entre 0 e 1, prato igual à
# posição da linha no cardápio.
ENTRADA_SIMULADOR = {
    "participante": "int64",
    "dia": "int64",
    "tipo": "str",
    "horario": "float64",
    "carbo": "float64",
    "proteina": "float64",
    "gordura": "float64",
    "fibra": "float64",
    "kcal": "float64",
    "glicose_pre": "float64",
    "minutos_ativos_60": "float64",
}

REFEICOES = {
    **ENTRADA_SIMULADOR,
    "grupo": "str",
    "foto": "str",
    "iauc_dexcom": "float64",
    "iauc_libre": "float64",
    "pico": "float64",
}

DIAS = {
    "participante": "int64",
    "dia": "int64",
    "data": "datetime64[ns]",
    "cobertura_cgm": "float64",
    "n_refeicoes": "int64",
    "kcal": "float64",
    "proteina": "float64",
    "glicose_media": "float64",
    "glicose_jejum_cgm": "float64",
}

PERFIS = {
    "participante": "int64",
    "grupo": "str",
    "idade": "float64",
    "sexo": "str",
    "imc": "float64",
    "hba1c": "float64",
    "glicose_jejum_lab": "float64",
    "insulina_jejum": "float64",
    "glicose_media": "float64",
    "glicose_jejum_cgm": "float64",
    "sensibilidade_carbo": "float64",
    "kcal_dia": "float64",
    "proteina_dia": "float64",
    **{f"{coluna}_{tipo}": "float64" for tipo in TIPOS_REFEICAO for coluna in (*MACROS, "horario")},
}

CARDAPIO = {
    "prato": "int64",
    "tipo": "str",
    "carbo": "float64",
    "proteina": "float64",
    "gordura": "float64",
    "fibra": "float64",
    "kcal": "float64",
    "foto": "str",
    "participante_origem": "int64",
}

TABELAS = {"refeicoes": REFEICOES, "dias": DIAS, "perfis": PERFIS, "cardapio": CARDAPIO}

DIAS_HABITO = range(1, 7)
DIAS_PLANO = range(7, 10)

# Fórmula ADAG (Nathan et al., 2008), com HbA1c em %.
ADAG_MGDL_POR_PONTO_HBA1C = 28.7
ADAG_INTERCEPTO_MGDL = -46.7
# Apêndice A do plano: no CGMacros o jejum do Dexcom fica 12,5 mg/dL acima do laboratório.
VIES_JEJUM_CGM_MGDL = 12.5
BASE_TRIANGULO_IAUC_MIN = 90
MINUTOS_POR_DIA = 24 * 60
# Mediana, mínimo e máximo medidos no CGMacros (Apêndice A do plano).
HORAS_DESDE_REFEICAO_ANTERIOR = {
    "cafe": (8.3, 6.0, 10.5),
    "almoco": (4.2, 3.0, 6.5),
    "jantar": (5.9, 3.0, 8.5),
}


def pasta_dados() -> Path:
    return Path(os.environ.get("GLICOSE_DADOS", "data")).expanduser()


def conformar(tabela: pd.DataFrame, colunas: dict[str, str]) -> pd.DataFrame:
    return tabela[list(colunas)].astype(colunas).reset_index(drop=True)


def inclinacao(x: pd.Series, y: pd.Series) -> float:
    return float(np.polyfit(x, y, 1)[0])


def dados_sinteticos(n_participantes: int, semente: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(semente)
    cardapio = _cardapio_sintetico(rng, n_participantes)
    perfis_base = _perfis_base(rng, n_participantes)
    refeicoes = _refeicoes_sinteticas(rng, cardapio, perfis_base)
    dias = _dias(refeicoes)
    perfis = _perfis(perfis_base, refeicoes, dias)
    return {
        "refeicoes": conformar(refeicoes, REFEICOES),
        "dias": conformar(dias, DIAS),
        "perfis": conformar(perfis, PERFIS),
        "cardapio": conformar(cardapio, CARDAPIO),
    }


def _cardapio_sintetico(rng, n_participantes):
    pratos_por_tipo = 4
    linhas = []
    for tipo in TIPOS_REFEICAO:
        for _ in range(pratos_por_tipo):
            carbo, proteina, gordura = rng.uniform((10, 5, 3), (90, 45, 35))
            linhas.append(
                {
                    "tipo": tipo,
                    "carbo": carbo,
                    "proteina": proteina,
                    "gordura": gordura,
                    "fibra": rng.uniform(0, 12),
                    "kcal": 4 * carbo + 4 * proteina + 9 * gordura,
                    "participante_origem": int(rng.integers(1, n_participantes + 1)),
                }
            )
    cardapio = pd.DataFrame(linhas)
    cardapio["prato"] = range(len(cardapio))
    cardapio["foto"] = [f"sintetico/{p:03d}.jpg" for p in cardapio["prato"]]
    return cardapio


def _perfis_base(rng, n_participantes):
    grupos = [GRUPOS[i % len(GRUPOS)] for i in range(n_participantes)]
    hba1c_por_grupo = {"saudavel": 5.3, "pre_diabetes": 6.0, "dm2": 7.4}
    sensibilidade_por_grupo = {"saudavel": 25.0, "pre_diabetes": 35.0, "dm2": 50.0}
    hba1c = np.array([hba1c_por_grupo[g] for g in grupos]) + rng.normal(0, 0.15, n_participantes)
    glicose_estimada = ADAG_MGDL_POR_PONTO_HBA1C * hba1c + ADAG_INTERCEPTO_MGDL
    return pd.DataFrame(
        {
            "participante": np.arange(1, n_participantes + 1),
            "grupo": grupos,
            "idade": rng.integers(20, 70, n_participantes).astype(float),
            "sexo": rng.choice(["F", "M"], n_participantes),
            "imc": rng.uniform(19, 38, n_participantes),
            "hba1c": hba1c,
            "glicose_jejum_lab": glicose_estimada + rng.normal(0, 5, n_participantes),
            "insulina_jejum": rng.uniform(2, 25, n_participantes),
            "sensibilidade_real": [sensibilidade_por_grupo[g] for g in grupos],
        }
    )


def _horarios_do_dia(rng):
    horarios, horario = {}, 0.0
    for tipo, (mediana, minimo, maximo) in HORAS_DESDE_REFEICAO_ANTERIOR.items():
        horario += float(np.clip(rng.normal(mediana, 0.8), minimo, maximo))
        horarios[tipo] = horario
    return horarios


def _refeicoes_sinteticas(rng, cardapio, perfis_base):
    linhas = []
    for perfil in perfis_base.itertuples():
        jejum = perfil.glicose_jejum_lab + VIES_JEJUM_CGM_MGDL
        for dia in range(1, DIAS_PLANO.stop):
            horarios = _horarios_do_dia(rng)
            for tipo in TIPOS_REFEICAO:
                prato = cardapio[cardapio["tipo"] == tipo].sample(1, random_state=rng).iloc[0]
                minutos = float(rng.choice([0, 0, 0, 10, 20, 30]))
                glicose_pre = jejum + rng.normal(5, 8)
                iauc = max(
                    perfil.sensibilidade_real * prato.carbo - 15 * minutos + rng.normal(0, 150), 0.0
                )
                linhas.append(
                    {
                        "participante": perfil.participante,
                        "dia": dia,
                        "tipo": tipo,
                        "horario": horarios[tipo],
                        **prato[[*MACROS, "kcal", "foto"]].to_dict(),
                        "glicose_pre": glicose_pre,
                        "minutos_ativos_60": minutos,
                        "grupo": perfil.grupo,
                        "iauc_dexcom": iauc,
                        "iauc_libre": iauc * rng.normal(1.0, 0.1),
                        "pico": glicose_pre + 2 * iauc / BASE_TRIANGULO_IAUC_MIN,
                    }
                )
    return pd.DataFrame(linhas)


def _dias(refeicoes):
    dias = refeicoes.groupby(["participante", "dia"], as_index=False).agg(
        n_refeicoes=("tipo", "size"),
        kcal=("kcal", "sum"),
        proteina=("proteina", "sum"),
        glicose_jejum_cgm=("glicose_pre", "min"),
        iauc_dia=("iauc_dexcom", "sum"),
    )
    dias["glicose_media"] = dias["glicose_jejum_cgm"] + dias["iauc_dia"] / MINUTOS_POR_DIA
    dias["data"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(dias["dia"], unit="D")
    dias["cobertura_cgm"] = 0.95
    return dias


def _perfis(perfis_base, refeicoes, dias):
    habito = refeicoes[refeicoes["dia"].isin(DIAS_HABITO)]
    dias_habito = dias[dias["dia"].isin(DIAS_HABITO)]
    por_tipo = habito.pivot_table(
        index="participante", columns="tipo", values=[*MACROS, "horario"], aggfunc="mean"
    )
    por_tipo.columns = [f"{coluna}_{tipo}" for coluna, tipo in por_tipo.columns]
    resumo = dias_habito.groupby("participante").agg(
        glicose_media=("glicose_media", "mean"),
        glicose_jejum_cgm=("glicose_jejum_cgm", "mean"),
        kcal_dia=("kcal", "mean"),
        proteina_dia=("proteina", "mean"),
    )
    sensibilidade = pd.Series(
        {p: inclinacao(r["carbo"], r["iauc_dexcom"]) for p, r in habito.groupby("participante")},
        name="sensibilidade_carbo",
    )
    return (
        perfis_base.join(resumo, on="participante")
        .join(sensibilidade, on="participante")
        .join(por_tipo, on="participante")
    )
