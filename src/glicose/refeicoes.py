from pathlib import Path

import numpy as np
import pandas as pd

from glicose import leitura
from glicose.esquema import DIAS, REFEICOES, conformar

ARTEFATOS = Path("artefatos")
# 2 horas é a janela da iAUC no método do índice glicêmico (ISO 26642).
MINUTOS_JANELA = 120
JANELA = pd.Timedelta(minutes=MINUTOS_JANELA)
# A atividade conta na primeira hora, enquanto a glicose ainda sobe. O pico mediano vem aos 67
# minutos.
JANELA_ATIVIDADE = pd.Timedelta(minutes=60)
# Com 90% da janela coberta, falta no máximo 12 minutos, menos de 3 leituras do Dexcom.
COBERTURA_MINIMA_JANELA = 0.9
# O Dexcom lê a cada 5 minutos. Uma leitura a mais de 20 minutos da borda da janela vem do outro
# lado de uma falha do sensor, e não da glicose naquele momento.
DISTANCIA_MAXIMA_DA_BORDA = pd.Timedelta(minutes=20)
# 3 METs é o limite de atividade moderada da OMS. Os 11 participantes que só têm Intensity
# ficam sem atividade, porque a escala do Intensity não se compara com METs.
METS_ATIVO = 3
# Às 5h a última refeição foi horas antes. A média dessa janela teve correlação de 0,81 com a
# glicose de jejum do laboratório no protótipo.
JEJUM = ("05:00", "07:00")
# Os dias antes de D1, como o de chegada, que é sempre parcial, ficam fora de D1 a D9. As
# refeições deles com janela completa ficam na tabela, e o treino do simulador decide se as usa.
DIA_ANTES_DO_ESTUDO = 0
MACROS = ["carbo", "proteina", "gordura", "fibra", "kcal"]


def calcular_iauc(minutos: np.ndarray, glicose: np.ndarray, base: float) -> float:
    alturas = np.asarray(glicose, dtype=float) - base
    larguras = np.diff(minutos)
    trechos = zip(larguras, alturas[:-1], alturas[1:], strict=True)
    return float(sum(area_acima_da_base(*trecho) for trecho in trechos))


def area_acima_da_base(largura: float, altura_inicial: float, altura_final: float) -> float:
    if altura_inicial >= 0 and altura_final >= 0:
        return largura * (altura_inicial + altura_final) / 2
    if altura_inicial <= 0 and altura_final <= 0:
        return 0.0
    # O trecho cruza a base. Só o triângulo acima dela conta, e a largura dele sai da semelhança
    # de triângulos.
    topo = max(altura_inicial, altura_final)
    fundo = min(altura_inicial, altura_final)
    largura_acima = largura * topo / (topo - fundo)
    return largura_acima * topo / 2


def janela_limpa(instantes_refeicoes: pd.DatetimeIndex, inicio: pd.Timestamp) -> bool:
    seguintes = (instantes_refeicoes > inicio) & (instantes_refeicoes <= inicio + JANELA)
    return not seguintes.any()


def resposta(nativas: pd.Series, grade: pd.Series, inicio: pd.Timestamp) -> dict | None:
    fim = inicio + JANELA
    cobertura = grade[inicio:fim].notna().mean()
    antes = nativas[:inicio]
    depois = nativas[fim:]
    if cobertura < COBERTURA_MINIMA_JANELA or antes.empty or depois.empty:
        return None
    ultima_antes, primeira_depois = antes.index[-1], depois.index[0]
    if inicio - ultima_antes > DISTANCIA_MAXIMA_DA_BORDA:
        return None
    if primeira_depois - fim > DISTANCIA_MAXIMA_DA_BORDA:
        return None
    base = antes.iloc[-1]
    minutos, glicose = curva_de_2_horas(nativas[ultima_antes:primeira_depois], inicio, base)
    return {"base": base, "iauc": calcular_iauc(minutos, glicose, base), "pico": glicose.max()}


def curva_de_2_horas(
    leituras: pd.Series, inicio: pd.Timestamp, base: float
) -> tuple[np.ndarray, np.ndarray]:
    minutos = ((leituras.index - inicio) / pd.Timedelta(minutes=1)).to_numpy()
    glicose = leituras.to_numpy()
    dentro = (minutos > 0) & (minutos < MINUTOS_JANELA)
    glicose_no_fim = np.interp(MINUTOS_JANELA, minutos, glicose)
    minutos_da_curva = np.concatenate([[0], minutos[dentro], [MINUTOS_JANELA]])
    glicose_da_curva = np.concatenate([[base], glicose[dentro], [glicose_no_fim]])
    return minutos_da_curva, glicose_da_curva


def minutos_ativos(tabela: pd.DataFrame, inicio: pd.Timestamp) -> float:
    janela = tabela[inicio + pd.Timedelta(minutes=1) : inicio + JANELA_ATIVIDADE]
    if janela["mets"].isna().all():
        return np.nan
    return float((janela["mets"] >= METS_ATIVO).sum())


def dia_do_estudo(dias: pd.DataFrame, inicio: pd.Timestamp) -> int | None:
    data = inicio.normalize()
    if data < dias["data"].min():
        return DIA_ANTES_DO_ESTUDO
    encontrado = dias.loc[dias["data"] == data, "dia"]
    return None if encontrado.empty else int(encontrado.iloc[0])


def refeicoes_participante(
    participante: int, tabela: pd.DataFrame, dias: pd.DataFrame
) -> pd.DataFrame:
    nativas = {sensor: leitura.leituras_nativas(tabela[sensor]) for sensor in ("dexcom", "libre")}
    registros = tabela[tabela["refeicao"].notna()]
    linhas = []
    for inicio, registro in registros.iterrows():
        dia = dia_do_estudo(dias, inicio)
        if registro["refeicao"] == leitura.LANCHE or dia is None:
            continue
        if not janela_limpa(registros.index, inicio):
            continue
        dexcom = resposta(nativas["dexcom"], tabela["dexcom"], inicio)
        if dexcom is None:
            continue
        libre = resposta(nativas["libre"], tabela["libre"], inicio)
        linhas.append(
            {
                "participante": participante,
                "dia": dia,
                "tipo": registro["refeicao"],
                "horario": inicio.hour + inicio.minute / 60,
                **registro[MACROS].to_dict(),
                "glicose_pre": dexcom["base"],
                "minutos_ativos_60": minutos_ativos(tabela, inicio),
                "foto": registro["foto"],
                "iauc_dexcom": dexcom["iauc"],
                "iauc_libre": np.nan if libre is None else libre["iauc"],
                "pico": dexcom["pico"],
            }
        )
    return pd.DataFrame(linhas)


def resumo_dias(participante: int, tabela: pd.DataFrame, dias: pd.DataFrame) -> pd.DataFrame:
    principais = tabela[tabela["refeicao"].isin(leitura.TIPOS_REFEICAO.values())]
    dexcom = leitura.leituras_nativas(tabela["dexcom"])
    jejum = dexcom.between_time(*JEJUM, inclusive="left")
    por_data = pd.DataFrame(
        {
            "kcal": principais["kcal"].groupby(principais.index.normalize()).sum(),
            "proteina": principais["proteina"].groupby(principais.index.normalize()).sum(),
            "glicose_media": dexcom.groupby(dexcom.index.normalize()).mean(),
            "glicose_jejum_cgm": jejum.groupby(jejum.index.normalize()).mean(),
        }
    )
    return dias.join(por_data, on="data").assign(participante=participante)


def montar_tabelas() -> tuple[pd.DataFrame, pd.DataFrame]:
    grupos = leitura.ler_grupos()
    refeicoes, dias = [], []
    for participante in leitura.participantes():
        tabela = leitura.ler_participante(participante)
        validos = leitura.dias_validos(tabela)
        refeicoes.append(refeicoes_participante(participante, tabela, validos))
        dias.append(resumo_dias(participante, tabela, validos))
    refeicoes = pd.concat(refeicoes, ignore_index=True)
    refeicoes["grupo"] = refeicoes["participante"].map(grupos)
    return conformar(refeicoes, REFEICOES), conformar(pd.concat(dias, ignore_index=True), DIAS)


def montar_refeicoes() -> pd.DataFrame:
    return montar_tabelas()[0]


def main() -> None:
    refeicoes, dias = montar_tabelas()
    ARTEFATOS.mkdir(exist_ok=True)
    refeicoes.to_parquet(ARTEFATOS / "refeicoes.parquet", index=False)
    dias.to_parquet(ARTEFATOS / "dias.parquet", index=False)
    print(
        f"{len(refeicoes)} refeicoes utilizaveis de {refeicoes['participante'].nunique()} "
        "participantes"
    )
    print(f"{len(dias)} dias validos de {dias['participante'].nunique()} participantes")


if __name__ == "__main__":
    main()
