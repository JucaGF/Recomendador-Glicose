import pandas as pd

from glicose.esquema import pasta_dados

COLUNAS = {
    "Dexcom GL": "dexcom",
    "Libre GL": "libre",
    "METs": "mets",
    "Meal Type": "refeicao",
    "Carbs": "carbo",
    "Protein": "proteina",
    "Fat": "gordura",
    "Fiber": "fibra",
    "Calories": "kcal",
    "Image path": "foto",
}

TIPOS_REFEICAO = {"breakfast": "cafe", "lunch": "almoco", "dinner": "jantar"}
LANCHE = "lanche"

PARTICIPANTES_EXCLUIDOS = {7: "grade de 3 minutos no lugar de 1 minuto"}

# O CSV guarda METs multiplicados por 10, e o repouso aparece como 10.
ESCALA_METS = 10

# Limites de HbA1c da American Diabetes Association, em %.
A1C_PRE_DIABETES = 5.7
A1C_DM2 = 6.5

# Leituras nativas são inteiras, então a inclinação muda pelo menos 1/15 mg/dL por minuto
# num cotovelo do Libre. Segundas diferenças menores vêm do arredondamento da interpolação.
TOLERANCIA_COTOVELO = 1e-3

MINUTOS_DIA = 24 * 60
# Um dia com menos de 80% de glicose ou menos de 2 refeições registradas não mostra o hábito
# do dia inteiro. O dia de chegada tem no máximo 68% e sempre fica de fora.
COBERTURA_MINIMA_DIA = 0.8
REFEICOES_MINIMAS_DIA = 2


def participantes() -> list[int]:
    pastas = sorted((pasta_dados() / "CGMacros").glob("CGMacros-0*"))
    ids = [int(pasta.name.removeprefix("CGMacros-")) for pasta in pastas if pasta.is_dir()]
    return [participante for participante in ids if participante not in PARTICIPANTES_EXCLUIDOS]


def ler_grupos() -> pd.Series:
    biometria = pd.read_csv(pasta_dados() / "CGMacros" / "bio.csv")
    a1c = biometria.set_index("subject")["A1c PDL (Lab)"]
    grupos = pd.cut(
        a1c,
        [-float("inf"), A1C_PRE_DIABETES, A1C_DM2, float("inf")],
        right=False,
        labels=["saudavel", "pre_diabetes", "dm2"],
    )
    return grupos.astype("str").rename("grupo")


def ler_participante(participante: int) -> pd.DataFrame:
    pasta = pasta_dados() / "CGMacros" / f"CGMacros-{participante:03d}"
    bruto = pd.read_csv(pasta / f"CGMacros-{participante:03d}.csv")
    bruto.columns = bruto.columns.str.strip()
    tabela = bruto.reindex(columns=list(COLUNAS)).rename(columns=COLUNAS)
    tabela.index = pd.DatetimeIndex(pd.to_datetime(bruto["Timestamp"]), name="instante")
    tabela = tabela.asfreq("min")
    tabela["mets"] = tabela["mets"] / ESCALA_METS
    tabela["refeicao"] = tabela["refeicao"].map(_tipo_refeicao, na_action="ignore")
    tabela["foto"] = (f"CGMacros/CGMacros-{participante:03d}/" + tabela["foto"]).where(
        tabela["refeicao"].notna()
    )
    return tabela


def _tipo_refeicao(rotulo: str) -> str:
    rotulo = rotulo.strip().lower()
    return LANCHE if rotulo.startswith("snack") else TIPOS_REFEICAO[rotulo]


def leituras_nativas(serie: pd.Series) -> pd.Series:
    """Leituras reais do sensor dentro de uma série interpolada minuto a minuto.

    A interpolação é linear entre leituras, então a segunda diferença só sai de zero nas
    leituras reais. Uma leitura colinear com as vizinhas não forma cotovelo, por isso a série
    usa o passo mais comum entre cotovelos, e cada trecho contínuo usa a fase dos seus, para
    recuperar todas.
    """
    minuto = pd.Series((serie.index - serie.index[0]) // pd.Timedelta(minutes=1), index=serie.index)
    tem_valor = serie.notna()
    numero_do_trecho = tem_valor.ne(tem_valor.shift()).cumsum()
    segunda_diferenca = serie.diff().diff().shift(-1)
    e_cotovelo = segunda_diferenca.abs() > TOLERANCIA_COTOVELO
    minuto_do_cotovelo = minuto[e_cotovelo]
    trecho_do_cotovelo = numero_do_trecho[e_cotovelo]
    mesmo_trecho_do_anterior = trecho_do_cotovelo.eq(trecho_do_cotovelo.shift())
    distancia_entre_cotovelos = minuto_do_cotovelo.diff()[mesmo_trecho_do_anterior]
    if distancia_entre_cotovelos.empty:
        return serie.iloc[:0]
    passo = mais_comum(distancia_entre_cotovelos)
    fase_do_cotovelo = minuto_do_cotovelo % passo
    fase_por_trecho = fase_do_cotovelo.groupby(trecho_do_cotovelo).agg(mais_comum)
    fase_do_minuto = numero_do_trecho.map(fase_por_trecho)
    return serie[tem_valor & (minuto % passo == fase_do_minuto)]


def mais_comum(valores: pd.Series) -> int:
    return int(valores.mode().min())


def dias_validos(tabela: pd.DataFrame) -> pd.DataFrame:
    principais = tabela["refeicao"].isin(TIPOS_REFEICAO.values())
    com_cgm = tabela[["dexcom", "libre"]].notna().any(axis=1)
    data = tabela.index.normalize()
    dias = pd.DataFrame(
        {
            "cobertura_cgm": com_cgm.groupby(data).sum() / MINUTOS_DIA,
            "n_refeicoes": principais.groupby(data).sum(),
        }
    )
    dias.index.name = "data"
    validos = dias[
        (dias["cobertura_cgm"] > COBERTURA_MINIMA_DIA)
        & (dias["n_refeicoes"] >= REFEICOES_MINIMAS_DIA)
    ].reset_index()
    validos.insert(0, "dia", range(1, len(validos) + 1))
    return validos
