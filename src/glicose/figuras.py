"""Figuras e tabelas finais a partir de `resultados.parquet` e `curvas.parquet`.

Compara as estratégias por grupo clínico, o AG com a busca aleatória e o ótimo
exato, e mostra macros, caminhada e as frentes de Pareto. Grava os PNGs em
`artefatos/figuras/`.

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from glicose.esquema import GRUPOS, MACROS, TIPOS_REFEICAO
from glicose.experimentos import (
    ARQUIVO_CURVAS,
    ARQUIVO_RESULTADOS,
    DIA_ALVO,
    ESTRATEGIAS,
    FRENTE,
    vitrine,
)
from glicose.recomendar import NOMES_GRUPO, NOMES_TIPO, desenhar_grade
from glicose.refeicoes import ARTEFATOS
from glicose.simulador import IAUC_MINIMA

PASTA_FIGURAS = ARTEFATOS / "figuras"
CHAVE_PLANO = ["participante", "estrategia", "semente", "plano"]
SEMENTE_EXIBIDA = 0
# A rede calcula em float32, e o mesmo plano pode variar no último dígito entre lotes.
TOLERANCIA_EMPATE = 1e-4
OTIMIZADORES = ("menos_carbo_caminhada", "aleatoria", "ag", "nsga2_joelho", FRENTE)
ESTRATEGIAS_MACROS = ("habito", "menos_carbo", "aleatoria", "ag")
ESTRATEGIAS_PISO = ("real", "menos_carbo_caminhada", "aleatoria", "ag", "otimo")

DPI = 120
POLEGADAS_POR_PAINEL = 4.5
ALTURA_FIGURA = 4.5
POLEGADAS_POR_COLUNA_TABELA = 3.6
POLEGADAS_POR_LINHA_TABELA = 0.9
ESCALA_LINHA_TABELA = 3.5
LARGURA_GRADE = 9.6
ALTURA_GRADE = 12
FRACAO_OCUPADA_PELAS_BARRAS = 0.8
ESPACO_ENTRE_BARRAS = 0.95
ESPESSURA_LINHA = 2
TRANSPARENCIA_DESVIO = 0.2
TAMANHO_PONTO_FRENTE = 4
TAMANHO_MARCADOR = 11
CINZA_REFERENCIA = "0.5"
ROTACAO_ROTULOS = 20
LARGURA_FIGURA_MUDANCA = 6.5

PALETA = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#8c564b")
COR_ESTRATEGIA = dict(zip(ESTRATEGIAS, PALETA, strict=True))
COR_GRUPO = dict(zip(GRUPOS, PALETA, strict=False))
NOMES_ESTRATEGIA = {
    "real": "real (D9 registrado)",
    "habito": "hábito",
    "menos_carbo": "menos carbo",
    "menos_carbo_caminhada": "menos carbo + 30 min",
    "aleatoria": "busca aleatória",
    "ag": "AG",
    "nsga2_joelho": "NSGA-II (joelho)",
    "otimo": "ótimo exato",
}
NOMES_MACRO = {"carbo": "carbo", "proteina": "proteína", "gordura": "gordura", "fibra": "fibra"}


def planos(resultados: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por plano, sem repetir as refeições dele."""
    return resultados.drop_duplicates(CHAVE_PLANO)


def media_por_pessoa(tabela: pd.DataFrame, colunas: list[str], por: list[str]) -> pd.DataFrame:
    """Média das colunas primeiro por pessoa e depois por grupo, para cada pessoa pesar igual.

    Args:
        tabela: Tabela com a coluna `participante`.
        colunas: Colunas a promediar.
        por: Colunas que definem os grupos finais.
    """
    por_pessoa = tabela.groupby([*por, "participante"])[colunas].mean()
    return por_pessoa.groupby(por).mean()


def tabela_por_grupo(resultados: pd.DataFrame) -> pd.DataFrame:
    """Métricas de cada estratégia, por grupo clínico, médias entre pessoas.

    Returns:
        Tabela indexada por `grupo` e `estrategia`.
    """
    todos = planos(resultados)
    todos = todos[todos["estrategia"].isin(ESTRATEGIAS)].assign(
        viola=lambda t: (t["dias_fora_kcal"] + t["dias_fora_proteina"]) > 0,
        avaliacoes=lambda t: t["avaliacoes"].astype("float64"),
    )
    metricas = [
        "iauc_d9",
        "iauc_d9_sem_caminhada",
        "viola",
        "dias_fora_kcal",
        "dias_fora_proteina",
        "distancia_habito",
        "aptidao",
        "avaliacoes",
    ]
    por_pessoa = todos.groupby(["grupo", "estrategia", "participante"])[metricas].mean()
    tabela = por_pessoa.groupby(["grupo", "estrategia"]).agg(
        participantes=("iauc_d9", "size"),
        iauc_d9=("iauc_d9", "mean"),
        desvio=("iauc_d9", "std"),
        iauc_d9_sem_caminhada=("iauc_d9_sem_caminhada", "mean"),
        planos_que_violam=("viola", "mean"),
        dias_fora_kcal=("dias_fora_kcal", "mean"),
        dias_fora_proteina=("dias_fora_proteina", "mean"),
        distancia_habito=("distancia_habito", "mean"),
        aptidao=("aptidao", "mean"),
        avaliacoes=("avaliacoes", "mean"),
    )
    ordem = pd.MultiIndex.from_product([GRUPOS, ESTRATEGIAS], names=["grupo", "estrategia"])
    return tabela.reindex(ordem).dropna(how="all")


def caminhada_depois_da_troca(resultados: pd.DataFrame) -> pd.DataFrame:
    """Parte do ganho de iAUC do AG sobre o hábito que vem da caminhada a mais, por grupo."""
    # O efeito da caminhada depende do prato e a previsão para no piso, então a parte mudaria
    # se a conta trocasse primeiro a caminhada. Ela fica negativa para quem o prato do AG deixa
    # sem margem para a caminhada baixar a iAUC.
    todos = planos(resultados)
    por_pessoa = (
        todos[todos["estrategia"].isin(["ag", "habito"])]
        .groupby(["grupo", "estrategia", "participante"])[["iauc_d9", "iauc_d9_sem_caminhada"]]
        .mean()
    )
    ag = por_pessoa.xs("ag", level="estrategia")
    habito = por_pessoa.xs("habito", level="estrategia")
    ganho = habito["iauc_d9"] - ag["iauc_d9"]
    a_mais = (ag["iauc_d9_sem_caminhada"] - ag["iauc_d9"]) - (
        habito["iauc_d9_sem_caminhada"] - habito["iauc_d9"]
    )
    por_grupo = pd.DataFrame({"ganho": ganho, "a_mais": a_mais}).groupby("grupo")
    tabela = pd.DataFrame(
        {
            "parte": por_grupo["a_mais"].mean() / por_grupo["ganho"].mean(),
            "pessoas": por_grupo.size(),
            "pessoas_com_parte_negativa": (a_mais < 0).groupby("grupo").sum(),
        }
    )
    return tabela.reindex([g for g in GRUPOS if g in tabela.index])


def so_com_real(resultados: pd.DataFrame) -> pd.DataFrame:
    """Limita a tabela às pessoas que têm o D9 real comparável."""
    com_real = resultados.loc[resultados["estrategia"] == "real", "participante"].unique()
    return resultados[resultados["participante"].isin(com_real)]


def placar(ag: pd.Series, aleatoria: pd.Series) -> pd.Series:
    """Fração de vitórias, empates e derrotas do AG sobre a busca aleatória."""
    # No piso do simulador os dois chegam à mesma iAUC sem que nenhum tenha achado plano melhor.
    empata = np.isclose(ag, aleatoria, rtol=TOLERANCIA_EMPATE, atol=0)
    return pd.Series(
        {
            "vence": ((ag < aleatoria) & ~empata).mean(),
            "empata": empata.mean(),
            "perde": ((ag > aleatoria) & ~empata).mean(),
        }
    )


def vitorias_ag(resultados: pd.DataFrame) -> pd.DataFrame:
    """Placar do AG contra a busca aleatória, por execução e por participante, em três
    métricas.
    """
    disputa = planos(resultados)
    disputa = disputa[disputa["estrategia"].isin(["ag", "aleatoria"])]
    linhas = {}
    for coluna in ("aptidao", "iauc_d9", "iauc_d9_sem_caminhada"):
        por_semente = disputa.pivot_table(
            index=["participante", "semente"], columns="estrategia", values=coluna
        )
        por_pessoa = por_semente.groupby(level="participante").mean()
        linhas[(coluna, "por_execucao")] = placar(por_semente["ag"], por_semente["aleatoria"])
        linhas[(coluna, "por_participante")] = placar(por_pessoa["ag"], por_pessoa["aleatoria"])
    return pd.DataFrame(linhas).T.rename_axis(["metrica", "nivel"])


def macros_d9(resultados: pd.DataFrame, estrategias=ESTRATEGIAS_MACROS) -> pd.DataFrame:
    """Gramas de cada macro em D9, média por pessoa, por grupo e estratégia."""
    d9 = resultados[(resultados["dia"] == DIA_ALVO) & resultados["estrategia"].isin(estrategias)]
    por_plano = d9.groupby([*CHAVE_PLANO, "grupo"], dropna=False)[list(MACROS)].sum()
    return media_por_pessoa(por_plano.reset_index(), list(MACROS), ["grupo", "estrategia"])


def no_piso(resultados: pd.DataFrame) -> pd.DataFrame:
    """Fração de refeições de D9 cuja iAUC prevista está no piso do simulador."""
    d9 = resultados[
        (resultados["dia"] == DIA_ALVO) & resultados["estrategia"].isin(ESTRATEGIAS_PISO)
    ].assign(no_piso=lambda t: t["iauc"] <= IAUC_MINIMA)
    fracao = media_por_pessoa(d9, ["no_piso"], ["estrategia", "grupo"])["no_piso"]
    fracao = fracao.unstack("grupo").reindex(list(ESTRATEGIAS_PISO))
    return fracao[[g for g in GRUPOS if g in fracao.columns]]


def distancia_do_otimo(resultados: pd.DataFrame) -> pd.DataFrame:
    """Quanto a aptidão de cada estratégia passa da do ótimo exato, em fração."""
    todos = planos(resultados)
    otimo = todos[todos["estrategia"] == "otimo"].set_index("participante")["aptidao"]
    melhor_de_cada_execucao = (
        todos[todos["estrategia"].isin(OTIMIZADORES)]
        .groupby(["estrategia", "participante", "semente"], dropna=False)["aptidao"]
        .min()
        .reset_index()
    )
    excesso = (
        melhor_de_cada_execucao["aptidao"] / melhor_de_cada_execucao["participante"].map(otimo) - 1
    )
    melhor_de_cada_execucao = melhor_de_cada_execucao.assign(
        excesso=excesso, no_otimo=excesso <= TOLERANCIA_EMPATE
    )
    return (
        melhor_de_cada_execucao.groupby("estrategia")
        .agg(
            excesso_mediano=("excesso", "median"),
            excesso_medio=("excesso", "mean"),
            excesso_maximo=("excesso", "max"),
            execucoes_no_otimo=("no_otimo", "mean"),
        )
        .reindex(list(OTIMIZADORES))
    )


def mudanca_por_dia(resultados: pd.DataFrame, estrategias=("ag", "otimo")) -> pd.DataFrame:
    """Fração de pratos de cada dia diferentes dos do hábito."""
    chave = ["participante", "dia", "tipo"]
    habito = resultados.loc[resultados["estrategia"] == "habito", [*chave, "prato"]]
    outras = resultados[resultados["estrategia"].isin(estrategias)]
    juntos = outras.merge(habito, on=chave, suffixes=("", "_habito"))
    juntos = juntos.assign(prato_trocado=juntos["prato"] != juntos["prato_habito"])
    trocados = media_por_pessoa(juntos, ["prato_trocado"], ["dia", "estrategia"])
    return trocados["prato_trocado"].unstack("estrategia")[list(estrategias)]


def real_previsto_e_medido(resultados: pd.DataFrame) -> pd.Series:
    """Compara a iAUC prevista e a medida do D9 real, por pessoa e por refeição.

    Returns:
        Médias, erro absoluto médio e correlações.
    """
    reais = resultados[(resultados["estrategia"] == "real") & (resultados["dia"] == DIA_ALVO)]
    por_pessoa = reais.groupby("participante")[["iauc", "iauc_medida"]].sum()
    return pd.Series(
        {
            "participantes": len(por_pessoa),
            "previsto_medio": por_pessoa["iauc"].mean(),
            "medido_medio": por_pessoa["iauc_medida"].mean(),
            "erro_absoluto_medio": (por_pessoa["iauc"] - por_pessoa["iauc_medida"]).abs().mean(),
            "pearson": por_pessoa["iauc"].corr(por_pessoa["iauc_medida"]),
            "pearson_por_refeicao": reais["iauc"].corr(reais["iauc_medida"]),
        }
    )


def ganho_das_frentes(resultados: pd.DataFrame) -> pd.DataFrame:
    """Quanto a iAUC de D9 cai do plano mais perto do hábito ao mais longe dele, por grupo."""
    frentes = planos(resultados)
    frentes = frentes[frentes["estrategia"] == FRENTE]
    linhas = []
    for (participante, grupo, semente), frente in frentes.groupby(
        ["participante", "grupo", "semente"]
    ):
        pequena = frente.loc[frente["distancia_habito"].idxmin()]
        grande = frente.loc[frente["distancia_habito"].idxmax()]
        linhas.append(
            {
                "participante": participante,
                "grupo": grupo,
                "semente": semente,
                "planos": len(frente),
                "ganho": pequena["iauc_d9"] - grande["iauc_d9"],
            }
        )
    return media_por_pessoa(pd.DataFrame(linhas), ["planos", "ganho"], ["grupo"])


def figura_curvas(curvas: pd.DataFrame, destaque: list[int]) -> Figure:
    """Evolução da melhor e da média da população do AG, por participante em destaque."""
    figura = Figure(
        figsize=(POLEGADAS_POR_PAINEL * len(destaque), ALTURA_FIGURA), layout="constrained"
    )
    eixos = np.atleast_1d(figura.subplots(1, len(destaque)))
    ag = curvas[curvas["estrategia"] == "ag"]
    cor = COR_ESTRATEGIA["ag"]
    for eixo, participante in zip(eixos, destaque, strict=True):
        dela = ag[ag["participante"] == participante]
        for coluna, rotulo, estilo in (
            ("melhor", "melhor da população", "-"),
            ("media", "média da população", "--"),
        ):
            por_semente = dela.pivot(index="geracao", columns="semente", values=coluna).ffill()
            media, desvio = por_semente.mean(axis=1), por_semente.std(axis=1).fillna(0)
            eixo.plot(
                media.index, media, estilo, color=cor, linewidth=ESPESSURA_LINHA, label=rotulo
            )
            eixo.fill_between(
                media.index, media - desvio, media + desvio, color=cor, alpha=TRANSPARENCIA_DESVIO
            )
        grupo = NOMES_GRUPO[dela["grupo"].iloc[0]]
        sementes = dela["semente"].nunique()
        eixo.set_title(f"Participante {participante:03d} ({grupo}), {sementes} sementes")
        eixo.set_xlabel("geração")
        eixo.set_ylabel("aptidão (menor é melhor)")
        eixo.set_yscale("log")
        eixo.legend()
    figura.suptitle("Aptidão do AG por geração com o simulador real, média e desvio das sementes")
    return figura


def figura_tabela(tabela: pd.DataFrame) -> Figure:
    """Tabela de métricas por estratégia e grupo, como figura."""
    grupos = [g for g in GRUPOS if g in tabela.index.get_level_values("grupo")]
    estrategias = [e for e in ESTRATEGIAS if e in tabela.index.get_level_values("estrategia")]
    pessoas = tabela.groupby("grupo")["participantes"].max()
    celulas = []
    for estrategia in estrategias:
        linha = []
        for grupo in grupos:
            if (grupo, estrategia) not in tabela.index:
                linha.append("")
                continue
            valores = tabela.loc[(grupo, estrategia)]
            distancia = valores["distancia_habito"]
            desvio = "" if pd.isna(valores["desvio"]) else f" ± {valores['desvio']:.0f}"
            outras_pessoas = valores["participantes"] != pessoas[grupo]
            linha.append(
                f"{valores['iauc_d9']:.0f}{desvio}"
                f" (sem caminhada {valores['iauc_d9_sem_caminhada']:.0f})\n"
                f"{valores['planos_que_violam']:.0%} violam"
                + ("" if pd.isna(distancia) else f", distância {distancia:.2f}")
                + ("" if pd.isna(valores["aptidao"]) else f", aptidão {valores['aptidao']:.0f}")
                + (f", só {valores['participantes']:.0f} pessoas" if outras_pessoas else "")
            )
        celulas.append(linha)
    figura = Figure(
        figsize=(
            POLEGADAS_POR_COLUNA_TABELA * (len(grupos) + 1),
            POLEGADAS_POR_LINHA_TABELA * (len(estrategias) + 1),
        )
    )
    eixo = figura.subplots()
    eixo.axis("off")
    quadro = eixo.table(
        cellText=celulas,
        rowLabels=[NOMES_ESTRATEGIA[e] for e in estrategias],
        colLabels=[f"{NOMES_GRUPO[g]} (n={pessoas[g]:.0f})" for g in grupos],
        loc="center",
        cellLoc="center",
    )
    quadro.scale(1, ESCALA_LINHA_TABELA)
    eixo.set_title(
        "iAUC prevista de D9 (mg/dL·min, média ± desvio entre pessoas). O D9 real só existe "
        "para parte das pessoas\nfração de planos com algum dia fora da faixa de kcal ou "
        "proteína, distância do hábito em desvios-padrão e aptidão, que o AG minimiza"
    )
    return figura


def figura_ag_vs_aleatoria(resultados: pd.DataFrame, curvas: pd.DataFrame, destaque) -> Figure:
    """Dispersão do AG contra a busca aleatória e a curva de evolução."""
    figura = Figure(figsize=(3 * POLEGADAS_POR_PAINEL, ALTURA_FIGURA), layout="constrained")
    dispersao, evolucao = figura.subplots(1, 2)
    disputa = planos(resultados)
    disputa = disputa[disputa["estrategia"].isin(["ag", "aleatoria"])]
    por_pessoa = disputa.pivot_table(
        index=["participante", "grupo"], columns="estrategia", values="iauc_d9"
    ).reset_index()
    for grupo in GRUPOS:
        do_grupo = por_pessoa[por_pessoa["grupo"] == grupo]
        dispersao.scatter(
            do_grupo["aleatoria"], do_grupo["ag"], color=COR_GRUPO[grupo], label=NOMES_GRUPO[grupo]
        )
    valores = por_pessoa[["ag", "aleatoria"]]
    limites = [valores.min().min(), valores.max().max()]
    dispersao.plot(limites, limites, color=CINZA_REFERENCIA, linewidth=1, label="empate")
    resultado = placar(por_pessoa["ag"], por_pessoa["aleatoria"])
    dispersao.set_title(
        f"iAUC prevista de D9 por participante. O AG vence em {resultado['vence']:.0%},\n"
        f"empata em {resultado['empata']:.0%} e perde em {resultado['perde']:.0%}"
    )
    dispersao.set_xlabel("busca aleatória, mesmo número de avaliações (mg/dL·min)")
    dispersao.set_ylabel("AG (mg/dL·min)")
    dispersao.legend()

    estilos = {"ag": "-", "aleatoria": "--"}
    for participante in destaque:
        grupo = resultados.loc[resultados["participante"] == participante, "grupo"].iloc[0]
        for estrategia, estilo in estilos.items():
            dela = curvas[
                (curvas["participante"] == participante) & (curvas["estrategia"] == estrategia)
            ]
            media = dela.pivot(index="avaliacoes", columns="semente", values="melhor").ffill()
            evolucao.plot(
                media.index,
                media.mean(axis=1),
                estilo,
                color=COR_GRUPO[grupo],
                linewidth=ESPESSURA_LINHA,
                label=f"{participante:03d}, {NOMES_ESTRATEGIA[estrategia]}",
            )
    evolucao.set_yscale("log")
    evolucao.set_title(
        "Melhor aptidão contra avaliações (média das sementes)\n"
        "o AG começa com metade da população perto do hábito, a busca aleatória não"
    )
    evolucao.set_xlabel("avaliações do simulador")
    evolucao.set_ylabel("melhor aptidão (menor é melhor)")
    evolucao.legend()
    return figura


def barras_agrupadas(eixo, valores: pd.DataFrame, cores: dict, nomes: dict) -> None:
    """Barras lado a lado, uma cor por coluna, em um eixo."""
    largura = FRACAO_OCUPADA_PELAS_BARRAS / len(valores.columns)
    posicoes = np.arange(len(valores))
    for indice, coluna in enumerate(valores.columns):
        deslocamento = (indice - (len(valores.columns) - 1) / 2) * largura
        eixo.bar(
            posicoes + deslocamento,
            valores[coluna],
            largura * ESPACO_ENTRE_BARRAS,
            color=cores[coluna],
            label=nomes[coluna],
        )
    eixo.set_xticks(posicoes, valores.index)


def figura_macros(macros: pd.DataFrame) -> Figure:
    """Macros de D9 por estratégia e grupo."""
    figura = Figure(
        figsize=(POLEGADAS_POR_PAINEL * len(MACROS), ALTURA_FIGURA), layout="constrained"
    )
    eixos = figura.subplots(1, len(MACROS))
    for eixo, macro in zip(eixos, MACROS, strict=True):
        valores = macros[macro].unstack("estrategia")[list(ESTRATEGIAS_MACROS)]
        valores = valores.reindex([g for g in GRUPOS if g in valores.index])
        valores.index = [NOMES_GRUPO[g] for g in valores.index]
        barras_agrupadas(eixo, valores, COR_ESTRATEGIA, NOMES_ESTRATEGIA)
        eixo.set_title(NOMES_MACRO[macro])
        eixo.set_ylabel("gramas em D9 (média por pessoa)")
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    figura.legend(alcas, rotulos, loc="outside lower center", ncols=len(rotulos))
    figura.suptitle(
        "Macros de D9 por estratégia. O simulador nunca prevê menos glicose com mais carbo, e a "
        "faixa de kcal\nobriga a trocar o carbo por proteína e gordura. A busca aleatória, que "
        "não mira nenhum macro, chega aos mesmos macros do AG"
    )
    return figura


def figura_caminhada(resultados: pd.DataFrame, piso: pd.DataFrame) -> Figure:
    """Caminhada do AG em cada refeição e fração de refeições no piso."""
    figura = Figure(figsize=(3 * POLEGADAS_POR_PAINEL, ALTURA_FIGURA), layout="constrained")
    por_tipo, no_piso_do_simulador = figura.subplots(1, 2)
    d9 = resultados[(resultados["estrategia"] == "ag") & (resultados["dia"] == DIA_ALVO)]
    caminhada = media_por_pessoa(d9, ["caminhada"], ["tipo", "grupo"])["caminhada"]
    caminhada = caminhada.unstack("grupo").reindex(list(TIPOS_REFEICAO))
    caminhada = caminhada[[g for g in GRUPOS if g in caminhada.columns]]
    caminhada.index = [NOMES_TIPO[t] for t in caminhada.index]
    barras_agrupadas(por_tipo, caminhada, COR_GRUPO, NOMES_GRUPO)
    por_tipo.set_ylabel("minutos de caminhada (média por pessoa)")
    por_tipo.set_title(
        "Caminhada depois de cada refeição de D9 no plano do AG\n"
        "o simulador nunca prevê mais glicose com mais caminhada, então o AG vai ao limite"
    )
    por_tipo.legend(loc="lower center")

    piso = piso.dropna(how="all")
    piso.index = [NOMES_ESTRATEGIA[e] for e in piso.index]
    barras_agrupadas(no_piso_do_simulador, piso, COR_GRUPO, NOMES_GRUPO)
    no_piso_do_simulador.set_ylim(0, 1)
    no_piso_do_simulador.set_ylabel("fração das refeições de D9 (média por pessoa)")
    no_piso_do_simulador.set_title(
        "Refeições de D9 com iAUC prevista igual a 0, o piso do simulador\n"
        "nelas o prato e a caminhada não mudam mais a previsão"
    )
    no_piso_do_simulador.tick_params(axis="x", labelrotation=ROTACAO_ROTULOS)
    no_piso_do_simulador.legend()
    return figura


def figura_mudanca(mudanca: pd.DataFrame) -> Figure:
    """Fração de pratos trocados em relação ao hábito, por dia."""
    figura = Figure(figsize=(LARGURA_FIGURA_MUDANCA, ALTURA_FIGURA), layout="constrained")
    eixo = figura.subplots()
    mudanca.index = [f"D{d}" for d in mudanca.index]
    barras_agrupadas(eixo, mudanca, COR_ESTRATEGIA, NOMES_ESTRATEGIA)
    eixo.set_ylim(0, 1)
    eixo.set_ylabel("fração de pratos trocados (média por pessoa)")
    eixo.set_xlabel("dia do plano")
    eixo.set_title(
        "Pratos diferentes do hábito em cada dia\n"
        "em D7 e D8 o ótimo quase sempre fica no hábito,\ne o AG troca por ruído da busca"
    )
    eixo.legend()
    return figura


def figura_frentes(resultados: pd.DataFrame, destaque: list[int]) -> Figure:
    """Frente de Pareto de cada participante em destaque."""
    figura = Figure(
        figsize=(POLEGADAS_POR_PAINEL * len(destaque), ALTURA_FIGURA), layout="constrained"
    )
    eixos = np.atleast_1d(figura.subplots(1, len(destaque)))
    todos = planos(resultados)
    for eixo, participante in zip(eixos, destaque, strict=True):
        dela = todos[todos["participante"] == participante]
        dela = dela[dela["semente"].isna() | (dela["semente"] == SEMENTE_EXIBIDA)]
        frente = dela[dela["estrategia"] == FRENTE].sort_values("distancia_habito")
        eixo.plot(
            frente["iauc_d9"],
            frente["distancia_habito"],
            "o-",
            color=CINZA_REFERENCIA,
            markersize=TAMANHO_PONTO_FRENTE,
            label=f"frente do NSGA-II ({len(frente)} planos)",
        )
        for estrategia, marcador in (
            ("habito", "*"),
            ("ag", "s"),
            ("nsga2_joelho", "D"),
            ("otimo", "P"),
        ):
            ponto = dela[dela["estrategia"] == estrategia]
            eixo.plot(
                ponto["iauc_d9"],
                ponto["distancia_habito"],
                marcador,
                markersize=TAMANHO_MARCADOR,
                color=COR_ESTRATEGIA[estrategia],
                label=NOMES_ESTRATEGIA[estrategia],
            )
        grupo = NOMES_GRUPO[dela["grupo"].iloc[0]]
        eixo.set_title(f"Participante {participante:03d} ({grupo}), semente {SEMENTE_EXIBIDA}")
        eixo.set_xlabel("iAUC prevista de D9 (mg/dL·min)")
        eixo.set_ylabel("distância do hábito (desvios-padrão)")
        eixo.legend()
    figura.suptitle("Frente de Pareto real: glicose prevista contra mudança no hábito")
    return figura


def figura_planos(resultados: pd.DataFrame, destaque: list[int]) -> Figure:
    """Melhor plano do AG de cada participante em destaque, com as fotos."""
    figura = Figure(figsize=(LARGURA_GRADE, ALTURA_GRADE * len(destaque)), layout="constrained")
    subfiguras = np.atleast_1d(figura.subfigures(len(destaque), 1))
    for subfigura, participante in zip(subfiguras, destaque, strict=True):
        do_ag = resultados[
            (resultados["participante"] == participante) & (resultados["estrategia"] == "ag")
        ]
        melhor_semente = do_ag.loc[do_ag["aptidao"].idxmin(), "semente"]
        plano = do_ag[do_ag["semente"] == melhor_semente].sort_values(["dia", "horario"])
        grupo = NOMES_GRUPO[plano["grupo"].iloc[0]]
        titulo = (
            f"Participante {participante:03d} ({grupo}), melhor plano do AG em "
            f"{do_ag['semente'].nunique()} sementes, iAUC prevista de D9 "
            f"{plano['iauc_d9'].iloc[0]:.0f} mg/dL·min"
        )
        desenhar_grade(subfigura, plano, titulo)
    return figura


def gerar(resultados: pd.DataFrame, curvas: pd.DataFrame, pasta: Path = PASTA_FIGURAS) -> dict:
    """Gera e grava todas as figuras.

    Args:
        resultados: Tabela de `resultados.parquet`.
        curvas: Tabela de `curvas.parquet`.
        pasta: Destino dos PNGs.

    Returns:
        Dicionário `nome -> caminho gravado`.
    """
    destaque = vitrine(planos(resultados).groupby("participante")["grupo"].first())
    figuras = {
        "curvas_reais": figura_curvas(curvas, destaque),
        "tabela_grupos": figura_tabela(tabela_por_grupo(resultados)),
        "ag_vs_aleatoria_real": figura_ag_vs_aleatoria(resultados, curvas, destaque),
        "plausibilidade_macros": figura_macros(macros_d9(resultados)),
        "caminhada_refeicao": figura_caminhada(resultados, no_piso(resultados)),
        "d7_d8_vs_habito": figura_mudanca(mudanca_por_dia(resultados)),
        "frente_real": figura_frentes(resultados, destaque),
        "planos_tres_grupos": figura_planos(resultados, destaque),
    }
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    for nome, figura in figuras.items():
        caminhos[nome] = pasta / f"{nome}.png"
        figura.savefig(caminhos[nome], dpi=DPI, bbox_inches="tight")
    return caminhos


def imprimir(resultados: pd.DataFrame) -> None:
    """Imprime as tabelas de resumo no terminal."""
    colunas_tabela = [
        "participantes",
        "iauc_d9",
        "desvio",
        "iauc_d9_sem_caminhada",
        "planos_que_violam",
        "dias_fora_kcal",
        "dias_fora_proteina",
        "distancia_habito",
        "aptidao",
        "avaliacoes",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("iAUC prevista de D9 por grupo e estratégia (mg/dL·min), todos os participantes")
        print(tabela_por_grupo(resultados)[colunas_tabela].round(2).to_string())
        print("\nmesma tabela só com quem tem D9 real completo")
        print(tabela_por_grupo(so_com_real(resultados))[colunas_tabela].round(2).to_string())
        print(
            "\nparte do ganho do AG sobre o hábito que a caminhada a mais soma depois da troca de"
            "\npratos, por grupo. Trocando primeiro a caminhada a parte seria outra, e ela fica"
            "\nnegativa para quem o prato do AG deixa sem margem para a caminhada baixar"
        )
        print(caminhada_depois_da_troca(resultados).round(2).to_string())
        print("\nAG contra busca aleatória com o mesmo número de avaliações (fração das execuções")
        print("e das pessoas em que o AG vence, empata ou perde)")
        print(vitorias_ag(resultados).round(3).to_string())
        print("\naptidão de cada execução acima do ótimo exato (fração) e execuções no ótimo")
        print(distancia_do_otimo(resultados).round(3).to_string())
        print("\nmacros de D9 em gramas por estratégia")
        print(macros_d9(resultados).unstack("estrategia").round(1).to_string())
        print("\nfração das refeições de D9 com iAUC prevista no piso do simulador")
        print(no_piso(resultados).round(3).to_string())
        print("\nD9 real: iAUC prevista com a glicose de jejum contra a medida, soma do dia")
        print(real_previsto_e_medido(resultados).round(2).to_string())
        print("\nplano do AG contra o hábito, por dia")
        print(mudanca_por_dia(resultados).round(2).to_string())
        print("\nfrentes do NSGA-II: planos e ganho entre as pontas (mg/dL·min)")
        print(ganho_das_frentes(resultados).round(1).to_string())


def main() -> None:
    """Lê os resultados, imprime as tabelas e grava as figuras."""
    parser = argparse.ArgumentParser(description="Gera as figuras a partir dos resultados.")
    parser.add_argument("--pasta", type=Path, default=PASTA_FIGURAS)
    args = parser.parse_args()
    resultados = pd.read_parquet(ARQUIVO_RESULTADOS)
    curvas = pd.read_parquet(ARQUIVO_CURVAS)
    imprimir(resultados)
    for caminho in gerar(resultados, curvas, args.pasta).values():
        print(f"figura gravada em {caminho}")


if __name__ == "__main__":
    main()
