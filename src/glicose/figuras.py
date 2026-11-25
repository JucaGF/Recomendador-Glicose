import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from glicose.esquema import GRUPOS, HORAS_DESDE_REFEICAO_ANTERIOR, MACROS, TIPOS_REFEICAO
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

PASTA_FIGURAS = ARTEFATOS / "figuras"
CHAVE_PLANO = ["participante", "estrategia", "semente", "plano"]
SEMENTE_EXIBIDA = 0
TOLERANCIA_CAMINHADA_MIN = 1.0
ACASO_MAIOR_CAMINHADA = 1 / len(TIPOS_REFEICAO)
TOLERANCIA_LIMITE_HORAS = 1 / 60

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

PALETA = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2")
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
}
NOMES_MACRO = {"carbo": "carbo", "proteina": "proteína", "gordura": "gordura", "fibra": "fibra"}


def planos(resultados: pd.DataFrame) -> pd.DataFrame:
    return resultados.drop_duplicates(CHAVE_PLANO)


def media_por_pessoa(tabela: pd.DataFrame, colunas: list[str], por: list[str]) -> pd.DataFrame:
    por_pessoa = tabela.groupby([*por, "participante"])[colunas].mean()
    return por_pessoa.groupby(por).mean()


def tabela_por_grupo(resultados: pd.DataFrame) -> pd.DataFrame:
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
        avaliacoes=("avaliacoes", "mean"),
    )
    ordem = pd.MultiIndex.from_product([GRUPOS, ESTRATEGIAS], names=["grupo", "estrategia"])
    return tabela.reindex(ordem).dropna(how="all")


def so_com_real(resultados: pd.DataFrame) -> pd.DataFrame:
    com_real = resultados.loc[resultados["estrategia"] == "real", "participante"].unique()
    return resultados[resultados["participante"].isin(com_real)]


def vitorias_ag(resultados: pd.DataFrame) -> dict[str, float]:
    disputa = planos(resultados)
    disputa = disputa[disputa["estrategia"].isin(["ag", "aleatoria"])]
    vitorias = {}
    for coluna in ("aptidao", "iauc_d9"):
        por_semente = disputa.pivot_table(
            index=["participante", "semente"], columns="estrategia", values=coluna
        )
        por_pessoa = por_semente.groupby(level="participante").mean()
        vitorias[f"{coluna}_por_execucao"] = (por_semente["ag"] < por_semente["aleatoria"]).mean()
        vitorias[f"{coluna}_por_participante"] = (por_pessoa["ag"] < por_pessoa["aleatoria"]).mean()
    return vitorias


def macros_d9(resultados: pd.DataFrame, estrategias=("habito", "ag")) -> pd.DataFrame:
    d9 = resultados[(resultados["dia"] == DIA_ALVO) & resultados["estrategia"].isin(estrategias)]
    por_plano = d9.groupby([*CHAVE_PLANO, "grupo"], dropna=False)[list(MACROS)].sum()
    return media_por_pessoa(por_plano.reset_index(), list(MACROS), ["grupo", "estrategia"])


def caminhada_e_carbo(resultados: pd.DataFrame) -> pd.DataFrame:
    d9 = resultados[(resultados["estrategia"] == "ag") & (resultados["dia"] == DIA_ALVO)]
    linhas = []
    for (participante, grupo, semente), plano in d9.groupby(["participante", "grupo", "semente"]):
        maior = plano["caminhada"].max()
        no_maximo = plano["caminhada"] >= maior - TOLERANCIA_CAMINHADA_MIN
        linhas.append(
            {
                "participante": participante,
                "grupo": grupo,
                "semente": semente,
                "caminhada_unica": no_maximo.sum() == 1,
                "maior_caminhada_no_mais_carbo": no_maximo.loc[plano["carbo"].idxmax()],
            }
        )
    return pd.DataFrame(linhas)


def fracao_no_mais_carbo(coincidencias: pd.DataFrame) -> pd.Series:
    unicas = coincidencias[coincidencias["caminhada_unica"]]
    por_grupo = media_por_pessoa(unicas, ["maior_caminhada_no_mais_carbo"], ["grupo"])
    return por_grupo["maior_caminhada_no_mais_carbo"]


def mudanca_por_dia(resultados: pd.DataFrame) -> pd.DataFrame:
    chave = ["participante", "dia", "tipo"]
    habito = resultados.loc[resultados["estrategia"] == "habito", [*chave, "prato", "horario"]]
    ag = resultados[resultados["estrategia"] == "ag"]
    juntos = ag.merge(habito, on=chave, suffixes=("", "_habito"))
    juntos = juntos.assign(
        prato_trocado=juntos["prato"] != juntos["prato_habito"],
        horas_mais_tarde=juntos["horario"] - juntos["horario_habito"],
    )
    colunas = ["prato_trocado", "horas_mais_tarde", "caminhada"]
    return media_por_pessoa(juntos, colunas, ["dia"])


def horario_no_limite(resultados: pd.DataFrame) -> pd.Series:
    d9 = resultados[(resultados["estrategia"] == "ag") & (resultados["dia"] == DIA_ALVO)]
    _, minimo, maximo = zip(*d9["tipo"].map(HORAS_DESDE_REFEICAO_ANTERIOR), strict=True)
    no_limite = (d9["intervalo"] - np.array(minimo) < TOLERANCIA_LIMITE_HORAS) | (
        np.array(maximo) - d9["intervalo"] < TOLERANCIA_LIMITE_HORAS
    )
    return no_limite.groupby(d9["tipo"]).mean().reindex(list(TIPOS_REFEICAO))


def real_previsto_e_medido(resultados: pd.DataFrame) -> pd.Series:
    reais = resultados[(resultados["estrategia"] == "real") & (resultados["dia"] == DIA_ALVO)]
    por_pessoa = reais.groupby("participante")[["iauc", "iauc_medida"]].sum()
    return pd.Series(
        {
            "participantes": len(por_pessoa),
            "previsto_medio": por_pessoa["iauc"].mean(),
            "medido_medio": por_pessoa["iauc_medida"].mean(),
            "erro_absoluto_medio": (por_pessoa["iauc"] - por_pessoa["iauc_medida"]).abs().mean(),
            "pearson": por_pessoa["iauc"].corr(por_pessoa["iauc_medida"]),
        }
    )


def ganho_das_frentes(resultados: pd.DataFrame) -> pd.DataFrame:
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
    grupos = [g for g in GRUPOS if g in tabela.index.get_level_values("grupo")]
    estrategias = [e for e in ESTRATEGIAS if e in tabela.index.get_level_values("estrategia")]
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
            linha.append(
                f"{valores['iauc_d9']:.0f}{desvio}"
                f" (sem caminhada {valores['iauc_d9_sem_caminhada']:.0f})\n"
                f"{valores['planos_que_violam']:.0%} violam"
                + ("" if pd.isna(distancia) else f", distância {distancia:.2f}")
            )
        celulas.append(linha)
    pessoas = tabela.groupby("grupo")["participantes"].max()
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
        "iAUC prevista de D9 (mg/dL·min, média ± desvio entre pessoas), as mesmas pessoas em "
        "todas as linhas\nfração de planos com algum dia fora da faixa de kcal ou proteína e "
        "distância do hábito em desvios-padrão. O plano do hábito usa o prato mais parecido do "
        "cardápio e por isso muitas vezes sai da faixa"
    )
    return figura


def figura_ag_vs_aleatoria(resultados: pd.DataFrame, curvas: pd.DataFrame, destaque) -> Figure:
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
    venceu = (por_pessoa["ag"] < por_pessoa["aleatoria"]).mean()
    dispersao.set_title(f"iAUC prevista de D9 por participante. O AG vence em {venceu:.0%}")
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
    figura = Figure(
        figsize=(POLEGADAS_POR_PAINEL * len(MACROS), ALTURA_FIGURA), layout="constrained"
    )
    eixos = figura.subplots(1, len(MACROS))
    for eixo, macro in zip(eixos, MACROS, strict=True):
        valores = macros[macro].unstack("estrategia")[["habito", "ag"]]
        valores = valores.reindex([g for g in GRUPOS if g in valores.index])
        valores.index = [NOMES_GRUPO[g] for g in valores.index]
        barras_agrupadas(eixo, valores, COR_ESTRATEGIA, NOMES_ESTRATEGIA)
        eixo.set_title(NOMES_MACRO[macro])
        eixo.set_ylabel("gramas em D9 (média por pessoa)")
    eixos[0].legend()
    figura.suptitle("Macros de D9 no plano do AG contra o plano do hábito")
    return figura


def figura_caminhada(resultados: pd.DataFrame, coincidencias: pd.DataFrame) -> Figure:
    figura = Figure(figsize=(3 * POLEGADAS_POR_PAINEL, ALTURA_FIGURA), layout="constrained")
    por_tipo, coincide = figura.subplots(1, 2)
    d9 = resultados[(resultados["estrategia"] == "ag") & (resultados["dia"] == DIA_ALVO)]
    caminhada = media_por_pessoa(d9, ["caminhada"], ["tipo", "grupo"])["caminhada"]
    caminhada = caminhada.unstack("grupo").reindex(list(TIPOS_REFEICAO))
    caminhada = caminhada[[g for g in GRUPOS if g in caminhada.columns]]
    caminhada.index = [NOMES_TIPO[t] for t in caminhada.index]
    barras_agrupadas(por_tipo, caminhada, COR_GRUPO, NOMES_GRUPO)
    por_tipo.set_ylabel("minutos de caminhada (média por pessoa)")
    por_tipo.set_title("Caminhada depois de cada refeição de D9 no plano do AG")
    por_tipo.legend()

    fracao = fracao_no_mais_carbo(coincidencias)
    grupos = [g for g in GRUPOS if g in fracao.index]
    coincide.bar(
        [NOMES_GRUPO[g] for g in grupos], fracao[grupos], color=[COR_GRUPO[g] for g in grupos]
    )
    coincide.axhline(
        ACASO_MAIOR_CAMINHADA, color=CINZA_REFERENCIA, linestyle="--", label="acaso (1/3)"
    )
    coincide.set_ylim(0, 1)
    unicas = int(coincidencias["caminhada_unica"].sum())
    coincide.set_ylabel("fração dos planos (média por pessoa)")
    coincide.set_title(
        "A maior caminhada de D9 vem depois da refeição de mais carbo\n"
        f"só os {unicas} de {len(coincidencias)} planos com uma maior caminhada única"
    )
    coincide.legend()
    return figura


def figura_mudanca(mudanca: pd.DataFrame) -> Figure:
    rotulos = [f"D{d}" for d in mudanca.index]
    paineis = (
        ("prato_trocado", "fração de pratos trocados", "Pratos diferentes do hábito"),
        ("horas_mais_tarde", "horas (média com sinal)", "Quanto mais tarde que o hábito"),
        ("caminhada", "minutos (média)", "Caminhada (o hábito tem 0)"),
    )
    figura = Figure(
        figsize=(POLEGADAS_POR_PAINEL * len(paineis), ALTURA_FIGURA), layout="constrained"
    )
    eixos = figura.subplots(1, len(paineis))
    for eixo, (coluna, rotulo_y, titulo) in zip(eixos, paineis, strict=True):
        eixo.bar(rotulos, mudanca[coluna], color=COR_ESTRATEGIA["ag"])
        eixo.axhline(0, color=CINZA_REFERENCIA, linewidth=1)
        eixo.set_ylabel(rotulo_y)
        eixo.set_title(titulo)
        eixo.set_xlabel("dia do plano")
    figura.suptitle(
        "Plano do AG contra o hábito em cada dia. Em D7 e D8 só os pratos entram na aptidão,\n"
        "então horário e caminhada desses dias ficam iguais aos do hábito"
    )
    return figura


def figura_frentes(resultados: pd.DataFrame, destaque: list[int]) -> Figure:
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
        for estrategia, marcador in (("habito", "*"), ("ag", "s"), ("nsga2_joelho", "D")):
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
    figura = Figure(figsize=(LARGURA_GRADE, ALTURA_GRADE * len(destaque)), layout="constrained")
    subfiguras = np.atleast_1d(figura.subfigures(len(destaque), 1))
    for subfigura, participante in zip(subfiguras, destaque, strict=True):
        plano = resultados[
            (resultados["participante"] == participante)
            & (resultados["estrategia"] == "ag")
            & (resultados["semente"] == SEMENTE_EXIBIDA)
        ].sort_values(["dia", "horario"])
        grupo = NOMES_GRUPO[plano["grupo"].iloc[0]]
        titulo = (
            f"Participante {participante:03d} ({grupo}), plano do AG, iAUC prevista de D9 "
            f"{plano['iauc_d9'].iloc[0]:.0f} mg/dL·min"
        )
        desenhar_grade(subfigura, plano, titulo)
    return figura


def gerar(resultados: pd.DataFrame, curvas: pd.DataFrame, pasta: Path = PASTA_FIGURAS) -> dict:
    destaque = vitrine(planos(resultados).groupby("participante")["grupo"].first())
    figuras = {
        "curvas_reais": figura_curvas(curvas, destaque),
        "tabela_grupos": figura_tabela(tabela_por_grupo(so_com_real(resultados))),
        "ag_vs_aleatoria_real": figura_ag_vs_aleatoria(resultados, curvas, destaque),
        "plausibilidade_macros": figura_macros(macros_d9(resultados)),
        "caminhada_refeicao": figura_caminhada(resultados, caminhada_e_carbo(resultados)),
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
    colunas_tabela = [
        "participantes",
        "iauc_d9",
        "desvio",
        "iauc_d9_sem_caminhada",
        "planos_que_violam",
        "dias_fora_kcal",
        "dias_fora_proteina",
        "distancia_habito",
        "avaliacoes",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("iAUC prevista de D9 por grupo e estratégia (mg/dL·min), todos os participantes")
        print(tabela_por_grupo(resultados)[colunas_tabela].round(2).to_string())
        print("\nmesma tabela só com quem tem D9 real completo (a da figura)")
        print(tabela_por_grupo(so_com_real(resultados))[colunas_tabela].round(2).to_string())
        print("\nAG contra busca aleatória com o mesmo número de avaliações (fração de vitórias)")
        for nome, valor in vitorias_ag(resultados).items():
            print(f"  {nome}: {valor:.1%}")
        print("\nmacros de D9 em gramas, plano do hábito contra plano do AG")
        print(macros_d9(resultados).round(1).to_string())
        coincidencias = caminhada_e_carbo(resultados)
        unicas = int(coincidencias["caminhada_unica"].sum())
        print(
            "\nmaior caminhada de D9 depois da refeição de mais carbo, só nos "
            f"{unicas} de {len(coincidencias)} planos com uma maior caminhada única"
        )
        print(fracao_no_mais_carbo(coincidencias).round(2).to_string())
        print("\nfração das refeições de D9 do AG com o horário preso num extremo da busca")
        print(horario_no_limite(resultados).round(2).to_string())
        print("\nD9 real: iAUC prevista com a glicose de jejum contra a medida, soma do dia")
        print(real_previsto_e_medido(resultados).round(2).to_string())
        print("\nplano do AG contra o hábito, por dia")
        print(mudanca_por_dia(resultados).round(2).to_string())
        print("\nfrentes do NSGA-II: planos e ganho entre as pontas (mg/dL·min)")
        print(ganho_das_frentes(resultados).round(1).to_string())


def main() -> None:
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
