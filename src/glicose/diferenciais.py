import argparse
import os
import time
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from glicose import ag, leitura, simulador
from glicose.ag import ConfigAG, Lambdas
from glicose.cardapio import kcal_dos_macros
from glicose.esquema import (
    DIAS_HABITO,
    DIAS_PLANO,
    ENTRADA_SIMULADOR,
    GRUPOS,
    MACROS,
    TIPOS_REFEICAO,
    conformar,
    pasta_dados,
)
from glicose.experimentos import dados, elegiveis, otimo_exato
from glicose.perfil import montar_perfis
from glicose.plano import (
    LIMITES_CAMINHADA,
    N_REFEICOES,
    Populacao,
    falta_proteina,
    habito,
    horarios,
    limites_intervalo,
    para_refeicoes,
)
from glicose.refeicoes import ARTEFATOS, JANELA, calcular_iauc

PASTA = ARTEFATOS / "diferenciais"
PROCESSOS = 3
SEMENTE = 0
DIA_ALVO = DIAS_PLANO[-1]
CONFIG = ConfigAG()
VERSOES = ("nativa", "grade")
VARIANTES = ("com_travas", "sem_travas", "ag_horario_fixo", "horario_livre")

GENES_LIVRES = (*MACROS, "minutos_ativos_60")
# A caixa de busca vai até o dobro do maior valor visto no treino, para o AG poder sair da faixa.
FOLGA_GRAMAS = 2.0
PASSO_HORARIO = 0.25
MINUTOS_POR_HORA = 60
PERCENTIL_GLICOSE_PRE = 90
LINEAR_ATE = 100

DPI = 120
POLEGADAS_POR_PAINEL = 4.5
NOMES_GRUPO = {"saudavel": "saudável", "pre_diabetes": "pré-diabetes", "dm2": "DM2"}
NOMES_TIPO = {"cafe": "café", "almoco": "almoço", "jantar": "jantar"}
NOMES_MACRO = {"carbo": "carbo", "proteina": "proteína", "gordura": "gordura", "fibra": "fibra"}
COR = {
    "com_travas": "#257de0",
    "sem_travas": "#f47230",
    "ag_horario_fixo": "#00b894",
    "horario_livre": "#6a4ce2",
    "nativa": "#257de0",
    "grade": "#f47230",
}
COR_GRUPO = dict(zip(GRUPOS, ("#00b894", "#f2a600", "#ee7fac"), strict=True))
CINZA = "0.6"

Simular = Callable[[pd.DataFrame], np.ndarray]


class Saida(NamedTuple):
    tabela: pd.DataFrame
    plano: Populacao | None
    segundos_cpu: float


def treino_do_simulador(refeicoes: pd.DataFrame, perfis: pd.DataFrame) -> pd.DataFrame:
    estudo = simulador.particao_do_estudo(refeicoes, perfis).refeicoes
    return estudo[estudo["dia"].isin(DIAS_HABITO)].dropna(subset=list(MACROS))


def horario_d9(perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    return horarios(habito(perfil, cardapio))[0, ag.POSICOES_D9]


def refeicoes_livres(genes: np.ndarray, perfil: pd.Series, horario: np.ndarray) -> pd.DataFrame:
    """genes tem forma (planos, refeições de D9, GENES_LIVRES)."""
    linhas = genes.reshape(-1, len(GENES_LIVRES))
    tabela = pd.DataFrame(linhas, columns=list(GENES_LIVRES)).assign(
        participante=perfil["participante"],
        dia=DIA_ALVO,
        tipo=np.tile(TIPOS_REFEICAO, len(genes)),
        horario=np.tile(horario, len(genes)),
        glicose_pre=perfil["glicose_jejum_cgm"],
    )
    return conformar(tabela.assign(kcal=kcal_dos_macros(tabela)), ENTRADA_SIMULADOR)


def aptidao_livre(
    genes: np.ndarray, simular: Simular, perfil: pd.Series, horario: np.ndarray, lambdas: Lambdas
) -> np.ndarray:
    iauc = np.asarray(simular(refeicoes_livres(genes, perfil, horario)))
    iauc_d9 = iauc.reshape(len(genes), len(TIPOS_REFEICAO)).sum(axis=1)
    proteina_d9 = genes[..., GENES_LIVRES.index("proteina")].sum(axis=1)
    return iauc_d9 + lambdas.proteina * falta_proteina(proteina_d9, perfil)


def mutacao_livre(
    genes: np.ndarray, sigma: float, taxa: float, limites: tuple, rng: np.random.Generator
) -> np.ndarray:
    minimo, maximo = limites
    ruido = rng.normal(0, sigma * (maximo - minimo), genes.shape)
    mutado = genes + np.where(rng.random(genes.shape) < taxa, ruido, 0)
    return np.clip(mutado, minimo, maximo)


def ag_sem_travas(
    config: ConfigAG,
    simular: Simular,
    perfil: pd.Series,
    cardapio: pd.DataFrame,
    limites: tuple[np.ndarray, np.ndarray],
    semente: int,
) -> np.ndarray:
    """O AG de P4 sem cardápio, sem P_kcal e sem distância do hábito. Só D9, a única parte que a
    iAUC mede. Devolve os genes do melhor plano, com forma (refeições de D9, GENES_LIVRES)."""
    rng = np.random.default_rng(semente)
    horario = horario_d9(perfil, cardapio)
    pratos = para_refeicoes(habito(perfil, cardapio), cardapio, perfil).iloc[ag.POSICOES_D9]
    do_habito = pratos[list(GENES_LIVRES)].to_numpy()
    metade = config.tamanho_populacao // 2
    aleatorios = rng.uniform(*limites, (metade, len(TIPOS_REFEICAO), len(GENES_LIVRES)))
    copias = np.repeat(do_habito[None], config.tamanho_populacao - metade, axis=0)
    copias = mutacao_livre(copias, ag.sigma(0, config), config.taxa_mutacao_gaussiana, limites, rng)
    pop = np.concatenate([aleatorios, copias])
    aptidoes = aptidao_livre(pop, simular, perfil, horario, config.lambdas)
    melhor_ate_agora, sem_melhora = aptidoes.min(), 0

    for geracao in range(1, config.max_geracoes + 1):
        elite = pop[np.argsort(aptidoes)[: config.elitismo]]
        n_filhos = config.tamanho_populacao - config.elitismo
        casais = np.array(
            [ag.torneio(aptidoes, config.tamanho_torneio, rng) for _ in range(n_filhos)]
        )
        do_pai = rng.random((n_filhos, len(TIPOS_REFEICAO), 1)) < ag.PROB_GENE_DO_PAI
        filhos = np.where(do_pai, pop[casais[:, 0]], pop[casais[:, 1]])
        filhos = mutacao_livre(
            filhos, ag.sigma(geracao, config), config.taxa_mutacao_gaussiana, limites, rng
        )
        pop = np.concatenate([elite, filhos])
        aptidoes = aptidao_livre(pop, simular, perfil, horario, config.lambdas)
        if aptidoes.min() < melhor_ate_agora:
            melhor_ate_agora, sem_melhora = aptidoes.min(), 0
        else:
            sem_melhora += 1
        if sem_melhora >= config.paciencia:
            break
    return pop[np.argmin(aptidoes)]


def limites_livres(treino: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    maximos = FOLGA_GRAMAS * treino[list(MACROS)].max().to_numpy()
    return np.zeros(len(GENES_LIVRES)), np.append(maximos, LIMITES_CAMINHADA[1])


def d9_do_plano(plano: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> pd.DataFrame:
    tabela = para_refeicoes(plano, cardapio, perfil)
    return tabela[tabela["dia"] == DIA_ALVO].reset_index(drop=True)


def rodar_variante(variante: str, participante: int, semente: int) -> Saida:
    inicio = time.process_time()
    tabelas = dados()
    perfil = tabelas["perfis"].loc[participante]
    cardapio = tabelas["cardapio"]
    sim = tabelas["simulador"]

    def simular(refeicoes):
        return sim.prever(perfil, refeicoes)

    plano = None
    if variante == "sem_travas":
        perfis = tabelas["perfis"].reset_index(drop=True)
        limites = limites_livres(treino_do_simulador(tabelas["refeicoes"], perfis))
        genes = ag_sem_travas(CONFIG, simular, perfil, cardapio, limites, semente)
        d9 = refeicoes_livres(genes[None], perfil, horario_d9(perfil, cardapio))
    elif variante in ("ag_horario_fixo", "horario_livre"):
        config = ConfigAG(horario_livre=variante == "horario_livre")
        plano = ag.rodar(config, simular, perfil, cardapio, semente).melhor
        d9 = d9_do_plano(plano, perfil, cardapio)
    else:
        plano = otimo_exato(perfil, cardapio, simular, CONFIG)
        d9 = d9_do_plano(plano, perfil, cardapio)
    d9 = d9.assign(iauc=simular(d9), variante=variante, semente=semente, grupo=perfil["grupo"])
    return Saida(d9, plano, time.process_time() - inicio)


def matriz_personalizacao(
    planos: dict[int, Populacao],
    perfis: pd.DataFrame,
    cardapio: pd.DataFrame,
    simulador_de: Callable[[pd.Series], Simular],
    lambdas: Lambdas | None = None,
) -> pd.DataFrame:
    """O plano leva os pratos e a caminhada. O horário é o do hábito da coluna, porque o sistema
    fixa os horários no hábito de quem segue o plano, e assim a matriz mede só a escolha dos
    pratos e da caminhada."""
    donos = list(planos)
    todos = Populacao.juntar(*planos.values())
    colunas = {}
    for participante in donos:
        perfil = perfis.loc[participante]
        nos_horarios = todos.copiar()
        nos_horarios.intervalo[:] = habito(perfil, cardapio).intervalo
        simular = simulador_de(perfil)
        if lambdas is None:
            colunas[participante] = ag.iauc_d9(nos_horarios, simular, perfil, cardapio)
        else:
            colunas[participante] = ag.aptidao(nos_horarios, simular, perfil, cardapio, lambdas)
    matriz = pd.DataFrame(colunas, index=donos)
    matriz.index.name, matriz.columns.name = "plano_de", "perfil_de"
    return matriz


def diagonal_minima(matriz: pd.DataFrame) -> pd.Series:
    diagonal = pd.Series(np.diag(matriz), index=matriz.columns)
    return diagonal <= matriz.min(axis=0)


def perda_plano_do_grupo(matriz: pd.DataFrame, grupos: pd.Series) -> pd.DataFrame:
    """O plano do grupo é o que um recomendador sem personalização daria a todos do grupo."""
    linhas = []
    for grupo, membros in grupos.groupby(grupos):
        no_grupo = matriz.loc[membros.index, membros.index]
        plano_do_grupo = no_grupo.mean(axis=1).idxmin()
        for participante in membros.index:
            proprio = no_grupo.loc[participante, participante]
            linhas.append(
                {
                    "participante": participante,
                    "grupo": grupo,
                    "plano_do_grupo": plano_do_grupo,
                    "perda": no_grupo.loc[plano_do_grupo, participante] - proprio,
                }
            )
    return pd.DataFrame(linhas)


def ordem_por_grupo(grupos: pd.Series) -> list[int]:
    posicao = {grupo: i for i, grupo in enumerate(GRUPOS)}
    return sorted(grupos.index, key=lambda p: (posicao[grupos[p]], p))


@cache
def refeicoes_da_grade() -> pd.DataFrame:
    """As refeições de D1 a D9 com a iAUC e a glicose pré-refeição também na grade de 1 minuto.

    refeicoes.py usa as leituras nativas. Aqui a base é o valor da grade no minuto da refeição,
    que a interpolação já puxa na direção da primeira leitura depois de comer."""
    tabelas = dados()
    refeicoes, dias = tabelas["refeicoes"], tabelas["dias"]
    estudo = refeicoes[refeicoes["dia"].isin(simulador.DIAS_ESTUDO)]
    datas = dias.set_index(["participante", "dia"])["data"]
    data = datas.loc[pd.MultiIndex.from_frame(estudo[["participante", "dia"]])].to_numpy()
    minutos = np.round(estudo["horario"].to_numpy() * MINUTOS_POR_HORA)
    instante = pd.DatetimeIndex(data) + pd.to_timedelta(minutos, unit="min")
    estudo = estudo.assign(instante=instante)
    linhas = []
    for participante, do_participante in estudo.groupby("participante"):
        grade = leitura.ler_participante(participante)["dexcom"]
        for linha, inicio in do_participante["instante"].items():
            janela = grade[inicio : inicio + JANELA].dropna()
            base = grade.get(inicio, np.nan)
            if np.isnan(base):
                continue
            minutos_janela = ((janela.index - inicio) / pd.Timedelta(minutes=1)).to_numpy()
            iauc = calcular_iauc(minutos_janela, janela.to_numpy(), base)
            linhas.append((linha, iauc, base))
    grade = pd.DataFrame(linhas, columns=["linha", "iauc_grade", "glicose_pre_grade"])
    return estudo.drop(columns="instante").join(grade.set_index("linha"), how="inner")


def versao_das_refeicoes(versao: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    refeicoes = refeicoes_da_grade().assign(iauc_nativa=lambda t: t["iauc_dexcom"])
    if versao == "grade":
        refeicoes = refeicoes.assign(
            iauc_dexcom=refeicoes["iauc_grade"], glicose_pre=refeicoes["glicose_pre_grade"]
        )
    bio = pd.read_csv(pasta_dados() / "CGMacros" / "bio.csv")
    return refeicoes, montar_perfis(refeicoes, dados()["dias"], bio)


def rodar_fold(versao: str, fold: int) -> Saida:
    inicio = time.process_time()
    refeicoes, perfis = versao_das_refeicoes(versao)
    treino, teste = list(simulador.dividir(refeicoes, perfis))[fold]
    sim, _ = simulador.treinar(treino, simulador.ConfigRede(), SEMENTE)
    previsto = teste.refeicoes.assign(
        versao=versao, fold=fold, previsto=sim.prever_atributos(teste.atributos)
    )
    return Saida(previsto, None, time.process_time() - inicio)


def erro_por_fold(previsoes: pd.DataFrame, alvo: str) -> pd.DataFrame:
    erro = (previsoes[alvo] - previsoes["previsto"]).abs()
    return erro.groupby([previsoes["versao"], previsoes["fold"]]).mean().unstack("versao")


def figura(
    n_paineis: int, largura: float = POLEGADAS_POR_PAINEL, altura: float = POLEGADAS_POR_PAINEL
) -> tuple[Figure, np.ndarray]:
    fig = Figure(figsize=(largura * n_paineis, altura), layout="constrained")
    return fig, np.atleast_1d(fig.subplots(1, n_paineis))


def gravar(fig: Figure, nome: str) -> None:
    PASTA.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA / nome, dpi=DPI)


def distancia_ao_treino(refeicoes: pd.DataFrame, treino: pd.DataFrame) -> np.ndarray:
    escala = treino[list(MACROS)].std().to_numpy()
    pontos = refeicoes[list(MACROS)].to_numpy() / escala
    referencia = treino[list(MACROS)].to_numpy() / escala
    return np.linalg.norm(pontos[:, None, :] - referencia[None, :, :], axis=2).min(axis=1)


def fora_da_faixa(refeicoes: pd.DataFrame, treino: pd.DataFrame) -> pd.Series:
    macros = refeicoes[list(MACROS)]
    acima = macros > treino[list(MACROS)].max()
    abaixo = macros < treino[list(MACROS)].min()
    return (acima | abaixo).any(axis=1)


def analisar_sem_travas(planos_d9: pd.DataFrame, treino: pd.DataFrame) -> str:
    variantes = ["com_travas", "sem_travas"]
    d9 = planos_d9[planos_d9["variante"].isin(variantes)]
    d9 = d9.assign(
        distancia=lambda t: distancia_ao_treino(t, treino), fora=lambda t: fora_da_faixa(t, treino)
    )
    por_plano = d9.groupby(["variante", "participante", "semente"])["iauc"].sum().unstack(0)

    fig, eixos = figura(len(MACROS), largura=3.4)
    rng = np.random.default_rng(SEMENTE)
    for eixo, macro in zip(eixos, MACROS, strict=True):
        eixo.axhspan(treino[macro].min(), treino[macro].max(), color=CINZA, alpha=0.25)
        eixo.boxplot(treino[macro], positions=[0], widths=0.5, showfliers=False)
        for posicao, variante in enumerate(variantes, start=1):
            valores = d9.loc[d9["variante"] == variante, macro]
            espalhado = posicao + rng.uniform(-0.15, 0.15, len(valores))
            eixo.scatter(espalhado, valores, s=14, color=COR[variante], alpha=0.7)
        eixo.set_xticks([0, 1, 2], ["treino", "ótimo com\ntravas", "AG sem\ntravas"])
        eixo.set_title(f"{NOMES_MACRO[macro]} (g por refeição)")
    fig.suptitle("Macros das refeições de D9 contra a faixa do treino (faixa cinza)")
    gravar(fig, "sem_travas_macros.png")

    fig, (antes_depois, distancias) = figura(2)
    for _, linha in por_plano.iterrows():
        antes_depois.plot(range(len(variantes)), linha[variantes], color=CINZA, alpha=0.6)
    for posicao, variante in enumerate(variantes):
        antes_depois.scatter(
            np.full(len(por_plano), posicao), por_plano[variante], color=COR[variante], zorder=3
        )
    antes_depois.set_xticks([0, 1], ["ótimo com travas", "AG sem travas"])
    antes_depois.set_ylabel("iAUC prevista de D9 (mg/dL·min)")
    antes_depois.set_title("O que a rede promete")
    distancias.boxplot(
        [d9.loc[d9["variante"] == v, "distancia"] for v in variantes],
        tick_labels=["ótimo com travas", "AG sem travas"],
    )
    distancias.set_ylabel("distância à refeição de treino mais próxima (desvios-padrão)")
    distancias.set_title("Quanto o plano se afasta dos dados")
    gravar(fig, "sem_travas_vs_com.png")

    livres = d9[d9["variante"] == "sem_travas"]
    com = d9[d9["variante"] == "com_travas"]
    return (
        f"sem travas: iAUC prevista de D9 mediana {por_plano['sem_travas'].median():.0f} contra "
        f"{por_plano['com_travas'].median():.0f} do ótimo com travas (mg/dL·min, {len(por_plano)} "
        f"planos); "
        f"{livres['fora'].mean():.0%} das refeições sem travas têm macro fora da faixa do treino "
        f"(com travas {com['fora'].mean():.0%}); distância mediana ao treino "
        f"{livres['distancia'].median():.2f} contra {com['distancia'].median():.2f} desvios-padrão"
    )


def faixa_de_horario() -> dict[str, tuple[float, float]]:
    minimos, maximos = limites_intervalo()
    extremos = Populacao(
        np.zeros((2, N_REFEICOES), dtype=np.int64),
        np.stack([minimos, maximos]),
        np.zeros((2, N_REFEICOES)),
    )
    cedo, tarde = horarios(extremos)[:, ag.POSICOES_D9]
    return {tipo: (cedo[i], tarde[i]) for i, tipo in enumerate(TIPOS_REFEICAO)}


def efeito_do_horario(participantes: list[int]) -> pd.DataFrame:
    tabelas = dados()
    sim, cardapio = tabelas["simulador"], tabelas["cardapio"]
    faixas = faixa_de_horario()
    linhas = []
    for participante in participantes:
        perfil = tabelas["perfis"].loc[participante]
        d9 = d9_do_plano(habito(perfil, cardapio), perfil, cardapio)
        no_horario_do_habito = sim.prever(perfil, d9)
        for (_, refeicao), base in zip(d9.iterrows(), no_horario_do_habito, strict=True):
            grade = np.arange(*faixas[refeicao["tipo"]], PASSO_HORARIO)
            movida = pd.DataFrame([refeicao] * len(grade)).assign(horario=grade)
            iauc = sim.prever(perfil, conformar(movida, ENTRADA_SIMULADOR))
            linhas.append(
                pd.DataFrame(
                    {
                        "participante": participante,
                        "tipo": refeicao["tipo"],
                        "horario": grade,
                        "diferenca": iauc - base,
                    }
                )
            )
    return pd.concat(linhas, ignore_index=True)


def cai_ou_fica_acima(ganho: float) -> str:
    return f"cai {ganho:.0f}" if ganho >= 0 else f"fica {-ganho:.0f} acima"


def analisar_horario_livre(planos_d9: pd.DataFrame) -> str:
    variantes = ["com_travas", "ag_horario_fixo", "horario_livre"]
    d9 = planos_d9[planos_d9["variante"].isin(variantes)]
    chave = ["participante", "semente", "tipo"]
    horario = d9.pivot_table(index=chave, columns="variante", values="horario")
    deslocamento = (horario["horario_livre"] - horario["ag_horario_fixo"]).rename("horas")
    por_plano = d9.groupby(["variante", "participante", "semente"])["iauc"].sum().unstack(0)
    ganho = por_plano["ag_horario_fixo"] - por_plano["horario_livre"]
    ganho_sobre_otimo = por_plano["com_travas"] - por_plano["horario_livre"]
    efeito = efeito_do_horario(sorted(d9["participante"].unique()))

    fig, (antes_depois, horas, curva) = figura(3)
    for _, linha in por_plano.iterrows():
        antes_depois.plot(range(len(variantes)), linha[variantes], color=CINZA, alpha=0.6)
    for posicao, variante in enumerate(variantes):
        antes_depois.scatter(
            np.full(len(por_plano), posicao), por_plano[variante], color=COR[variante], zorder=3
        )
    antes_depois.set_xticks(
        range(len(variantes)),
        ["ótimo exato,\nhorário do hábito", "AG, horário\ndo hábito", "AG, horário\nlivre"],
    )
    antes_depois.set_ylabel("iAUC prevista de D9 (mg/dL·min)")
    antes_depois.set_title("O ganho prometido")
    por_tipo = [deslocamento.xs(tipo, level="tipo") for tipo in TIPOS_REFEICAO]
    horas.boxplot(por_tipo, tick_labels=[NOMES_TIPO[t] for t in TIPOS_REFEICAO])
    horas.axhline(0, color=CINZA, linewidth=1)
    horas.set_ylabel("horas depois do hábito em D9")
    horas.set_title("Para onde o AG move as refeições")
    for tipo, cor in zip(TIPOS_REFEICAO, ("#257de0", "#f47230", "#00b894"), strict=True):
        media = efeito[efeito["tipo"] == tipo].groupby("horario")["diferenca"].mean()
        curva.plot(media.index, media.to_numpy(), color=cor, label=NOMES_TIPO[tipo])
    curva.axhline(0, color=CINZA, linewidth=1)
    curva.set_xlabel("horário da refeição (h)")
    curva.set_ylabel("iAUC prevista menos a do horário do hábito")
    curva.set_title("Efeito do horário que a rede aprendeu")
    curva.legend()
    gravar(fig, "horario_livre.png")

    return (
        f"horário livre: o AG move as refeições de D9 em mediana "
        f"{deslocamento.abs().median():.1f} h (máximo {deslocamento.abs().max():.1f} h) e a "
        f"iAUC prevista {cai_ou_fica_acima(ganho.median())} mg/dL·min em mediana "
        f"({(ganho / por_plano['ag_horario_fixo']).median():.0%}) frente ao mesmo AG no horário "
        f"do hábito, em {len(por_plano)} planos; frente ao ótimo exato no horário do hábito "
        f"{cai_ou_fica_acima(ganho_sobre_otimo.median())} e vence em "
        f"{ganho_sobre_otimo.gt(0).mean():.0%}"
    )


MEDIDAS_PERSONALIZACAO = {"iauc": None, "aptidao": CONFIG.lambdas}
TITULOS_MEDIDA = {
    "iauc": "iAUC prevista de D9",
    "aptidao": "aptidão do AG (iAUC + penalidades)",
}


def desenhar_mapa(eixo, matriz: pd.DataFrame, grupos: pd.Series, titulo: str):
    acima_do_proprio = 100 * (matriz / np.diag(matriz) - 1)
    limite = np.nanpercentile(np.abs(acima_do_proprio.to_numpy()), 95)
    imagem = eixo.imshow(acima_do_proprio, cmap="RdBu_r", vmin=-limite, vmax=limite)
    fronteiras = np.cumsum(grupos.loc[matriz.index].value_counts().reindex(GRUPOS).fillna(0))
    for fronteira in fronteiras[:-1]:
        eixo.axhline(fronteira - 0.5, color="k", linewidth=1)
        eixo.axvline(fronteira - 0.5, color="k", linewidth=1)
    rotulos = [f"{p:03d}" for p in matriz.index]
    eixo.set_xticks(range(len(rotulos)), rotulos, rotation=90, fontsize=5)
    eixo.set_yticks(range(len(rotulos)), rotulos, fontsize=5)
    eixo.set_xlabel("perfil em que o plano é avaliado")
    eixo.set_ylabel("dono do plano")
    eixo.set_title(titulo)
    return imagem


def analisar_personalizacao(planos: dict[int, Populacao], grupos: pd.Series) -> str:
    tabelas = dados()
    sim = tabelas["simulador"]
    ordem = ordem_por_grupo(grupos.loc[list(planos)])
    matrizes = {
        medida: matriz_personalizacao(
            {p: planos[p] for p in ordem},
            tabelas["perfis"],
            tabelas["cardapio"],
            lambda perfil: lambda refeicoes: sim.prever(perfil, refeicoes),
            lambdas,
        )
        for medida, lambdas in MEDIDAS_PERSONALIZACAO.items()
    }
    PASTA.mkdir(parents=True, exist_ok=True)
    longa = pd.concat(
        {medida: m.stack().rename("valor") for medida, m in matrizes.items()}, names=["medida"]
    )
    longa.reset_index().to_parquet(PASTA / "matriz_personalizacao.parquet", index=False)
    acertos = {medida: diagonal_minima(m) for medida, m in matrizes.items()}
    perdas = {medida: perda_plano_do_grupo(m, grupos.loc[ordem]) for medida, m in matrizes.items()}

    fig, eixos = figura(2, largura=7.5, altura=7.5)
    for eixo, (medida, matriz) in zip(eixos, matrizes.items(), strict=True):
        titulo = (
            f"{TITULOS_MEDIDA[medida]}\ndiagonal é a menor da coluna em "
            f"{acertos[medida].sum()} de {len(ordem)}"
        )
        imagem = desenhar_mapa(eixo, matriz, grupos, titulo)
        fig.colorbar(imagem, ax=eixo, label="% acima do próprio plano", shrink=0.8)
    fig.suptitle("Plano de uma pessoa no perfil de outra (ordem: saudável, pré-diabetes, DM2)")
    gravar(fig, "matriz_personalizacao.png")

    fig, eixos = figura(2)
    rng = np.random.default_rng(SEMENTE)
    for eixo, (medida, perda) in zip(eixos, perdas.items(), strict=True):
        for posicao, grupo in enumerate(GRUPOS):
            valores = perda.loc[perda["grupo"] == grupo, "perda"]
            eixo.boxplot(valores, positions=[posicao], widths=0.5, showfliers=False)
            espalhado = posicao + rng.uniform(-0.15, 0.15, len(valores))
            eixo.scatter(espalhado, valores, color=COR_GRUPO[grupo], zorder=3)
        eixo.axhline(0, color=CINZA, linewidth=1)
        eixo.set_xticks(range(len(GRUPOS)), [NOMES_GRUPO[g] for g in GRUPOS])
        eixo.set_ylabel("a mais com o plano do grupo")
        eixo.set_title(TITULOS_MEDIDA[medida])
        if medida == "aptidao":
            # Sair da faixa de kcal custa dezenas de milhares e esmaga o resto na escala linear.
            eixo.set_yscale("symlog", linthresh=LINEAR_ATE)
    fig.suptitle("Quanto cada pessoa perde seguindo o plano do grupo no lugar do seu")
    gravar(fig, "perda_plano_grupo.png")

    iauc, aptidao = acertos["iauc"], acertos["aptidao"]
    matriz_aptidao = matrizes["aptidao"]
    posicao_do_proprio = (matriz_aptidao < np.diag(matriz_aptidao)).sum(axis=0) + 1
    return (
        f"personalização: a diagonal é a menor da coluna em {iauc.sum()} de {len(iauc)} colunas "
        f"na iAUC ({iauc.mean():.0%}) e em {aptidao.sum()} ({aptidao.mean():.0%}) na aptidão, "
        f"onde o próprio plano fica no pior caso em {posicao_do_proprio.max()}º; "
        f"o plano do grupo muda a iAUC de D9 em mediana "
        f"{perdas['iauc']['perda'].median():+.0f} mg/dL·min e a aptidão em "
        f"{perdas['aptidao']['perda'].median():+.0f}"
    )


def analisar_interpolacao(previsoes: pd.DataFrame) -> str:
    refeicoes = refeicoes_da_grade()
    contra_nativa = erro_por_fold(previsoes, "iauc_nativa")
    contra_propria = erro_por_fold(previsoes, "iauc_dexcom")
    pearson = refeicoes["iauc_grade"].corr(refeicoes["iauc_dexcom"])
    diferenca_pre = refeicoes["glicose_pre_grade"] - refeicoes["glicose_pre"]

    fig, (barras, dispersao) = figura(2)
    rotulos = {"nativa": "alvo nativo", "grade": "alvo da grade"}
    for posicao, versao in enumerate(rotulos):
        barras.bar(
            posicao,
            contra_nativa[versao].mean(),
            yerr=contra_nativa[versao].std(),
            color=COR[versao],
            capsize=6,
        )
    barras.set_xticks([0, 1], [f"treinado com\n{r}" for r in rotulos.values()])
    barras.set_ylabel("MAE contra a iAUC nativa (mg/dL·min)")
    barras.set_title("Erro do simulador (barra: desvio entre os 5 folds)")
    dispersao.scatter(refeicoes["iauc_dexcom"], refeicoes["iauc_grade"], s=6, alpha=0.4)
    limite = max(refeicoes["iauc_dexcom"].max(), refeicoes["iauc_grade"].max())
    dispersao.plot([0, limite], [0, limite], color=CINZA, linewidth=1)
    dispersao.set_xlabel("iAUC das leituras nativas (mg/dL·min)")
    dispersao.set_ylabel("iAUC da grade de 1 minuto (mg/dL·min)")
    dispersao.set_title(f"Mesma refeição, Pearson {pearson:.4f}")
    gravar(fig, "interpolacao_mae.png")

    fig, (histograma,) = figura(1, largura=6)
    histograma.hist(diferenca_pre, bins=60, color=COR["grade"])
    mediana = diferenca_pre.median()
    p90 = np.percentile(diferenca_pre, PERCENTIL_GLICOSE_PRE)
    histograma.axvline(mediana, color="k", label=f"mediana {mediana:.1f}")
    histograma.axvline(p90, color="k", linestyle="--", label=f"percentil 90 {p90:.1f}")
    histograma.set_xlabel("glicose pré-refeição da grade menos a nativa (mg/dL)")
    histograma.set_ylabel("refeições")
    histograma.set_title("A grade já vê o começo da subida no minuto da refeição")
    histograma.legend()
    gravar(fig, "interpolacao_glicose_pre.png")

    return (
        f"interpolação: MAE contra a iAUC nativa {contra_nativa['nativa'].mean():.0f} ± "
        f"{contra_nativa['nativa'].std():.0f} treinando com o alvo nativo e "
        f"{contra_nativa['grade'].mean():.0f} ± {contra_nativa['grade'].std():.0f} com o da grade "
        f"(contra o próprio alvo {contra_propria['grade'].mean():.0f}); Pearson das iAUC "
        f"{pearson:.4f} em {len(refeicoes)} refeições; glicose pré da grade menos a nativa "
        f"mediana {mediana:.1f} e percentil 90 {p90:.1f} mg/dL"
    )


def tarefas(grupos: pd.Series, semente: int) -> list[tuple]:
    # Os folds são as tarefas mais longas. Vão primeiro para não sobrarem sozinhos no fim.
    lista = [("fold", versao, fold) for fold in range(simulador.N_FOLDS) for versao in VERSOES]
    return lista + [("ag", variante, p, semente) for variante in VARIANTES for p in grupos.index]


def executar(tarefa: tuple):
    tipo, *argumentos = tarefa
    return {"fold": rodar_fold, "ag": rodar_variante}[tipo](*argumentos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mostra os limites do sistema e a personalização.")
    parser.add_argument("--processos", type=int, default=PROCESSOS)
    parser.add_argument("--semente", type=int, default=SEMENTE)
    args = parser.parse_args()

    inicio, carga_inicial = time.perf_counter(), os.getloadavg()
    tabelas = dados()
    grupos = elegiveis(tabelas["dias"], tabelas["perfis"].reset_index(drop=True))
    lista = tarefas(grupos, args.semente)

    # Com python -m este arquivo vira o __main__, e o joblib não consegue mandar aos filhos a
    # refeicoes_da_grade com @cache definida no __main__. Importada pelo nome do pacote, executar
    # chega aos filhos como referência.
    from glicose.diferenciais import executar as executar_no_filho

    saidas = joblib.Parallel(n_jobs=args.processos)(
        joblib.delayed(executar_no_filho)(tarefa) for tarefa in lista
    )
    cpu_tarefas = sum(saida.segundos_cpu for saida in saidas)
    por_tipo = {"fold": [], "ag": []}
    for tarefa, saida in zip(lista, saidas, strict=True):
        por_tipo[tarefa[0]].append(saida.tabela)
    planos = {
        tarefa[2]: saida.plano
        for tarefa, saida in zip(lista, saidas, strict=True)
        if tarefa[:2] == ("ag", "com_travas")
    }
    planos_d9, folds = pd.concat(por_tipo["ag"]), pd.concat(por_tipo["fold"])

    treino = treino_do_simulador(tabelas["refeicoes"], tabelas["perfis"].reset_index(drop=True))
    resumos = [
        analisar_sem_travas(planos_d9, treino),
        analisar_horario_livre(planos_d9),
        analisar_personalizacao(planos, grupos),
        analisar_interpolacao(folds),
    ]
    for resumo in resumos:
        print(resumo)
    segundos = time.perf_counter() - inicio
    cpu_total = cpu_tarefas + time.process_time()
    print(
        f"tempo: {segundos:.0f} s de parede, {cpu_total:.0f} s de CPU em {len(lista)} tarefas e "
        f"{args.processos} processos; carga do sistema {carga_inicial[0]:.1f} no início e "
        f"{os.getloadavg()[0]:.1f} no fim"
    )
    print(f"figuras em {PASTA}")


if __name__ == "__main__":
    main()
