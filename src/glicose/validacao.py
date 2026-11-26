"""Validação metodológica do simulador: repetição, intervalos de confiança e testes pareados.

Compara a rede com os baselines em dois cenários, sempre sobre D7 a D9:

- ii, pessoa nova: GroupKFold por participante, repetido com partições diferentes.
- i, pessoa conhecida: treino em D1 a D6 de todos e teste em D7 a D9 de todos.

A unidade estatística é o participante. O intervalo de confiança vem de bootstrap sobre
participantes, e os testes são Wilcoxon pareados com correção de Holm dentro de cada métrica.
O MAE é a métrica principal e o RMSE é uma segunda família de sensibilidade. Viés, erro da soma
de D9 e correlação de Pearson são só descritivos. A calibração usa o desvio entre as redes do
conjunto como incerteza. Uma ablação treina também com MSE e Huber no cenário ii.

Autores: Miguel Mochizuki Silva
"""

import argparse
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import keras
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import GroupKFold

from glicose import simulador
from glicose.baselines_modelo import ALVO, BASELINES, Particao
from glicose.esquema import DIAS_HABITO, DIAS_PLANO, TIPOS_REFEICAO
from glicose.refeicoes import ARTEFATOS
from glicose.simulador import ConfigRede

PASTA = ARTEFATOS / "validacao"
REPETICOES = 5
BOOTSTRAP = 10_000
SEMENTE = 0
CONFIG = ConfigRede()
ALFA = 0.05
PERDAS = {"mae": "mae", "mse": "mse", "huber": keras.losses.Huber()}
PERDA_PRINCIPAL = "mae"
ABLACOES = tuple(perda for perda in PERDAS if perda != PERDA_PRINCIPAL)
METRICAS = ("mae", "rmse")
COBERTURAS = (1, 2)
DECIS = 10
DIA_DA_SOMA = DIAS_PLANO[-1]
DPI = 120


@contextmanager
def com_perda(nome: str) -> Iterator[None]:
    """Troca a perda de todo modelo que `simulador` construir dentro do bloco.

    Args:
        nome: Chave de `PERDAS`.
    """
    original = simulador.construir_modelo

    def construir(config: ConfigRede) -> keras.Model:
        modelo = original(config)
        modelo.compile(loss=PERDAS[nome], optimizer=keras.optimizers.SGD(config.taxa_aprendizado))
        return modelo

    with patch.object(simulador, "construir_modelo", construir):
        yield


def dobras(estudo: Particao, semente: int) -> Iterator[tuple[Particao, Particao]]:
    """Cenário ii: dobras por participante, embaralhadas pela semente.

    Yields:
        Pares `(treino, teste)`. O treino tem só D1 a D6 de quem não está no teste, e o teste
        tem todos os dias de quem ficou fora.
    """
    habito = estudo.refeicoes["dia"].isin(DIAS_HABITO).to_numpy()
    divisor = GroupKFold(simulador.N_FOLDS, shuffle=True, random_state=semente)
    for treino, teste in divisor.split(estudo.atributos, groups=estudo.refeicoes["participante"]):
        yield estudo.recortar(treino[habito[treino]]), estudo.recortar(teste)


def particao_conhecida(estudo: Particao) -> tuple[Particao, Particao]:
    """Cenário i: treino em D1 a D6 de todos. O teste devolve todos os dias.

    Os baselines precisam dos dias de hábito do teste, e a avaliação filtra D7 a D9 depois.
    """
    habito = estudo.refeicoes["dia"].isin(DIAS_HABITO).to_numpy()
    return estudo.recortar(habito), estudo


def prever_com_desvio(
    sim: simulador.Simulador, atributos: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Previsão do conjunto e desvio entre as redes, na escala da iAUC.

    Returns:
        Tupla `(previsao, desvio)`, uma entrada por linha.
    """
    entradas = sim.entradas(atributos)
    redes = [camada for camada in sim.modelo.layers if isinstance(camada, keras.Model)]
    por_rede = np.stack([sim.para_iauc(rede.predict_on_batch(entradas)) for rede in redes])
    return sim.prever_atributos(atributos), por_rede.std(axis=0, ddof=1)


def tabela_de_previsoes(teste: Particao, previsto: dict, desvio: np.ndarray) -> pd.DataFrame:
    """Previsões de D7 a D9, com uma coluna por modelo e o desvio entre as redes."""
    tabela = teste.refeicoes.assign(linha=teste.refeicoes.index, desvio=desvio, **previsto)
    return tabela[tabela["dia"].isin(DIAS_PLANO)].reset_index(drop=True)


def prever_desconhecido(
    estudo: Particao, config: ConfigRede, semente: int
) -> tuple[pd.DataFrame, list[int]]:
    """Uma repetição do cenário ii.

    Returns:
        As previsões de D7 a D9 e a melhor época de cada rede.
    """
    partes, melhores_epocas = [], []
    for fold, (treino, teste) in enumerate(dobras(estudo, semente)):
        sim, epocas = simulador.treinar(treino, config, semente)
        melhores_epocas.extend(epocas)
        previsto = {nome: baseline(treino, teste) for nome, baseline in BASELINES.items()}
        previsto["rede"], desvio = prever_com_desvio(sim, teste.atributos)
        partes.append(tabela_de_previsoes(teste, previsto, desvio).assign(fold=fold))
    return pd.concat(partes, ignore_index=True), melhores_epocas


def prever_conhecido(
    refeicoes: pd.DataFrame,
    perfis: pd.DataFrame,
    estudo: Particao,
    config: ConfigRede,
    semente: int,
) -> pd.DataFrame:
    """Uma repetição do cenário i, com o número de épocas já fixado em `config`."""
    treino, teste = particao_conhecida(estudo)
    sim = simulador.treinar_final(refeicoes, perfis, config, semente)
    previsto = {nome: baseline(treino, teste) for nome, baseline in BASELINES.items()}
    previsto["rede"], desvio = prever_com_desvio(sim, teste.atributos)
    return tabela_de_previsoes(teste, previsto, desvio)


def rodar_protocolo(
    refeicoes: pd.DataFrame,
    perfis: pd.DataFrame,
    config: ConfigRede = CONFIG,
    repeticoes: int = REPETICOES,
    semente: int = SEMENTE,
    perdas: tuple[str, ...] = tuple(PERDAS),
) -> pd.DataFrame:
    """Treina e prevê todos os cenários, perdas e repetições.

    O cenário i só roda com a perda principal, com as épocas iguais à mediana das paradas do
    cenário ii, como `simulador.main` faz.

    Returns:
        Previsões de D7 a D9 com as colunas `cenario`, `perda` e `repeticao`.
    """
    estudo = simulador.particao_do_estudo(refeicoes, perfis)
    partes = []
    for perda in perdas:
        epocas = []
        with com_perda(perda):
            for repeticao in range(repeticoes):
                inicio = time.perf_counter()
                previsoes, parada = prever_desconhecido(estudo, config, semente + repeticao)
                epocas.extend(parada)
                partes.append(previsoes.assign(cenario="ii", perda=perda, repeticao=repeticao))
                print(f"perda {perda}, repetição {repeticao}: {time.perf_counter() - inicio:.0f} s")
            if perda == PERDA_PRINCIPAL:
                final = replace(config, epocas=int(np.median(epocas)))
                for repeticao in range(repeticoes):
                    conhecida = prever_conhecido(
                        refeicoes, perfis, estudo, final, semente + repeticao
                    )
                    partes.append(conhecida.assign(cenario="i", perda=perda, repeticao=repeticao))
    return pd.concat(partes, ignore_index=True)


def holm(p: np.ndarray) -> np.ndarray:
    """Valores p ajustados pelo método de Holm-Bonferroni."""
    p = np.asarray(p, dtype=float)
    ordem = np.argsort(p)
    ajustado = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[ordem])
    resultado = np.empty_like(p)
    resultado[ordem] = np.minimum(ajustado, 1.0)
    return resultado


def p_wilcoxon(diferencas: np.ndarray) -> float:
    """Valor p do Wilcoxon pareado, 1 se todas as diferenças são nulas."""
    if not np.any(diferencas):
        return 1.0
    return float(wilcoxon(diferencas).pvalue)


def ic_mediana(
    valores: np.ndarray, bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mediana e intervalo de 95% por bootstrap sobre participantes.

    Args:
        valores: Um valor por participante.
        bootstrap: Número de reamostragens.
        rng: Gerador aleatório.

    Returns:
        Tupla `(mediana, limite_inferior, limite_superior)`.
    """
    sorteio = rng.integers(0, len(valores), (bootstrap, len(valores)))
    medianas = np.median(valores[sorteio], axis=1)
    inferior, superior = np.quantile(medianas, [ALFA / 2, 1 - ALFA / 2])
    return float(np.median(valores)), float(inferior), float(superior)


def metricas_por_participante(previsoes: pd.DataFrame, modelo: str) -> pd.DataFrame:
    """MAE, RMSE, viés e erro da soma de D9 de cada participante.

    Calcula em cada repetição e tira a média entre repetições, para a partição não pesar.

    Returns:
        Tabela indexada por participante.
    """
    chave = [previsoes["repeticao"], previsoes["participante"]]
    erro = previsoes[modelo] - previsoes[ALVO]
    d9 = previsoes[previsoes["dia"] == DIA_DA_SOMA]
    refeicoes_do_d9 = d9.groupby(["repeticao", "participante"])[ALVO].transform("size")
    completo = d9[refeicoes_do_d9 == len(TIPOS_REFEICAO)]
    soma_d9 = (
        erro[completo.index].groupby([completo["repeticao"], completo["participante"]]).sum().abs()
    )
    por_repeticao = pd.DataFrame(
        {
            "mae": erro.abs().groupby(chave).mean(),
            "rmse": (erro**2).groupby(chave).mean() ** 0.5,
            "vies": erro.groupby(chave).mean(),
            "erro_soma_d9": soma_d9,
        }
    )
    return por_repeticao.groupby("participante").mean()


def contrastes_da_familia(
    por_participante: dict, metrica: str, bootstrap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Contrastes pareados de uma métrica, com Holm sobre a família inteira.

    A diferença é `A - B`, então valor negativo favorece A. São 9 contrastes: a rede contra cada
    baseline nos cenários ii e i, o ganho de conhecer a pessoa e as duas perdas da ablação.

    Args:
        por_participante: Dicionário `(cenario, perda, modelo) -> metricas_por_participante`.
        metrica: Uma das `METRICAS`.
        bootstrap: Número de reamostragens do intervalo.
        rng: Gerador aleatório.
    """
    rede_ii = ("ii", PERDA_PRINCIPAL, "rede")
    pares = [
        (
            f"rede vs {base} ({cenario})",
            (cenario, PERDA_PRINCIPAL, "rede"),
            (cenario, PERDA_PRINCIPAL, base),
        )
        for cenario in ("ii", "i")
        for base in BASELINES
    ]
    pares.append(("rede, pessoa conhecida vs nova", ("i", PERDA_PRINCIPAL, "rede"), rede_ii))
    pares += [
        (f"rede {perda} vs {PERDA_PRINCIPAL} (ii)", ("ii", perda, "rede"), rede_ii)
        for perda in ABLACOES
        if ("ii", perda, "rede") in por_participante
    ]
    linhas = []
    for nome, a, b in pares:
        if a not in por_participante or b not in por_participante:
            continue
        juntos = pd.concat([por_participante[a][metrica], por_participante[b][metrica]], axis=1)
        diferenca = (juntos.iloc[:, 0] - juntos.iloc[:, 1]).dropna().to_numpy()
        mediana, inferior, superior = ic_mediana(diferenca, bootstrap, rng)
        linhas.append(
            {
                "metrica": metrica,
                "contraste": nome,
                "n": len(diferenca),
                "mediana": mediana,
                "ic_inferior": inferior,
                "ic_superior": superior,
                "p": p_wilcoxon(diferenca),
            }
        )
    tabela = pd.DataFrame(linhas)
    return tabela.assign(p_holm=holm(tabela["p"].to_numpy()))


def media_por_linha(previsoes: pd.DataFrame, modelo: str) -> pd.DataFrame:
    """Média, entre repetições, da previsão e do desvio de cada refeição."""
    base = previsoes.groupby("linha").agg(
        participante=("participante", "first"),
        alvo=(ALVO, "first"),
        previsao=(modelo, "mean"),
        desvio=("desvio", "mean"),
    )
    return base


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Correlação de Pearson, NaN se alguma série é constante."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(x, y)[0, 1])


def pearson_com_ic(
    linhas: pd.DataFrame, bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Pearson das refeições, com intervalo por bootstrap sobre participantes."""
    participantes = linhas["participante"].to_numpy()
    nomes = np.unique(participantes)
    donos = [np.flatnonzero(participantes == nome) for nome in nomes]
    x, y = linhas["previsao"].to_numpy(), linhas["alvo"].to_numpy()
    sorteio = rng.integers(0, len(nomes), (bootstrap, len(nomes)))
    valores = np.array(
        [
            pearson(
                x[np.concatenate([donos[i] for i in s])], y[np.concatenate([donos[i] for i in s])]
            )
            for s in sorteio
        ]
    )
    inferior, superior = np.nanquantile(valores, [ALFA / 2, 1 - ALFA / 2])
    return pearson(x, y), float(inferior), float(superior)


def descritivas(
    previsoes: pd.DataFrame, por_participante: dict, bootstrap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Mediana e intervalo de cada métrica, mais o Pearson, por cenário, perda e modelo."""
    linhas = []
    for (cenario, perda, modelo), metricas in por_participante.items():
        linha = {"cenario": cenario, "perda": perda, "modelo": modelo}
        for coluna in (*METRICAS, "vies", "erro_soma_d9"):
            valores = metricas[coluna].dropna().to_numpy()
            mediana, inferior, superior = ic_mediana(valores, bootstrap, rng)
            linha |= {coluna: mediana, f"{coluna}_inf": inferior, f"{coluna}_sup": superior}
        grupo = previsoes[(previsoes["cenario"] == cenario) & (previsoes["perda"] == perda)]
        linha["pearson"], linha["pearson_inf"], linha["pearson_sup"] = pearson_com_ic(
            media_por_linha(grupo, modelo), bootstrap, rng
        )
        linhas.append(linha)
    return pd.DataFrame(linhas)


def calibracao(
    previsoes: pd.DataFrame, bootstrap: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """O desvio entre as redes acompanha o erro? Spearman, cobertura e erro por decil.

    Returns:
        Resumo por cenário (Spearman com intervalo e cobertura de ±k desvios) e o erro médio
        em cada decil do desvio.
    """
    resumo, decis = [], []
    principal = previsoes[previsoes["perda"] == PERDA_PRINCIPAL]
    for cenario, grupo in principal.groupby("cenario"):
        linhas = media_por_linha(grupo, "rede")
        erro = (linhas["alvo"] - linhas["previsao"]).abs()
        participantes = linhas["participante"].to_numpy()
        nomes = np.unique(participantes)
        donos = [np.flatnonzero(participantes == nome) for nome in nomes]
        rho = spearmanr(linhas["desvio"], erro).statistic
        sorteio = rng.integers(0, len(nomes), (min(bootstrap, 2000), len(nomes)))
        replicas = [
            spearmanr(linhas["desvio"].to_numpy()[i], erro.to_numpy()[i]).statistic
            for i in (np.concatenate([donos[j] for j in s]) for s in sorteio)
        ]
        inferior, superior = np.nanquantile(replicas, [ALFA / 2, 1 - ALFA / 2])
        linha = {
            "cenario": cenario,
            "spearman": float(rho),
            "spearman_inf": float(inferior),
            "spearman_sup": float(superior),
        }
        for k in COBERTURAS:
            linha[f"cobertura_{k}"] = float((erro <= k * linhas["desvio"]).mean())
        resumo.append(linha)
        faixa = pd.qcut(linhas["desvio"], DECIS, labels=False, duplicates="drop")
        decis.append(
            pd.DataFrame({"desvio": linhas["desvio"], "erro": erro})
            .groupby(faixa)
            .mean()
            .assign(cenario=cenario)
            .reset_index(names="decil")
        )
    return pd.DataFrame(resumo), pd.concat(decis, ignore_index=True)


def analisar(previsoes: pd.DataFrame, bootstrap: int = BOOTSTRAP, semente: int = SEMENTE) -> dict:
    """Todas as tabelas do relatório a partir das previsões.

    Returns:
        Dicionário com `contrastes`, `descritivas`, `calibracao` e `decis`.
    """
    rng = np.random.default_rng(semente)
    por_participante = {}
    for (cenario, perda), grupo in previsoes.groupby(["cenario", "perda"]):
        for modelo in ("rede", *BASELINES):
            if modelo != "rede" and perda != PERDA_PRINCIPAL:
                continue
            por_participante[(cenario, perda, modelo)] = metricas_por_participante(grupo, modelo)
    contrastes = pd.concat(
        [contrastes_da_familia(por_participante, m, bootstrap, rng) for m in METRICAS],
        ignore_index=True,
    )
    resumo, decis = calibracao(previsoes, bootstrap, rng)
    return {
        "contrastes": contrastes,
        "descritivas": descritivas(previsoes, por_participante, bootstrap, rng),
        "calibracao": resumo,
        "decis": decis,
    }


def desenhar_contrastes(contrastes: pd.DataFrame, caminho: Path) -> None:
    """Gráfico de floresta: mediana da diferença e intervalo de cada contraste, por métrica."""
    figura = Figure(figsize=(11, 5), layout="constrained")
    eixos = figura.subplots(1, len(METRICAS), sharey=True)
    for eixo, metrica in zip(eixos, METRICAS, strict=True):
        dados = contrastes[contrastes["metrica"] == metrica].reset_index(drop=True)
        posicao = np.arange(len(dados))[::-1]
        for y, linha in zip(posicao, dados.itertuples(), strict=True):
            cor = "tab:red" if linha.p_holm < ALFA else "0.4"
            eixo.errorbar(
                linha.mediana,
                y,
                xerr=[[linha.mediana - linha.ic_inferior], [linha.ic_superior - linha.mediana]],
                fmt="o",
                color=cor,
            )
        eixo.axvline(0, color="k", linewidth=1)
        eixo.set_yticks(posicao, dados["contraste"])
        eixo.set_xlabel(f"diferença mediana do {metrica.upper()} por participante (mg/dL·min)")
        eixo.set_title(f"{metrica.upper()} (vermelho: Holm < {ALFA})")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=DPI, bbox_inches="tight")


def desenhar_calibracao(decis: pd.DataFrame, caminho: Path) -> None:
    """Erro médio contra o desvio médio entre as redes, em decis."""
    figura = Figure(figsize=(6, 5), layout="constrained")
    eixo = figura.subplots()
    for cenario, grupo in decis.groupby("cenario"):
        eixo.plot(grupo["desvio"], grupo["erro"], "o-", label=f"cenário {cenario}")
    eixo.set_xlabel("desvio entre as redes (mg/dL·min)")
    eixo.set_ylabel("erro absoluto médio (mg/dL·min)")
    eixo.set_title("Calibração da incerteza do conjunto, em decis")
    eixo.legend()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=DPI, bbox_inches="tight")


def gravar_previsoes(previsoes: pd.DataFrame, perdas: tuple[str, ...], pasta: Path = PASTA) -> None:
    """Grava as previsões de um conjunto de perdas em `previsoes_<perdas>.parquet`."""
    pasta.mkdir(parents=True, exist_ok=True)
    previsoes.to_parquet(pasta / f"previsoes_{'_'.join(perdas)}.parquet", index=False)


def ler_previsoes(pasta: Path = PASTA) -> pd.DataFrame:
    """Junta todos os arquivos `previsoes_*.parquet` de `pasta`."""
    arquivos = sorted(pasta.glob("previsoes_*.parquet"))
    return pd.concat([pd.read_parquet(arquivo) for arquivo in arquivos], ignore_index=True)


def gravar_resultados(tabelas: dict, pasta: Path = PASTA) -> None:
    """Grava as tabelas e as duas figuras em `pasta`."""
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, tabela in tabelas.items():
        tabela.to_parquet(pasta / f"{nome}.parquet", index=False)
    desenhar_contrastes(tabelas["contrastes"], pasta / "contrastes.png")
    desenhar_calibracao(tabelas["decis"], pasta / "calibracao.png")


def imprimir(tabelas: dict, bootstrap: int) -> None:
    """Imprime o relatório no terminal."""
    print(f"\nbootstrap de {bootstrap} sobre participantes")
    print("cenário ii: pessoa nova. cenário i: pessoa conhecida. Erros em mg/dL·min.")
    print("\nContrastes (A - B por participante, negativo favorece A)")
    print(tabelas["contrastes"].to_string(index=False, float_format="%.3g"))
    print("\nDescritivas por participante (mediana e IC de 95%)")
    print(tabelas["descritivas"].to_string(index=False, float_format="%.3g"))
    print("\nCalibração do desvio entre as redes")
    print(tabelas["calibracao"].to_string(index=False, float_format="%.3g"))
    print(f"\ngravado em {PASTA}")


def main() -> None:
    """Treina, analisa ou faz as duas coisas, conforme `--etapa`.

    As perdas são independentes, então dá para treinar cada uma em um processo (`--etapa
    treinar --perdas mse`) e depois juntar tudo com `--etapa analisar`.
    """
    parser = argparse.ArgumentParser(description="Valida o simulador com IC e testes pareados.")
    parser.add_argument("--repeticoes", type=int, default=REPETICOES)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--semente", type=int, default=SEMENTE)
    parser.add_argument("--perdas", nargs="+", choices=list(PERDAS), default=list(PERDAS))
    parser.add_argument("--etapa", choices=("treinar", "analisar", "tudo"), default="tudo")
    args = parser.parse_args()

    if args.etapa != "analisar":
        refeicoes = pd.read_parquet(ARTEFATOS / "refeicoes.parquet")
        perfis = pd.read_parquet(ARTEFATOS / "perfis.parquet")
        perdas = tuple(args.perdas)
        previsoes = rodar_protocolo(
            refeicoes, perfis, CONFIG, args.repeticoes, args.semente, perdas
        )
        gravar_previsoes(previsoes, perdas)
    if args.etapa != "treinar":
        tabelas = analisar(ler_previsoes(), args.bootstrap, args.semente)
        gravar_resultados(tabelas)
        imprimir(tabelas, args.bootstrap)


if __name__ == "__main__":
    main()
