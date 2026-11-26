"""Compara o algoritmo genético com o RandomizedSearchCV na busca de hiperparâmetros.

Os dois usam o mesmo orçamento de treinos da rede. A busca escolhe pelo MAE nas
dobras de ajuste, e o resultado reportado vem de participantes que ela nunca viu.

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

import argparse
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import reciprocal, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    ParameterSampler,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from glicose.esquema import DIAS_HABITO, DIAS_PLANO
from glicose.refeicoes import ARTEFATOS
from glicose.simulador import (
    FRACAO_VALIDACAO,
    IAUC_MINIMA,
    SEMENTE,
    ConfigRede,
    construir_modelo,
    particao_do_estudo,
)

Configuracao = dict[str, int | float]

N_FOLDS_BUSCA = 3
FRACAO_TESTE_FINAL = 1 / 3
PARAMETRO_REDE = "regressor__sklearnregressor__model_kwargs"
PREFIXO_FIT = "sklearnregressor__"
BUSCAS = {"genetica": "AG", "aleatoria": "busca aleatória (RandomizedSearchCV)"}
CORES_BUSCA = ("tab:blue", "tab:orange")
N_SEMENTES_PADRAO = 1
ARQUIVO_TREINOS = "hiper_treinos.parquet"
ARQUIVO_POPULACOES = "hiper_populacoes.parquet"
ARQUIVO_FIGURA = "hiper_comparacao.png"
ARQUIVO_TESTE_FINAL = "hiper_teste_final.parquet"


@dataclass(frozen=True)
class Gene:
    """Um hiperparâmetro buscado, com seu intervalo e se é inteiro ou em escala log."""
    minimo: float
    maximo: float
    inteiro: bool = False
    log: bool = False

    def distribuicao(self):
        """Distribuição de onde sortear o gene, no formato do sklearn."""
        if self.inteiro:
            return np.arange(self.minimo, self.maximo + 1)
        if self.log:
            return reciprocal(self.minimo, self.maximo)
        return uniform(self.minimo, self.maximo - self.minimo)

    def normalizar(self, valor: float) -> int | float:
        """Arredonda se o gene é inteiro, senão converte para float."""
        return int(round(valor)) if self.inteiro else float(valor)

    def mutar(self, valor: float, sigma_relativo: float, rng: np.random.Generator) -> int | float:
        """Soma ruído gaussiano ao valor (na escala log, se for o caso) e limita ao intervalo.

        Args:
            valor: Valor atual.
            sigma_relativo: Desvio do ruído, como fração da amplitude.
            rng: Gerador aleatório.
        """
        ida, volta = (np.log10, lambda x: 10.0**x) if self.log else (float, float)
        inicio, fim = ida(self.minimo), ida(self.maximo)
        passo = rng.normal(0, sigma_relativo * (fim - inicio))
        return self.normalizar(np.clip(volta(ida(valor) + passo), self.minimo, self.maximo))


GENES = {
    "n_camadas": Gene(1, 3, inteiro=True),
    "n_neuronios": Gene(8, 128, inteiro=True),
    "taxa_aprendizado": Gene(3e-4, 3e-2, log=True),
    "dropout": Gene(0.0, 0.5),
}


@dataclass(frozen=True)
class ConfigBusca:
    """Hiperparâmetros do AG que busca os hiperparâmetros da rede."""
    tamanho_populacao: int = 10
    n_geracoes: int = 8
    tamanho_torneio: int = 3
    elitismo: int = 1
    prob_gene_do_pai: float = 0.5
    taxa_mutacao: float = 0.25
    sigma_relativo: float = 0.2
    processos: int = 4
    treino: ConfigRede = ConfigRede()


@dataclass(frozen=True)
class Dados:
    """Dados da busca.

    Attributes:
        atributos: Matriz de atributos de D1 a D9.
        iauc: iAUC medida.
        participantes: Participante de cada linha.
        divisoes: Dobras de ajuste, treino em D1 a D6 e teste em D7 a D9.
        teste_final: Treino em D1 a D6 de todo o ajuste e teste em D7 a D9 dos reservados.
    """
    atributos: pd.DataFrame
    iauc: np.ndarray
    participantes: np.ndarray
    divisoes: list[tuple[np.ndarray, np.ndarray]]
    teste_final: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Resultado:
    """Saída de uma execução das duas buscas.

    Attributes:
        treinos: Um treino por linha, com o MAE e as épocas.
        populacoes: População do AG em cada geração.
        segundos: Tempo de cada busca.
    """
    treinos: pd.DataFrame
    populacoes: pd.DataFrame
    segundos: dict[str, float]


def sortear(n: int, semente: int | np.random.RandomState) -> list[Configuracao]:
    """Sorteia `n` configurações com o mesmo sorteio da busca aleatória."""
    distribuicoes = {nome: gene.distribuicao() for nome, gene in GENES.items()}
    sorteadas = ParameterSampler(distribuicoes, n, random_state=semente)
    return [{nome: GENES[nome].normalizar(c[nome]) for nome in GENES} for c in sorteadas]


class SorteioConfiguracao:
    """Distribuição que sorteia uma configuração inteira, para o sklearn."""
    def rvs(self, random_state=None) -> Configuracao:
        """Sorteia uma configuração."""
        return sortear(1, random_state)[0]


def corte_keras(n_linhas: int) -> int:
    """Primeira linha que o `validation_split` do Keras usa para validar."""
    return math.floor(n_linhas * (1.0 - FRACAO_VALIDACAO))


def agrupar_e_cortar_para_validacao(
    linhas: np.ndarray, participantes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Ordena as linhas por participante e corta para a validação do Keras cair em
    participantes inteiros.

    Returns:
        As linhas mantidas, na ordem em que o Keras as lê.
    """
    # O validation_split do Keras valida nas linhas de corte_keras(n) em diante, sem sortear.
    sorteio = rng.permutation(np.unique(participantes[linhas]))
    posicao = pd.Series(np.arange(len(sorteio)), index=sorteio)
    linhas = linhas[np.argsort(posicao[participantes[linhas]].to_numpy(), kind="stable")]
    grupo = participantes[linhas]
    fronteiras = set(np.flatnonzero(grupo[1:] != grupo[:-1]) + 1)
    total = next(n for n in range(len(linhas), 0, -1) if corte_keras(n) in fronteiras)
    return linhas[:total]


def preparar_dados(refeicoes: pd.DataFrame, perfis: pd.DataFrame, semente: int) -> Dados:
    """Monta as dobras de ajuste e o teste final, separados por participante.

    O teste final reserva um terço dos participantes, estratificados por grupo.
    """
    estudo = particao_do_estudo(refeicoes, perfis)
    participantes = estudo.refeicoes["participante"].to_numpy()
    habito = estudo.refeicoes["dia"].isin(DIAS_HABITO).to_numpy()
    plano = estudo.refeicoes["dia"].isin(DIAS_PLANO).to_numpy()
    rng = np.random.default_rng(semente)

    def divisao(treino: np.ndarray, teste: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        treino = agrupar_e_cortar_para_validacao(treino[habito[treino]], participantes, rng)
        return treino, teste[plano[teste]]

    # A busca escolhe pelo MAE das dobras de ajuste. Medir a escolhida nessas mesmas dobras
    # premiaria a sorte da escolha, então o MAE que se reporta vem de participantes que a busca
    # nunca viu.
    grupos = estudo.refeicoes.groupby("participante")["grupo"].first()
    _, finais = train_test_split(
        grupos.index, test_size=FRACAO_TESTE_FINAL, stratify=grupos, random_state=semente
    )
    final = np.isin(participantes, finais)
    ajuste = np.flatnonzero(~final)
    divisoes = [
        divisao(ajuste[treino], ajuste[teste])
        for treino, teste in GroupKFold(N_FOLDS_BUSCA).split(ajuste, groups=participantes[ajuste])
    ]
    teste_final = divisao(ajuste, np.flatnonzero(final))
    return Dados(estudo.atributos, estudo.iauc, participantes, divisoes, teste_final)


def construir_com_semente_fixa(X, y, **genes) -> keras.Model:
    """Constrói a rede com a semente fixa, para o sklearn clonar."""
    keras.utils.set_random_seed(SEMENTE)
    # O predict do SKLearnRegressor não repassa verbose ao Keras.
    keras.config.disable_interactive_logging()
    return construir_modelo(ConfigRede(**genes))


class EscalaDaIauc(StandardScaler):
    """Padronizador do alvo que limita a volta à faixa de iAUC do simulador."""
    # Mesmo piso e teto de Simulador.para_iauc, para a busca medir o erro do simulador de P3.
    def fit(self, X, y=None, sample_weight=None):
        self.iauc_maxima_ = float(np.max(X))
        return super().fit(X, y, sample_weight)

    def inverse_transform(self, X, copy=None):
        return np.clip(super().inverse_transform(X, copy=copy), IAUC_MINIMA, self.iauc_maxima_)


def estimador() -> TransformedTargetRegressor:
    """Pipeline do sklearn com imputação, padronização, rede e alvo padronizado."""
    rede = keras.wrappers.SKLearnRegressor(model=construir_com_semente_fixa)
    imputacao = SimpleImputer(strategy="median", keep_empty_features=True)
    return TransformedTargetRegressor(
        regressor=make_pipeline(imputacao, StandardScaler(), rede),
        transformer=EscalaDaIauc(),
    )


def parametros_treino(config: ConfigRede) -> dict:
    """Parâmetros de `fit` da rede, com o prefixo do pipeline."""
    # Vão no fit da busca e nunca no construtor do SKLearnRegressor. O Keras 3.15.1 perde os
    # fit_kwargs do construtor quando o sklearn clona o estimador, e cada treino rodaria 1 época.
    parametros = {
        "epochs": config.epocas,
        "batch_size": config.tamanho_lote,
        "validation_split": FRACAO_VALIDACAO,
        "callbacks": [
            keras.callbacks.EarlyStopping(patience=config.paciencia, restore_best_weights=True)
        ],
        "verbose": 0,
    }
    return {PREFIXO_FIT + nome: valor for nome, valor in parametros.items()}


def epocas_treinadas(estimador, atributos: pd.DataFrame, iauc: np.ndarray) -> int:
    """Quantas épocas a rede treinou antes do early stopping."""
    return len(estimador.regressor_[-1].history_.epoch)


PONTUACAO = {"mae": "neg_mean_absolute_error", "epocas": epocas_treinadas}


def tabela_treinos(busca: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
    """Uma linha por configuração e dobra, com MAE e épocas."""
    resultados = busca.cv_results_
    linhas = [
        {
            **config[PARAMETRO_REDE],
            "fold": fold,
            "mae": -resultados[f"split{fold}_test_mae"][i],
            "epocas": int(resultados[f"split{fold}_test_epocas"][i]),
        }
        for i, config in enumerate(resultados["params"])
        for fold in range(busca.n_splits_)
    ]
    return pd.DataFrame(linhas)


def treinar(dados: Dados, configs: list[Configuracao], busca: ConfigBusca) -> pd.DataFrame:
    """Treina cada configuração em cada dobra.

    Returns:
        Tabela com os genes, a dobra, o MAE e as épocas.
    """
    grade = GridSearchCV(
        estimador(),
        {PARAMETRO_REDE: configs},
        cv=dados.divisoes,
        scoring=PONTUACAO,
        refit=False,
        n_jobs=busca.processos,
        error_score="raise",
    )
    grade.fit(dados.atributos, dados.iauc, **parametros_treino(busca.treino))
    return tabela_treinos(grade)


def avaliar_no_teste_final(
    dados: Dados, configs: list[Configuracao], busca: ConfigBusca
) -> pd.DataFrame:
    """Treina as configurações no ajuste e mede no teste final."""
    return treinar(replace(dados, divisoes=[dados.teste_final]), configs, busca)


def busca_aleatoria(
    dados: Dados, n_configuracoes: int, busca: ConfigBusca, semente: int
) -> pd.DataFrame:
    """Busca aleatória com `n_configuracoes` sorteios."""
    aleatoria = RandomizedSearchCV(
        estimador(),
        {PARAMETRO_REDE: SorteioConfiguracao()},
        n_iter=n_configuracoes,
        cv=dados.divisoes,
        scoring=PONTUACAO,
        refit=False,
        random_state=semente,
        n_jobs=busca.processos,
        error_score="raise",
    )
    aleatoria.fit(dados.atributos, dados.iauc, **parametros_treino(busca.treino))
    return tabela_treinos(aleatoria)


def torneio(aptidoes: np.ndarray, tamanho: int, rng: np.random.Generator) -> tuple[int, int]:
    """Sorteia candidatos e devolve os dois de menor MAE, como pai e mãe."""
    candidatos = rng.choice(len(aptidoes), tamanho, replace=False)
    primeiro, segundo = candidatos[np.argsort(aptidoes[candidatos])[:2]]
    return primeiro, segundo


def cruzar(
    pai: Configuracao, mae: Configuracao, prob_gene_do_pai: float, rng: np.random.Generator
) -> Configuracao:
    """Cada gene vem do pai com a probabilidade dada, senão da mãe."""
    return {nome: (pai if rng.random() < prob_gene_do_pai else mae)[nome] for nome in GENES}


def mutar(config: Configuracao, busca: ConfigBusca, rng: np.random.Generator) -> Configuracao:
    """Muta cada gene com a taxa da busca."""
    return {
        nome: GENES[nome].mutar(valor, busca.sigma_relativo, rng)
        if rng.random() < busca.taxa_mutacao
        else valor
        for nome, valor in config.items()
    }


def proxima_geracao(
    pop: list[Configuracao], aptidoes: np.ndarray, busca: ConfigBusca, rng: np.random.Generator
) -> list[Configuracao]:
    """Elite mais filhos de torneio, cruzamento e mutação."""
    elite = [pop[i] for i in np.argsort(aptidoes)[: busca.elitismo]]
    filhos = []
    for _ in range(busca.tamanho_populacao - busca.elitismo):
        pai, mae = torneio(aptidoes, busca.tamanho_torneio, rng)
        filho = cruzar(pop[pai], pop[mae], busca.prob_gene_do_pai, rng)
        filhos.append(mutar(filho, busca, rng))
    return elite + filhos


def chave(config: Configuracao) -> tuple:
    """Tupla dos genes, para reconhecer configurações já treinadas."""
    return tuple(config[nome] for nome in GENES)


def busca_genetica(
    dados: Dados, busca: ConfigBusca, semente: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AG sobre os hiperparâmetros. Não retreina configurações repetidas.

    Returns:
        Treinos e população de cada geração.
    """
    rng = np.random.default_rng(semente)
    pop = sortear(busca.tamanho_populacao, semente)
    mae_ja_treinado: dict[tuple, float] = {}
    treinos, populacoes = [], []
    for geracao in range(busca.n_geracoes + 1):
        novas = list({chave(c): c for c in pop if chave(c) not in mae_ja_treinado}.values())
        if novas:
            treinados = treinar(dados, novas, busca).assign(geracao=geracao)
            treinos.append(treinados)
            mae_ja_treinado |= treinados.groupby(list(GENES), sort=False)["mae"].mean().to_dict()
        aptidoes = np.array([mae_ja_treinado[chave(c)] for c in pop])
        populacoes.append(pd.DataFrame(pop).assign(geracao=geracao, mae=aptidoes))
        if geracao < busca.n_geracoes:
            pop = proxima_geracao(pop, aptidoes, busca, rng)
    return pd.concat(treinos, ignore_index=True), pd.concat(populacoes, ignore_index=True)


def rodar_buscas(dados: Dados, busca: ConfigBusca, semente: int) -> Resultado:
    """Roda o AG e, com o mesmo número de configurações, a busca aleatória."""
    inicio = time.perf_counter()
    genetica, populacoes = busca_genetica(dados, busca, semente)
    meio = time.perf_counter()
    n_configuracoes = len(genetica) // len(dados.divisoes)
    aleatoria = busca_aleatoria(dados, n_configuracoes, busca, semente)
    fim = time.perf_counter()
    treinos = pd.concat(
        [genetica.assign(busca="genetica"), aleatoria.assign(busca="aleatoria")],
        ignore_index=True,
    )
    return Resultado(
        treinos.assign(semente=semente),
        populacoes.assign(semente=semente),
        {"genetica": meio - inicio, "aleatoria": fim - meio},
    )


def por_configuracao(treinos: pd.DataFrame) -> pd.DataFrame:
    """Resume os treinos em uma linha por configuração, com o MAE médio."""
    return treinos.groupby(["semente", "busca", *GENES], sort=False, as_index=False).agg(
        mae=("mae", "mean"), treinos=("mae", "size"), epocas_min=("epocas", "min")
    )


def da_busca(configuracoes: pd.DataFrame, busca: str, semente: int) -> pd.DataFrame:
    """Configurações de uma busca e de uma semente."""
    return configuracoes[(configuracoes["busca"] == busca) & (configuracoes["semente"] == semente)]


def melhor_por_treinos(configuracoes: pd.DataFrame) -> pd.Series:
    """Melhor MAE até aqui, em função do número de treinos feitos."""
    return pd.Series(
        configuracoes["mae"].cummin().to_numpy(), index=configuracoes["treinos"].cumsum()
    )


def desenhar_comparacao(treinos: pd.DataFrame, escolhidas: pd.DataFrame, caminho: Path) -> None:
    """Grava o gráfico do melhor MAE contra o número de treinos, para as duas buscas.

    Args:
        treinos: Treinos de todas as sementes.
        escolhidas: Configuração escolhida por busca, com o MAE no teste final.
        caminho: Arquivo PNG de saída.
    """
    figura = Figure(figsize=(7, 5))
    eixo = figura.subplots()
    configuracoes = por_configuracao(treinos)
    sementes = configuracoes["semente"].unique()
    for cor, (busca, rotulo) in zip(CORES_BUSCA, BUSCAS.items(), strict=True):
        legenda = f"{rotulo}, {len(sementes)} sementes" if len(sementes) > 1 else rotulo
        for i, semente in enumerate(sementes):
            curva = melhor_por_treinos(da_busca(configuracoes, busca, semente))
            rotulo_curva = legenda if i == 0 else None
            eixo.step(curva.index, curva.to_numpy(), where="post", color=cor, label=rotulo_curva)
    no_teste = escolhidas.groupby("busca", sort=False)["mae_teste"].mean()
    media = f", média de {len(sementes)} sementes" if len(sementes) > 1 else ""
    padrao = escolhidas.set_index("busca").loc["padrao", "mae"]
    eixo.axhline(padrao, linestyle="--", color="gray", label="configuração padrão de P3")
    rotulos = {"genetica": "AG", "aleatoria": "busca aleatória", "padrao": "padrão de P3"}
    resumo = [f"{rotulo}: {no_teste[busca]:.0f}" for busca, rotulo in rotulos.items()]
    eixo.text(
        0.98,
        0.55,
        "\n".join([f"MAE das melhores no teste final{media}", *resumo]),
        transform=eixo.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "gray"},
    )
    eixo.set_xlabel("treinos feitos")
    eixo.set_ylabel(f"melhor MAE em D7 a D9, GroupKFold({N_FOLDS_BUSCA}) do ajuste (mg/dL·min)")
    eixo.set_title("Melhor MAE contra treinos: AG e busca aleatória")
    eixo.legend(loc="upper right", bbox_to_anchor=(1, 0.93))
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=120, bbox_inches="tight")


def config_padrao() -> Configuracao:
    """Hiperparâmetros padrão da rede, no formato de uma configuração."""
    padrao = ConfigRede()
    return {nome: getattr(padrao, nome) for nome in GENES}


def main() -> None:
    """Roda as buscas, mede as escolhidas no teste final e grava tabelas e figura."""
    argumentos = argparse.ArgumentParser(description="Compara o AG com o RandomizedSearchCV.")
    argumentos.add_argument(
        "--sementes",
        type=int,
        default=N_SEMENTES_PADRAO,
        help="quantas sementes de busca rodar, a partir de SEMENTE",
    )
    sementes = range(SEMENTE, SEMENTE + argumentos.parse_args().sementes)
    refeicoes = pd.read_parquet(ARTEFATOS / "refeicoes.parquet")
    perfis = pd.read_parquet(ARTEFATOS / "perfis.parquet")
    dados = preparar_dados(refeicoes, perfis, SEMENTE)
    busca = ConfigBusca()

    resultados = [rodar_buscas(dados, busca, semente) for semente in sementes]
    treinos = pd.concat([r.treinos for r in resultados], ignore_index=True)
    populacoes = pd.concat([r.populacoes for r in resultados], ignore_index=True)
    mae_padrao = treinar(dados, [config_padrao()], busca)["mae"].mean()

    configuracoes = por_configuracao(treinos)
    melhores = configuracoes.loc[configuracoes.groupby(["semente", "busca"])["mae"].idxmin()]
    padrao = pd.DataFrame([{**config_padrao(), "busca": "padrao", "mae": mae_padrao}])
    escolhidas = pd.concat([melhores, padrao], ignore_index=True)
    configs = escolhidas[list(GENES)].to_dict("records")
    escolhidas["mae_teste"] = avaliar_no_teste_final(dados, configs, busca)["mae"].to_numpy()
    por_busca = escolhidas.set_index(["semente", "busca"])

    treino_final, teste_final = dados.teste_final
    print(
        f"ajuste: {len(np.unique(dados.participantes[treino_final]))} participantes, "
        f"GroupKFold({N_FOLDS_BUSCA}); teste final: "
        f"{len(np.unique(dados.participantes[teste_final]))} participantes fora da busca"
    )
    print("MAE em D7 a D9 (mg/dL·min); cada treino é uma rede estruturada, não o conjunto de P3")
    for semente, resultado in zip(sementes, resultados, strict=True):
        print(f"\nsemente {semente}")
        for nome, rotulo in BUSCAS.items():
            da_semente = da_busca(configuracoes, nome, semente)
            melhor = por_busca.loc[(semente, nome)]
            genes = ", ".join(f"{gene}={melhor[gene]:.4g}" for gene in GENES)
            print(
                f"  {rotulo}: {da_semente['treinos'].sum()} treinos, "
                f"{len(da_semente)} configurações, {resultado.segundos[nome]:.0f} s, "
                f"menor número de épocas de um treino: {da_semente['epocas_min'].min()}"
            )
            print(
                f"    melhor: {genes}, MAE no ajuste {melhor['mae']:.1f}, "
                f"no teste final {melhor['mae_teste']:.1f}"
            )
    if len(sementes) > 1:
        print(f"\nmédia e desvio das {len(sementes)} sementes, ajuste e teste final")
        for nome, rotulo in BUSCAS.items():
            da_busca_toda = escolhidas[escolhidas["busca"] == nome]
            ajuste, teste = da_busca_toda["mae"], da_busca_toda["mae_teste"]
            print(
                f"  {rotulo}: {ajuste.mean():.1f} ({ajuste.std():.1f}), "
                f"{teste.mean():.1f} ({teste.std():.1f})"
            )
    padrao = escolhidas.iloc[-1]
    print(
        f"\nconfiguração padrão de P3 ({config_padrao()}): MAE no ajuste {padrao['mae']:.1f}, "
        f"no teste final {padrao['mae_teste']:.1f}"
    )

    ARTEFATOS.mkdir(exist_ok=True)
    treinos.to_parquet(ARTEFATOS / ARQUIVO_TREINOS)
    populacoes.to_parquet(ARTEFATOS / ARQUIVO_POPULACOES)
    escolhidas.to_parquet(ARTEFATOS / ARQUIVO_TESTE_FINAL)
    desenhar_comparacao(treinos, escolhidas, ARTEFATOS / ARQUIVO_FIGURA)
    print(f"gravado em {ARTEFATOS / ARQUIVO_FIGURA}")


if __name__ == "__main__":
    main()
