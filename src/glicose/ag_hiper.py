import math
import time
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd
from scipy.stats import reciprocal, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    ParameterSampler,
    RandomizedSearchCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from glicose.esquema import DIAS_HABITO, DIAS_PLANO
from glicose.simulador import (
    ALVOS,
    FRACAO_VALIDACAO,
    SEMENTE,
    ConfigRede,
    construir_modelo,
    particao_do_estudo,
)

Configuracao = dict[str, int | float]

N_FOLDS_BUSCA = 3
ALVO_REDE = "log1p"
PARAMETRO_REDE = "regressor__sklearnregressor__model_kwargs"
PREFIXO_FIT = "sklearnregressor__"


@dataclass(frozen=True)
class Gene:
    minimo: float
    maximo: float
    inteiro: bool = False
    log: bool = False

    def distribuicao(self):
        if self.inteiro:
            return np.arange(self.minimo, self.maximo + 1)
        if self.log:
            return reciprocal(self.minimo, self.maximo)
        return uniform(self.minimo, self.maximo - self.minimo)

    def normalizar(self, valor: float) -> int | float:
        return int(round(valor)) if self.inteiro else float(valor)

    def mutar(self, valor: float, sigma_relativo: float, rng: np.random.Generator) -> int | float:
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
    tamanho_populacao: int = 10
    n_geracoes: int = 8
    tamanho_torneio: int = 3
    elitismo: int = 1
    prob_gene_do_pai: float = 0.5
    taxa_mutacao: float = 0.25
    sigma_relativo: float = 0.2
    processos: int = 6
    treino: ConfigRede = ConfigRede()


@dataclass(frozen=True)
class Dados:
    atributos: pd.DataFrame
    iauc: np.ndarray
    divisoes: list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Resultado:
    treinos: pd.DataFrame
    populacoes: pd.DataFrame
    segundos: dict[str, float]


def sortear(n: int, semente: int | np.random.RandomState) -> list[Configuracao]:
    distribuicoes = {nome: gene.distribuicao() for nome, gene in GENES.items()}
    sorteadas = ParameterSampler(distribuicoes, n, random_state=semente)
    return [{nome: GENES[nome].normalizar(c[nome]) for nome in GENES} for c in sorteadas]


class SorteioConfiguracao:
    def rvs(self, random_state=None) -> Configuracao:
        return sortear(1, random_state)[0]


def corte_keras(n_linhas: int) -> int:
    return math.floor(n_linhas * (1.0 - FRACAO_VALIDACAO))


def agrupar_e_cortar_para_validacao(
    linhas: np.ndarray, participantes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    # O validation_split do Keras valida nas linhas de corte_keras(n) em diante, sem sortear.
    sorteio = rng.permutation(np.unique(participantes[linhas]))
    posicao = pd.Series(np.arange(len(sorteio)), index=sorteio)
    linhas = linhas[np.argsort(posicao[participantes[linhas]].to_numpy(), kind="stable")]
    grupo = participantes[linhas]
    fronteiras = set(np.flatnonzero(grupo[1:] != grupo[:-1]) + 1)
    total = next(n for n in range(len(linhas), 0, -1) if corte_keras(n) in fronteiras)
    return linhas[:total]


def preparar_dados(refeicoes: pd.DataFrame, perfis: pd.DataFrame, semente: int) -> Dados:
    estudo = particao_do_estudo(refeicoes, perfis)
    participantes = estudo.refeicoes["participante"].to_numpy()
    habito = estudo.refeicoes["dia"].isin(DIAS_HABITO).to_numpy()
    plano = estudo.refeicoes["dia"].isin(DIAS_PLANO).to_numpy()
    rng = np.random.default_rng(semente)
    divisoes = [
        (
            agrupar_e_cortar_para_validacao(treino[habito[treino]], participantes, rng),
            teste[plano[teste]],
        )
        for treino, teste in GroupKFold(N_FOLDS_BUSCA).split(estudo.atributos, groups=participantes)
    ]
    return Dados(estudo.atributos, estudo.iauc, divisoes)


def construir_com_semente_fixa(X, y, **genes) -> keras.Model:
    keras.utils.set_random_seed(SEMENTE)
    # O predict do SKLearnRegressor não repassa verbose ao Keras.
    keras.config.disable_interactive_logging()
    return construir_modelo(X.shape[1], **genes)


def estimador() -> TransformedTargetRegressor:
    alvo = ALVOS[ALVO_REDE]
    rede = keras.wrappers.SKLearnRegressor(model=construir_com_semente_fixa)
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), rede),
        transformer=make_pipeline(FunctionTransformer(alvo.ida, alvo.volta), StandardScaler()),
    )


def parametros_treino(config: ConfigRede) -> dict:
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
    return len(estimador.regressor_[-1].history_.epoch)


PONTUACAO = {"mae": "neg_mean_absolute_error", "epocas": epocas_treinadas}


def tabela_treinos(busca: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
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


def busca_aleatoria(
    dados: Dados, n_configuracoes: int, busca: ConfigBusca, semente: int
) -> pd.DataFrame:
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
    candidatos = rng.choice(len(aptidoes), tamanho, replace=False)
    primeiro, segundo = candidatos[np.argsort(aptidoes[candidatos])[:2]]
    return primeiro, segundo


def cruzar(
    pai: Configuracao, mae: Configuracao, prob_gene_do_pai: float, rng: np.random.Generator
) -> Configuracao:
    return {nome: (pai if rng.random() < prob_gene_do_pai else mae)[nome] for nome in GENES}


def mutar(config: Configuracao, busca: ConfigBusca, rng: np.random.Generator) -> Configuracao:
    return {
        nome: GENES[nome].mutar(valor, busca.sigma_relativo, rng)
        if rng.random() < busca.taxa_mutacao
        else valor
        for nome, valor in config.items()
    }


def proxima_geracao(
    pop: list[Configuracao], aptidoes: np.ndarray, busca: ConfigBusca, rng: np.random.Generator
) -> list[Configuracao]:
    elite = [pop[i] for i in np.argsort(aptidoes)[: busca.elitismo]]
    filhos = []
    for _ in range(busca.tamanho_populacao - busca.elitismo):
        pai, mae = torneio(aptidoes, busca.tamanho_torneio, rng)
        filho = cruzar(pop[pai], pop[mae], busca.prob_gene_do_pai, rng)
        filhos.append(mutar(filho, busca, rng))
    return elite + filhos


def chave(config: Configuracao) -> tuple:
    return tuple(config[nome] for nome in GENES)


def busca_genetica(
    dados: Dados, busca: ConfigBusca, semente: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return Resultado(treinos, populacoes, {"genetica": meio - inicio, "aleatoria": fim - meio})
