import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from glicose.esquema import TIPOS_REFEICAO, dados_sinteticos
from glicose.plano import (
    LIMITES_CAMINHADA,
    N_REFEICOES,
    REFEICOES_PLANO,
    Populacao,
    aleatoria,
    habito,
    indices_cardapio,
    limites_intervalo,
    macros_habito,
    macros_normalizados,
    para_refeicoes,
    pratos_do_tipo,
)

Simulador = Callable[[pd.DataFrame], np.ndarray]

POSICOES_D9 = [j for j, (dia, _) in enumerate(REFEICOES_PLANO) if dia == 9]
DIA_9 = -1
FAIXA_KCAL = 0.20
PROTEINA_MINIMA = 0.80
VIZINHOS_PRATO = 3
SEMENTE_DADOS = 0


# As penalidades de kcal e proteína são frações (0,1 = 10% fora da faixa), e a iAUC de uma
# refeição real fica perto de 2500 mg/dL·min. Com peso 100 000, sair 1% da faixa custa 1000,
# então o AG prefere ficar na faixa. A distância do hábito é medida em desvios-padrão do cardápio,
# e com peso 500 cada desvio custa cerca de 1/5 da iAUC de uma refeição.
@dataclass(frozen=True)
class Lambdas:
    kcal: float = 100_000.0
    proteina: float = 100_000.0
    habito: float = 500.0


@dataclass(frozen=True)
class ConfigAG:
    tamanho_populacao: int = 100
    max_geracoes: int = 200
    # Com 30 gerações sem melhora, o AG achou o ótimo da força bruta em 65 de 100 sementes.
    # Com 120, achou em 100 de 100.
    paciencia: int = 120
    tamanho_torneio: int = 4
    elitismo: int = 2
    taxa_mutacao_prato: float = 0.1
    taxa_mutacao_gaussiana: float = 0.2
    sigma_relativo_inicial: float = 0.2
    sigma_relativo_final: float = 0.01
    lambdas: Lambdas = field(default_factory=Lambdas)


@dataclass
class Resultado:
    melhor: Populacao
    fitness: float
    historico: pd.DataFrame
    avaliacoes: int


def simulador_falso(refeicoes: pd.DataFrame) -> np.ndarray:
    return 1.5 * refeicoes["carbo"].to_numpy() - 10 * refeicoes["minutos_ativos_60"].to_numpy()


def iauc_d9(pop: Populacao, simular: Simulador, perfil: pd.Series, cardapio) -> np.ndarray:
    iauc = np.asarray(simular(para_refeicoes(pop, cardapio, perfil))).reshape(len(pop), N_REFEICOES)
    return iauc[:, POSICOES_D9].sum(axis=1)


def total_por_dia(pop: Populacao, cardapio: pd.DataFrame, coluna: str) -> np.ndarray:
    valores = cardapio[coluna].to_numpy()[indices_cardapio(pop, cardapio)]
    return valores.reshape(len(pop), -1, len(TIPOS_REFEICAO)).sum(axis=2)


def penalidade_kcal(pop: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    desvio = np.abs(total_por_dia(pop, cardapio, "kcal") / perfil["kcal_dia"] - 1)
    return np.maximum(desvio - FAIXA_KCAL, 0).sum(axis=1)


def penalidade_proteina(pop: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    fracao = total_por_dia(pop, cardapio, "proteina") / perfil["proteina_dia"]
    return np.maximum(PROTEINA_MINIMA - fracao, 0).sum(axis=1)


def distancia_habito(pop: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    plano = macros_normalizados(cardapio)[indices_cardapio(pop, cardapio)]
    return np.linalg.norm(plano - macros_habito(perfil, cardapio), axis=2).mean(axis=1)


def fitness(pop, simular, perfil, cardapio, lambdas: Lambdas) -> np.ndarray:
    return (
        iauc_d9(pop, simular, perfil, cardapio)
        + lambdas.kcal * penalidade_kcal(pop, perfil, cardapio)
        + lambdas.proteina * penalidade_proteina(pop, perfil, cardapio)
        + lambdas.habito * distancia_habito(pop, perfil, cardapio)
    )


# O torneio da aula: sorteia `tamanho` indivíduos, e os 2 melhores viram pai e mãe.
def torneio(aptidao: np.ndarray, tamanho: int, rng: np.random.Generator) -> tuple[int, int]:
    candidatos = rng.choice(len(aptidao), tamanho, replace=False)
    primeiro, segundo = candidatos[np.argsort(aptidao[candidatos])[:2]]
    return primeiro, segundo


def cruzamento(pai: Populacao, mae: Populacao, rng: np.random.Generator) -> Populacao:
    do_pai = rng.random(N_REFEICOES) < 0.5
    return Populacao(
        np.where(do_pai, pai.prato, mae.prato),
        np.where(do_pai, pai.intervalo, mae.intervalo),
        np.where(do_pai, pai.caminhada, mae.caminhada),
    )


def vizinhos_prato(cardapio: pd.DataFrame) -> dict[str, np.ndarray]:
    macros = macros_normalizados(cardapio)
    vizinhos = {}
    for tipo in TIPOS_REFEICAO:
        do_tipo = macros[pratos_do_tipo(cardapio, tipo)]
        distancias = np.linalg.norm(do_tipo[:, None, :] - do_tipo[None, :, :], axis=2)
        np.fill_diagonal(distancias, np.inf)
        quantos = min(VIZINHOS_PRATO, len(do_tipo) - 1)
        vizinhos[tipo] = np.argsort(distancias, axis=1)[:, :quantos]
    return vizinhos


def mutacao_prato(pop: Populacao, vizinhos, taxa: float, rng: np.random.Generator) -> Populacao:
    novo = pop[np.arange(len(pop))]
    sorteados = rng.random(novo.prato.shape) < taxa
    for i, j in zip(*np.nonzero(sorteados), strict=True):
        tipo = REFEICOES_PLANO[j][1]
        novo.prato[i, j] = rng.choice(vizinhos[tipo][novo.prato[i, j]])
    return novo


def mutacao_gaussiana(pop: Populacao, sigma: float, taxa: float, rng) -> Populacao:
    minimos, maximos = limites_intervalo()
    caminhada_min, caminhada_max = LIMITES_CAMINHADA
    forma = pop.intervalo.shape
    ruido_intervalo = rng.normal(0, sigma * (maximos - minimos), forma)
    ruido_caminhada = rng.normal(0, sigma * (caminhada_max - caminhada_min), forma)
    intervalo = pop.intervalo + np.where(rng.random(forma) < taxa, ruido_intervalo, 0)
    caminhada = pop.caminhada + np.where(rng.random(forma) < taxa, ruido_caminhada, 0)
    return Populacao(
        pop.prato.copy(),
        np.clip(intervalo, minimos, maximos),
        np.clip(caminhada, caminhada_min, caminhada_max),
    )


def sigma(geracao: int, config: ConfigAG) -> float:
    inicial, final = config.sigma_relativo_inicial, config.sigma_relativo_final
    return inicial + (final - inicial) * geracao / config.max_geracoes


def diversidade(pop: Populacao) -> float:
    return np.mean([len(np.unique(pop.prato[:, j])) for j in range(N_REFEICOES)])


def populacao_inicial(config: ConfigAG, perfil, cardapio, vizinhos, rng) -> Populacao:
    metade = config.tamanho_populacao // 2
    copias = habito(perfil, cardapio)[np.zeros(config.tamanho_populacao - metade, dtype=int)]
    copias = mutacao_prato(copias, vizinhos, config.taxa_mutacao_prato, rng)
    copias = mutacao_gaussiana(copias, sigma(0, config), config.taxa_mutacao_gaussiana, rng)
    return Populacao.juntar(aleatoria(metade, cardapio, rng), copias)


def rodar(config: ConfigAG, simular: Simulador, perfil, cardapio, semente: int) -> Resultado:
    rng = np.random.default_rng(semente)
    vizinhos = vizinhos_prato(cardapio)
    pop = populacao_inicial(config, perfil, cardapio, vizinhos, rng)
    aptidao = fitness(pop, simular, perfil, cardapio, config.lambdas)
    historico = [(0, aptidao.min(), aptidao.mean(), sigma(0, config), diversidade(pop))]
    melhor_ate_agora = aptidao.min()
    geracoes_sem_melhora = 0

    for geracao in range(1, config.max_geracoes + 1):
        elite = pop[np.argsort(aptidao)[: config.elitismo]]
        filhos = []
        for _ in range(config.tamanho_populacao - config.elitismo):
            pai, mae = torneio(aptidao, config.tamanho_torneio, rng)
            filhos.append(cruzamento(pop[pai], pop[mae], rng))
        filhos = mutacao_prato(Populacao.juntar(*filhos), vizinhos, config.taxa_mutacao_prato, rng)
        sigma_geracao = sigma(geracao, config)
        filhos = mutacao_gaussiana(filhos, sigma_geracao, config.taxa_mutacao_gaussiana, rng)
        pop = Populacao.juntar(elite, filhos)
        aptidao = fitness(pop, simular, perfil, cardapio, config.lambdas)
        historico.append((geracao, aptidao.min(), aptidao.mean(), sigma_geracao, diversidade(pop)))

        if aptidao.min() < melhor_ate_agora:
            melhor_ate_agora = aptidao.min()
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1
        if geracoes_sem_melhora >= config.paciencia:
            break

    melhor = np.argmin(aptidao)
    return Resultado(
        melhor=pop[melhor],
        fitness=float(aptidao[melhor]),
        historico=pd.DataFrame(
            historico, columns=["geracao", "melhor", "media", "sigma", "diversidade"]
        ),
        avaliacoes=len(historico) * config.tamanho_populacao,
    )


def mostrar_plano(plano: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> str:
    tabela = para_refeicoes(plano, cardapio, perfil)
    tabela.insert(3, "prato", indices_cardapio(plano, cardapio).ravel())
    colunas = ["dia", "tipo", "horario", "prato", "carbo", "proteina", "kcal", "minutos_ativos_60"]
    return tabela[colunas].round(1).to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda o algoritmo genético de D7 a D9.")
    parser.add_argument("--falso", action="store_true", help="usa o simulador falso")
    parser.add_argument("--semente", type=int, default=0)
    parser.add_argument("--participantes", type=int, default=1)
    parser.add_argument("--sem-penalidades", action="store_true")
    args = parser.parse_args()
    if not args.falso:
        parser.error("por enquanto só existe o simulador falso, use --falso")

    dados = dados_sinteticos(args.participantes, semente=SEMENTE_DADOS)
    lambdas = Lambdas(0.0, 0.0, 0.0) if args.sem_penalidades else Lambdas()
    config = ConfigAG(lambdas=lambdas)
    for _, perfil in dados["perfis"].iterrows():
        resultado = rodar(config, simulador_falso, perfil, dados["cardapio"], args.semente)
        inicio = resultado.historico["melhor"].iloc[0]
        ultima = resultado.historico["geracao"].iloc[-1]
        kcal_d9 = total_por_dia(resultado.melhor, dados["cardapio"], "kcal")[0, DIA_9]
        print(f"participante {perfil['participante']} ({perfil['grupo']}), semente {args.semente}")
        print(f"fitness da geracao 0: {inicio:.1f}")
        print(f"fitness final: {resultado.fitness:.1f} na geracao {ultima}")
        print(f"avaliacoes: {resultado.avaliacoes}")
        print(f"kcal de D9: {kcal_d9:.0f} ({kcal_d9 / perfil['kcal_dia']:.0%} do habito)")
        print(mostrar_plano(resultado.melhor, perfil, dados["cardapio"]))
        print()


if __name__ == "__main__":
    main()
