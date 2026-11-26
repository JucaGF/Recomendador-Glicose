import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from glicose.esquema import DIAS_PLANO, TIPOS_REFEICAO, dados_sinteticos
from glicose.plano import (
    LIMITES_CAMINHADA,
    N_REFEICOES,
    REFEICOES_PLANO,
    Populacao,
    aleatoria,
    distancia_nos_macros_presentes,
    falta_proteina,
    fora_da_faixa_kcal,
    habito,
    indices_cardapio,
    limites_intervalo,
    macros_habito,
    macros_normalizados,
    para_refeicoes,
    pratos_do_tipo,
)

# Devolve um valor por linha da tabela ENTRADA_SIMULADOR, na mesma ordem. Uma saída fora de ordem
# não é detectada e dá a iAUC ao plano errado.
Simulador = Callable[[pd.DataFrame], np.ndarray]

POSICOES_D9 = [j for j, (dia, _) in enumerate(REFEICOES_PLANO) if dia == 9]
DIA_9 = -1
VIZINHOS_PRATO = 3
PROB_SALTO_PRATO = 0.1
PROB_GENE_DO_PAI = 0.5
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
    paciencia: int = 120
    tamanho_torneio: int = 4
    elitismo: int = 2
    taxa_mutacao_prato: float = 0.1
    taxa_mutacao_gaussiana: float = 0.2
    sigma_relativo_inicial: float = 0.2
    sigma_relativo_final: float = 0.01
    # No CGMacros cada pessoa come quase sempre na mesma hora, então o efeito do horário que a
    # rede prevê é confundido.
    horario_livre: bool = False
    lambdas: Lambdas = field(default_factory=Lambdas)


@dataclass
class Resultado:
    melhor: Populacao
    aptidao: float
    historico: pd.DataFrame
    avaliacoes: int


def simulador_falso(refeicoes: pd.DataFrame) -> np.ndarray:
    return 1.5 * refeicoes["carbo"].to_numpy() - 10 * refeicoes["minutos_ativos_60"].to_numpy()


def iauc_d9(
    pop: Populacao, simular: Simulador, perfil: pd.Series, cardapio: pd.DataFrame
) -> np.ndarray:
    iauc_por_linha = np.asarray(simular(para_refeicoes(pop, cardapio, perfil)))
    iauc_por_refeicao = iauc_por_linha.reshape(len(pop), N_REFEICOES)
    return iauc_por_refeicao[:, POSICOES_D9].sum(axis=1)


def total_por_dia(pop: Populacao, cardapio: pd.DataFrame, coluna: str) -> np.ndarray:
    por_refeicao = cardapio[coluna].to_numpy()[indices_cardapio(pop, cardapio)]
    por_dia_e_tipo = por_refeicao.reshape(len(pop), len(DIAS_PLANO), len(TIPOS_REFEICAO))
    return por_dia_e_tipo.sum(axis=2)


def penalidade_kcal(pop: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    return fora_da_faixa_kcal(total_por_dia(pop, cardapio, "kcal"), perfil).sum(axis=1)


def penalidade_proteina(pop: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    return falta_proteina(total_por_dia(pop, cardapio, "proteina"), perfil).sum(axis=1)


def distancia_habito(pop: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    alvo = macros_habito(perfil, cardapio)
    tem_habito = ~np.isnan(alvo).all(axis=1)
    macros_plano = macros_normalizados(cardapio)[indices_cardapio(pop, cardapio)]
    distancia_por_refeicao = distancia_nos_macros_presentes(
        macros_plano[:, tem_habito], alvo[tem_habito]
    )
    return distancia_por_refeicao.mean(axis=1)


def aptidao(
    pop: Populacao,
    simular: Simulador,
    perfil: pd.Series,
    cardapio: pd.DataFrame,
    lambdas: Lambdas,
) -> np.ndarray:
    """Menor é melhor, porque o AG minimiza. As penalidades contam D7 a D9, e a iAUC só D9."""
    return (
        iauc_d9(pop, simular, perfil, cardapio)
        + lambdas.kcal * penalidade_kcal(pop, perfil, cardapio)
        + lambdas.proteina * penalidade_proteina(pop, perfil, cardapio)
        + lambdas.habito * distancia_habito(pop, perfil, cardapio)
    )


def torneio(aptidoes: np.ndarray, tamanho: int, rng: np.random.Generator) -> tuple[int, int]:
    candidatos = rng.choice(len(aptidoes), tamanho, replace=False)
    primeiro, segundo = candidatos[np.argsort(aptidoes[candidatos])[:2]]
    return primeiro, segundo


def cruzamento(pai: Populacao, mae: Populacao, rng: np.random.Generator) -> Populacao:
    do_pai = rng.random(N_REFEICOES) < PROB_GENE_DO_PAI
    return Populacao(
        np.where(do_pai, pai.prato_no_tipo, mae.prato_no_tipo),
        np.where(do_pai, pai.intervalo, mae.intervalo),
        np.where(do_pai, pai.caminhada, mae.caminhada),
    )


def vizinhos_prato(cardapio: pd.DataFrame) -> dict[str, np.ndarray]:
    macros = macros_normalizados(cardapio)
    vizinhos = {}
    for tipo in TIPOS_REFEICAO:
        do_tipo = macros[pratos_do_tipo(cardapio, tipo)]
        diferenca_entre_pares = do_tipo[:, None, :] - do_tipo[None, :, :]
        distancias = np.linalg.norm(diferenca_entre_pares, axis=2)
        np.fill_diagonal(distancias, np.inf)
        quantos = min(VIZINHOS_PRATO, len(do_tipo) - 1)
        vizinhos[tipo] = np.argsort(distancias, axis=1)[:, :quantos]
    return vizinhos


def mutacao_prato(
    pop: Populacao, vizinhos: dict[str, np.ndarray], taxa: float, rng: np.random.Generator
) -> Populacao:
    novo = pop.copiar()
    sorteados = rng.random(novo.prato_no_tipo.shape) < taxa
    for i, j in zip(*np.nonzero(sorteados), strict=True):
        vizinhos_do_tipo = vizinhos[REFEICOES_PLANO[j][1]]
        if rng.random() < PROB_SALTO_PRATO:
            novo.prato_no_tipo[i, j] = rng.integers(len(vizinhos_do_tipo))
        else:
            novo.prato_no_tipo[i, j] = rng.choice(vizinhos_do_tipo[novo.prato_no_tipo[i, j]])
    return novo


def mutacao_gaussiana(
    pop: Populacao,
    sigma: float,
    taxa: float,
    rng: np.random.Generator,
    mutar_horario: bool = False,
) -> Populacao:
    forma = pop.intervalo.shape
    caminhada_min, caminhada_max = LIMITES_CAMINHADA
    ruido_caminhada = rng.normal(0, sigma * (caminhada_max - caminhada_min), forma)
    caminhada = pop.caminhada + np.where(rng.random(forma) < taxa, ruido_caminhada, 0)
    intervalo = pop.intervalo.copy()
    if mutar_horario:
        minimos, maximos = limites_intervalo()
        ruido_intervalo = rng.normal(0, sigma * (maximos - minimos), forma)
        intervalo += np.where(rng.random(forma) < taxa, ruido_intervalo, 0)
        intervalo = np.clip(intervalo, minimos, maximos)
    return Populacao(
        pop.prato_no_tipo.copy(), intervalo, np.clip(caminhada, caminhada_min, caminhada_max)
    )


def sigma(geracao: int, config: ConfigAG) -> float:
    inicial, final = config.sigma_relativo_inicial, config.sigma_relativo_final
    return inicial + (final - inicial) * geracao / config.max_geracoes


def diversidade(pop: Populacao) -> float:
    return float(np.mean([len(np.unique(pop.prato_no_tipo[:, j])) for j in range(N_REFEICOES)]))


def populacao_inicial(
    config: ConfigAG,
    perfil: pd.Series,
    cardapio: pd.DataFrame,
    vizinhos: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Populacao:
    referencia = habito(perfil, cardapio)
    metade = config.tamanho_populacao // 2
    aleatorios = aleatoria(metade, cardapio, rng)
    if not config.horario_livre:
        aleatorios.intervalo[:] = referencia.intervalo
    copias = referencia[np.zeros(config.tamanho_populacao - metade, dtype=int)]
    copias = mutacao_prato(copias, vizinhos, config.taxa_mutacao_prato, rng)
    copias = mutacao_gaussiana(
        copias, sigma(0, config), config.taxa_mutacao_gaussiana, rng, config.horario_livre
    )
    return Populacao.juntar(aleatorios, copias)


def rodar(
    config: ConfigAG,
    simular: Simulador,
    perfil: pd.Series,
    cardapio: pd.DataFrame,
    semente: int,
) -> Resultado:
    rng = np.random.default_rng(semente)
    vizinhos = vizinhos_prato(cardapio)
    pop = populacao_inicial(config, perfil, cardapio, vizinhos, rng)
    aptidoes = aptidao(pop, simular, perfil, cardapio, config.lambdas)
    historico = [(0, aptidoes.min(), aptidoes.mean(), sigma(0, config), diversidade(pop))]
    melhor_ate_agora = aptidoes.min()
    geracoes_sem_melhora = 0

    for geracao in range(1, config.max_geracoes + 1):
        elite = pop[np.argsort(aptidoes)[: config.elitismo]]
        filhos = []
        for _ in range(config.tamanho_populacao - config.elitismo):
            pai, mae = torneio(aptidoes, config.tamanho_torneio, rng)
            filhos.append(cruzamento(pop[pai], pop[mae], rng))
        filhos = mutacao_prato(Populacao.juntar(*filhos), vizinhos, config.taxa_mutacao_prato, rng)
        sigma_geracao = sigma(geracao, config)
        filhos = mutacao_gaussiana(
            filhos, sigma_geracao, config.taxa_mutacao_gaussiana, rng, config.horario_livre
        )
        pop = Populacao.juntar(elite, filhos)
        aptidoes = aptidao(pop, simular, perfil, cardapio, config.lambdas)
        historico.append(
            (geracao, aptidoes.min(), aptidoes.mean(), sigma_geracao, diversidade(pop))
        )

        if aptidoes.min() < melhor_ate_agora:
            melhor_ate_agora = aptidoes.min()
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1
        if geracoes_sem_melhora >= config.paciencia:
            break

    melhor = np.argmin(aptidoes)
    return Resultado(
        melhor=pop[melhor],
        aptidao=float(aptidoes[melhor]),
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
        print(f"aptidao da geracao 0: {inicio:.1f}")
        print(f"aptidao final: {resultado.aptidao:.1f} na geracao {ultima}")
        print(f"avaliacoes: {resultado.avaliacoes}")
        print(f"kcal de D9: {kcal_d9:.0f} ({kcal_d9 / perfil['kcal_dia']:.0%} do habito)")
        print(mostrar_plano(resultado.melhor, perfil, dados["cardapio"]))
        print()


if __name__ == "__main__":
    main()
