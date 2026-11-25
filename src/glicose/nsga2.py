import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from glicose.ag import (
    ConfigAG,
    Simulador,
    cruzamento,
    distancia_habito,
    iauc_d9,
    mutacao_gaussiana,
    mutacao_prato,
    penalidade_kcal,
    penalidade_proteina,
    populacao_inicial,
    sigma,
    simulador_falso,
    vizinhos_prato,
)
from glicose.esquema import dados_sinteticos
from glicose.plano import Populacao, habito

OBJETIVOS = ("iauc_d9", "distancia_habito")
SEMENTE_DADOS = 0
PASTA_ARTEFATOS = Path("artefatos")


# populacao_inicial e sigma, de ag.py, leem só estes campos da configuração. Por isso os nomes
# repetem os de ConfigAG, sem os campos que o NSGA-II não usa.
@dataclass(frozen=True)
class ConfigNSGA:
    tamanho_populacao: int = ConfigAG.tamanho_populacao
    max_geracoes: int = ConfigAG.max_geracoes
    taxa_mutacao_prato: float = ConfigAG.taxa_mutacao_prato
    taxa_mutacao_gaussiana: float = ConfigAG.taxa_mutacao_gaussiana
    sigma_relativo_inicial: float = ConfigAG.sigma_relativo_inicial
    sigma_relativo_final: float = ConfigAG.sigma_relativo_final


@dataclass
class ResultadoNSGA:
    frente: Populacao
    objetivos: np.ndarray
    frentes_por_geracao: list[np.ndarray]
    avaliacoes: int


def calcular_objetivos(pop: Populacao, simular: Simulador, perfil, cardapio) -> np.ndarray:
    return np.column_stack(
        [iauc_d9(pop, simular, perfil, cardapio), distancia_habito(pop, perfil, cardapio)]
    )


def calcular_violacao(pop: Populacao, perfil, cardapio) -> np.ndarray:
    return penalidade_kcal(pop, perfil, cardapio) + penalidade_proteina(pop, perfil, cardapio)


# Dominância com restrições do artigo de Deb et al. (2002).
def matriz_dominancia(objetivos: np.ndarray, violacao: np.ndarray) -> np.ndarray:
    a, b = objetivos[:, None, :], objetivos[None, :, :]
    pareto = (a <= b).all(axis=2) & (a < b).any(axis=2)
    viavel = violacao == 0
    viavel_a, viavel_b = viavel[:, None], viavel[None, :]
    viola_menos = violacao[:, None] < violacao[None, :]
    return (
        (viavel_a & viavel_b & pareto)
        | (viavel_a & ~viavel_b)
        | (~viavel_a & ~viavel_b & viola_menos)
    )


def ordenar_nao_dominados(objetivos: np.ndarray, violacao=None) -> list[np.ndarray]:
    if violacao is None:
        violacao = np.zeros(len(objetivos))
    domina = matriz_dominancia(objetivos, violacao)
    dominado_por = domina.sum(axis=0)
    frentes = []
    atual = [i for i in range(len(objetivos)) if dominado_por[i] == 0]
    while atual:
        frentes.append(np.array(atual))
        proxima = []
        for i in atual:
            for j in np.nonzero(domina[i])[0]:
                dominado_por[j] -= 1
                if dominado_por[j] == 0:
                    proxima.append(j)
        atual = proxima
    return frentes


def distancia_aglomeracao(frente: np.ndarray) -> np.ndarray:
    n, n_objetivos = frente.shape
    distancia = np.zeros(n)
    for m in range(n_objetivos):
        ordem = np.argsort(frente[:, m], kind="stable")
        valores = frente[ordem, m]
        distancia[ordem[0]] = distancia[ordem[-1]] = np.inf
        amplitude = valores[-1] - valores[0]
        if amplitude == 0:
            continue
        for k in range(1, n - 1):
            distancia[ordem[k]] += (valores[k + 1] - valores[k - 1]) / amplitude
    return distancia


def classificar(objetivos: np.ndarray, violacao: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posto = np.empty(len(objetivos), dtype=np.int64)
    aglomeracao = np.empty(len(objetivos))
    for numero, frente in enumerate(ordenar_nao_dominados(objetivos, violacao)):
        posto[frente] = numero
        aglomeracao[frente] = distancia_aglomeracao(objetivos[frente])
    return posto, aglomeracao


def torneio_binario(posto: np.ndarray, aglomeracao: np.ndarray, rng) -> int:
    i, j = rng.choice(len(posto), 2, replace=False)
    if posto[i] != posto[j]:
        return i if posto[i] < posto[j] else j
    return i if aglomeracao[i] >= aglomeracao[j] else j


def primeira_frente_sem_repeticao(pop: Populacao, objetivos, posto) -> tuple[Populacao, np.ndarray]:
    frente = np.nonzero(posto == 0)[0]
    _, primeiros = np.unique(objetivos[frente], axis=0, return_index=True)
    frente = frente[primeiros]
    return pop[frente], objetivos[frente]


def rodar(config: ConfigNSGA, simular: Simulador, perfil, cardapio, semente: int) -> ResultadoNSGA:
    rng = np.random.default_rng(semente)
    vizinhos = vizinhos_prato(cardapio)
    pop = populacao_inicial(config, perfil, cardapio, vizinhos, rng)
    objetivos = calcular_objetivos(pop, simular, perfil, cardapio)
    violacao = calcular_violacao(pop, perfil, cardapio)
    posto, aglomeracao = classificar(objetivos, violacao)
    frentes_por_geracao = [objetivos[posto == 0]]

    for geracao in range(1, config.max_geracoes + 1):
        filhos = []
        for _ in range(config.tamanho_populacao):
            pai = torneio_binario(posto, aglomeracao, rng)
            mae = torneio_binario(posto, aglomeracao, rng)
            filhos.append(cruzamento(pop[pai], pop[mae], rng))
        filhos = mutacao_prato(Populacao.juntar(*filhos), vizinhos, config.taxa_mutacao_prato, rng)
        filhos = mutacao_gaussiana(
            filhos, sigma(geracao, config), config.taxa_mutacao_gaussiana, rng
        )

        juntos = Populacao.juntar(pop, filhos)
        objetivos = np.vstack([objetivos, calcular_objetivos(filhos, simular, perfil, cardapio)])
        violacao = np.concatenate([violacao, calcular_violacao(filhos, perfil, cardapio)])
        posto, aglomeracao = classificar(objetivos, violacao)
        sobreviventes = np.lexsort((-aglomeracao, posto))[: config.tamanho_populacao]

        pop = juntos[sobreviventes]
        objetivos, violacao = objetivos[sobreviventes], violacao[sobreviventes]
        posto, aglomeracao = posto[sobreviventes], aglomeracao[sobreviventes]
        frentes_por_geracao.append(objetivos[posto == 0])

    frente, objetivos_frente = primeira_frente_sem_repeticao(pop, objetivos, posto)
    return ResultadoNSGA(
        frente=frente,
        objetivos=objetivos_frente,
        frentes_por_geracao=frentes_por_geracao,
        avaliacoes=len(frentes_por_geracao) * config.tamanho_populacao,
    )


def desenhar_frentes(frentes: dict[str, tuple[np.ndarray, np.ndarray]], caminho: Path) -> None:
    figura = Figure(figsize=(7, 5))
    eixo = figura.subplots()
    for rotulo, (objetivos, ponto_habito) in frentes.items():
        linha = eixo.plot(*objetivos.T, "o-", label=f"frente de {rotulo}")[0]
        eixo.plot(*ponto_habito, "*", markersize=14, color=linha.get_color())
    eixo.set_xlabel("iAUC prevista de D9 (mg/dL·min)")
    eixo.set_ylabel("distância do hábito (desvios-padrão)")
    eixo.set_title("Frente de Pareto do NSGA-II (estrela = hábito)")
    eixo.legend()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=120, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda o NSGA-II de D7 a D9.")
    parser.add_argument("--falso", action="store_true", help="usa o simulador falso")
    parser.add_argument("--semente", type=int, default=0)
    parser.add_argument("--participantes", type=int, default=1)
    args = parser.parse_args()
    if not args.falso:
        parser.error("por enquanto só existe o simulador falso, use --falso")

    dados = dados_sinteticos(args.participantes, semente=SEMENTE_DADOS)
    cardapio = dados["cardapio"]
    frentes = {}
    for _, perfil in dados["perfis"].iterrows():
        resultado = rodar(ConfigNSGA(), simulador_falso, perfil, cardapio, args.semente)
        violacoes = calcular_violacao(resultado.frente, perfil, cardapio) > 0
        ponto_habito = calcular_objetivos(
            habito(perfil, cardapio), simulador_falso, perfil, cardapio
        )
        print(f"participante {perfil['participante']} ({perfil['grupo']}), semente {args.semente}")
        print(f"tamanho da frente final: {len(resultado.frente)}")
        print(f"planos da frente que violam kcal ou proteina: {violacoes.sum()}")
        print(f"avaliacoes: {resultado.avaliacoes}")
        print(pd.DataFrame(resultado.objetivos, columns=OBJETIVOS).round(2).to_string(index=False))
        print()
        frentes[f"participante {perfil['participante']}"] = (resultado.objetivos, ponto_habito[0])

    caminho = PASTA_ARTEFATOS / "frente_falso.png"
    desenhar_frentes(frentes, caminho)
    print(f"figura gravada em {caminho}")


if __name__ == "__main__":
    main()
