import numpy as np
import pandas as pd

from glicose.ag import FAIXA_KCAL, Lambdas, Resultado, Simulador, fitness
from glicose.esquema import TIPOS_REFEICAO
from glicose.plano import REFEICOES_PLANO, Populacao, aleatoria, habito, pratos_do_tipo

__all__ = ["busca_aleatoria", "habito", "menos_carbo"]

TAMANHO_LOTE = 100


def busca_aleatoria(
    n_avaliacoes: int, simular: Simulador, perfil, cardapio, lambdas: Lambdas, semente: int
) -> Resultado:
    rng = np.random.default_rng(semente)
    melhor, melhor_fitness = None, np.inf
    historico = []
    avaliados = 0
    while avaliados < n_avaliacoes:
        lote = aleatoria(min(TAMANHO_LOTE, n_avaliacoes - avaliados), cardapio, rng)
        aptidao = fitness(lote, simular, perfil, cardapio, lambdas)
        avaliados += len(lote)
        if aptidao.min() < melhor_fitness:
            melhor, melhor_fitness = lote[np.argmin(aptidao)], float(aptidao.min())
        historico.append((avaliados, melhor_fitness))
    return Resultado(
        melhor=melhor,
        fitness=melhor_fitness,
        historico=pd.DataFrame(historico, columns=["avaliacoes", "melhor"]),
        avaliacoes=avaliados,
    )


def menos_carbo(perfil: pd.Series, cardapio: pd.DataFrame) -> Populacao:
    kcal = {t: cardapio["kcal"].to_numpy()[pratos_do_tipo(cardapio, t)] for t in TIPOS_REFEICAO}
    carbo = {t: cardapio["carbo"].to_numpy()[pratos_do_tipo(cardapio, t)] for t in TIPOS_REFEICAO}
    minimo_dia = (1 - FAIXA_KCAL) * perfil["kcal_dia"]
    maximo_dia = (1 + FAIXA_KCAL) * perfil["kcal_dia"]

    escolha = {}
    acumulado = 0.0
    for k, tipo in enumerate(TIPOS_REFEICAO):
        restantes = TIPOS_REFEICAO[k + 1 :]
        menor_dia = acumulado + kcal[tipo] + sum(kcal[t].min() for t in restantes)
        maior_dia = acumulado + kcal[tipo] + sum(kcal[t].max() for t in restantes)
        falta = np.maximum(minimo_dia - maior_dia, 0)
        sobra = np.maximum(menor_dia - maximo_dia, 0)
        fora_da_faixa = falta + sobra
        mais_perto_da_faixa = np.nonzero(fora_da_faixa == fora_da_faixa.min())[0]
        escolha[tipo] = mais_perto_da_faixa[np.argmin(carbo[tipo][mais_perto_da_faixa])]
        acumulado += kcal[tipo][escolha[tipo]]

    plano = habito(perfil, cardapio)
    for j, (_, tipo) in enumerate(REFEICOES_PLANO):
        plano.prato[0, j] = escolha[tipo]
    return plano
