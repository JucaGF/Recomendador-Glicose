import itertools

import numpy as np
import pandas as pd

from glicose.ag import (
    Lambdas,
    Resultado,
    Simulador,
    aptidao,
    penalidade_kcal,
    penalidade_proteina,
    total_por_dia,
)
from glicose.esquema import DIAS_PLANO, TIPOS_REFEICAO
from glicose.plano import Populacao, aleatoria, habito, pratos_do_tipo

__all__ = ["busca_aleatoria", "habito", "menos_carbo"]

TAMANHO_LOTE = 100


def busca_aleatoria(
    n_avaliacoes: int, simular: Simulador, perfil, cardapio, lambdas: Lambdas, semente: int
) -> Resultado:
    rng = np.random.default_rng(semente)
    melhor, melhor_aptidao = None, np.inf
    historico = []
    avaliados = 0
    while avaliados < n_avaliacoes:
        lote = aleatoria(min(TAMANHO_LOTE, n_avaliacoes - avaliados), cardapio, rng)
        notas = aptidao(lote, simular, perfil, cardapio, lambdas)
        avaliados += len(lote)
        if notas.min() < melhor_aptidao:
            melhor, melhor_aptidao = lote[np.argmin(notas)], float(notas.min())
        historico.append((avaliados, melhor_aptidao))
    return Resultado(
        melhor=melhor,
        aptidao=melhor_aptidao,
        historico=pd.DataFrame(historico, columns=["avaliacoes", "melhor"]),
        avaliacoes=avaliados,
    )


def menos_carbo(perfil: pd.Series, cardapio: pd.DataFrame) -> Populacao:
    """Horário e caminhada vêm do hábito para a comparação medir só os pratos."""
    quantos = [len(pratos_do_tipo(cardapio, tipo)) for tipo in TIPOS_REFEICAO]
    dias = np.array(list(itertools.product(*map(range, quantos))))
    planos = habito(perfil, cardapio)[np.zeros(len(dias), dtype=int)]
    planos.prato_no_tipo = np.tile(dias, len(DIAS_PLANO))
    penalidade = penalidade_kcal(planos, perfil, cardapio) + penalidade_proteina(
        planos, perfil, cardapio
    )
    carbo = total_por_dia(planos, cardapio, "carbo").sum(axis=1)
    melhor = min(range(len(dias)), key=lambda i: (penalidade[i], carbo[i]))
    return planos[melhor]
