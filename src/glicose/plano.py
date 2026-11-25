from dataclasses import dataclass

import numpy as np
import pandas as pd

from glicose.esquema import (
    DIAS_PLANO,
    ENTRADA_SIMULADOR,
    HORAS_DESDE_REFEICAO_ANTERIOR,
    MACROS,
    TIPOS_REFEICAO,
    conformar,
)

REFEICOES_PLANO = tuple((dia, tipo) for dia in DIAS_PLANO for tipo in TIPOS_REFEICAO)
N_REFEICOES = len(REFEICOES_PLANO)

# 99% dos jantares reais do CGMacros acontecem antes de 22,9 horas (Apêndice A do plano).
HORA_MAXIMA_JANTAR = 23.0
LIMITES_CAMINHADA = (0.0, 30.0)
FAIXA_KCAL = 0.20
PROTEINA_MINIMA = 0.80


@dataclass
class Populacao:
    prato_no_tipo: np.ndarray
    intervalo: np.ndarray
    caminhada: np.ndarray

    def __len__(self) -> int:
        return len(self.prato_no_tipo)

    def __getitem__(self, indices: int | np.ndarray) -> "Populacao":
        indices = np.atleast_1d(indices)
        return Populacao(
            self.prato_no_tipo[indices].copy(),
            self.intervalo[indices].copy(),
            self.caminhada[indices].copy(),
        )

    def copiar(self) -> "Populacao":
        return Populacao(self.prato_no_tipo.copy(), self.intervalo.copy(), self.caminhada.copy())

    @staticmethod
    def juntar(*partes: "Populacao") -> "Populacao":
        return Populacao(
            np.concatenate([p.prato_no_tipo for p in partes]),
            np.concatenate([p.intervalo for p in partes]),
            np.concatenate([p.caminhada for p in partes]),
        )


def limites_intervalo() -> tuple[np.ndarray, np.ndarray]:
    por_refeicao = np.array([HORAS_DESDE_REFEICAO_ANTERIOR[tipo] for _, tipo in REFEICOES_PLANO])
    _, minimos, maximos = por_refeicao.T
    return minimos, maximos


def pratos_do_tipo(cardapio: pd.DataFrame, tipo: str) -> np.ndarray:
    return cardapio.loc[cardapio["tipo"] == tipo, "prato"].to_numpy()


def indices_cardapio(pop: Populacao, cardapio: pd.DataFrame) -> np.ndarray:
    indices = np.empty_like(pop.prato_no_tipo)
    for j, (_, tipo) in enumerate(REFEICOES_PLANO):
        indices[:, j] = pratos_do_tipo(cardapio, tipo)[pop.prato_no_tipo[:, j]]
    return indices


def macros_normalizados(cardapio: pd.DataFrame) -> np.ndarray:
    macros = cardapio[list(MACROS)]
    return (macros / macros.std()).to_numpy()


def macros_habito(perfil: pd.Series, cardapio: pd.DataFrame) -> np.ndarray:
    escala = cardapio[list(MACROS)].std().to_numpy()
    habito_por_refeicao = [[perfil[f"{m}_{tipo}"] for m in MACROS] for _, tipo in REFEICOES_PLANO]
    return np.array(habito_por_refeicao, dtype=float) / escala


def distancia_nos_macros_presentes(macros: np.ndarray, alvo: np.ndarray) -> np.ndarray:
    quadrados = (macros - alvo) ** 2
    soma_na_escala_dos_quatro_macros = len(MACROS) * np.nanmean(quadrados, axis=-1)
    return np.sqrt(soma_na_escala_dos_quatro_macros)


def fora_da_faixa_kcal(kcal_por_dia: np.ndarray, perfil: pd.Series) -> np.ndarray:
    return np.maximum(np.abs(kcal_por_dia / perfil["kcal_dia"] - 1) - FAIXA_KCAL, 0)


def falta_proteina(proteina_por_dia: np.ndarray, perfil: pd.Series) -> np.ndarray:
    return np.maximum(PROTEINA_MINIMA - proteina_por_dia / perfil["proteina_dia"], 0)


def horarios(pop: Populacao) -> np.ndarray:
    horario = np.empty_like(pop.intervalo)
    for j, (_, tipo) in enumerate(REFEICOES_PLANO):
        if tipo == "cafe":
            horario[:, j] = pop.intervalo[:, j]
            continue
        anterior = horario[:, j - 1]
        intervalo = pop.intervalo[:, j]
        if tipo == "jantar":
            intervalo = np.minimum(intervalo, HORA_MAXIMA_JANTAR - anterior)
        horario[:, j] = anterior + intervalo
    return horario


def para_refeicoes(pop: Populacao, cardapio: pd.DataFrame, perfil: pd.Series) -> pd.DataFrame:
    pratos = cardapio.iloc[indices_cardapio(pop, cardapio).ravel()]
    tabela = pd.DataFrame(
        {
            "participante": perfil["participante"],
            "dia": np.tile([dia for dia, _ in REFEICOES_PLANO], len(pop)),
            "tipo": np.tile([tipo for _, tipo in REFEICOES_PLANO], len(pop)),
            "horario": horarios(pop).ravel(),
            **{coluna: pratos[coluna].to_numpy() for coluna in (*MACROS, "kcal")},
            # Um plano não tem glicose medida antes da refeição. A de jejum do CGM é a mesma
            # escala que o simulador vê no treino.
            "glicose_pre": perfil["glicose_jejum_cgm"],
            "minutos_ativos_60": pop.caminhada.ravel(),
        }
    )
    return conformar(tabela, ENTRADA_SIMULADOR)


def aleatoria(n: int, cardapio: pd.DataFrame, rng: np.random.Generator) -> Populacao:
    prato_no_tipo = np.empty((n, N_REFEICOES), dtype=np.int64)
    for j, (_, tipo) in enumerate(REFEICOES_PLANO):
        prato_no_tipo[:, j] = rng.integers(0, len(pratos_do_tipo(cardapio, tipo)), n)
    minimos, maximos = limites_intervalo()
    return Populacao(
        prato_no_tipo,
        rng.uniform(minimos, maximos, (n, N_REFEICOES)),
        rng.uniform(*LIMITES_CAMINHADA, (n, N_REFEICOES)),
    )


def habito(perfil: pd.Series, cardapio: pd.DataFrame) -> Populacao:
    macros = macros_normalizados(cardapio)
    alvo_do_dia = macros_habito(perfil, cardapio)[: len(TIPOS_REFEICAO)]
    distancia, kcal, proteina = [], [], []
    for tipo, alvo in zip(TIPOS_REFEICAO, alvo_do_dia, strict=True):
        pratos = pratos_do_tipo(cardapio, tipo)
        do_tipo = macros[pratos]
        alvo_tipo = np.median(do_tipo, axis=0) if np.isnan(alvo).all() else alvo
        distancia.append(distancia_nos_macros_presentes(do_tipo, alvo_tipo))
        kcal.append(cardapio["kcal"].to_numpy()[pratos])
        proteina.append(cardapio["proteina"].to_numpy()[pratos])

    dias = np.indices([len(d) for d in distancia]).reshape(len(TIPOS_REFEICAO), -1)

    def somar_no_dia(por_tipo: list[np.ndarray]) -> np.ndarray:
        return sum(valores[prato] for valores, prato in zip(por_tipo, dias, strict=True))

    violacao = fora_da_faixa_kcal(somar_no_dia(kcal), perfil) + falta_proteina(
        somar_no_dia(proteina), perfil
    )
    # np.lexsort ordena pela última chave, então a violação decide antes da distância.
    melhor_dia = dias[:, np.lexsort((somar_no_dia(distancia), violacao))[0]]

    horario = {tipo: perfil[f"horario_{tipo}"] for tipo in TIPOS_REFEICAO}
    intervalo_habito = {
        "cafe": horario["cafe"],
        "almoco": horario["almoco"] - horario["cafe"],
        "jantar": horario["jantar"] - horario["almoco"],
    }
    intervalo = np.array(
        [
            np.nan_to_num(intervalo_habito[tipo], nan=HORAS_DESDE_REFEICAO_ANTERIOR[tipo][0])
            for _, tipo in REFEICOES_PLANO
        ]
    )
    caminhada = [perfil[f"caminhada_{tipo}"] for _, tipo in REFEICOES_PLANO]
    minimos, maximos = limites_intervalo()
    return Populacao(
        np.tile(melhor_dia, len(DIAS_PLANO))[None, :],
        np.clip(intervalo, minimos, maximos)[None, :],
        np.clip(caminhada, *LIMITES_CAMINHADA)[None, :],
    )
