from dataclasses import dataclass

import numpy as np
import pandas as pd

from glicose.esquema import DIAS_PLANO, ENTRADA_SIMULADOR, MACROS, TIPOS_REFEICAO, conformar

REFEICOES_PLANO = tuple((dia, tipo) for dia in DIAS_PLANO for tipo in TIPOS_REFEICAO)
N_REFEICOES = len(REFEICOES_PLANO)

# O gene do café é a hora do dia, e os do almoço e do jantar são horas desde a refeição anterior.
# Os limites vão do percentil 5 ao 95 das refeições reais do CGMacros, com o piso de 3 horas do
# protocolo. A caminhada vai de 0 a 30 minutos, como no plano do projeto.
LIMITES_HORARIO = {
    "cafe": (6.0, 10.5),
    "almoco": (3.0, 6.5),
    "jantar": (3.0, 8.5),
}
LIMITES_CAMINHADA = (0.0, 30.0)


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
    minimos = np.array([LIMITES_HORARIO[tipo][0] for _, tipo in REFEICOES_PLANO])
    maximos = np.array([LIMITES_HORARIO[tipo][1] for _, tipo in REFEICOES_PLANO])
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


def horarios(pop: Populacao) -> np.ndarray:
    horario = np.empty_like(pop.intervalo)
    for j, (_, tipo) in enumerate(REFEICOES_PLANO):
        if tipo == "cafe":
            horario[:, j] = pop.intervalo[:, j]
        else:
            horario[:, j] = horario[:, j - 1] + pop.intervalo[:, j]
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
    alvo = macros_habito(perfil, cardapio)
    horario = {tipo: perfil[f"horario_{tipo}"] for tipo in TIPOS_REFEICAO}
    intervalo_habito = {
        "cafe": horario["cafe"],
        "almoco": horario["almoco"] - horario["cafe"],
        "jantar": horario["jantar"] - horario["almoco"],
    }
    prato_no_tipo = np.empty((1, N_REFEICOES), dtype=np.int64)
    intervalo = np.empty((1, N_REFEICOES))
    for j, (_, tipo) in enumerate(REFEICOES_PLANO):
        distancias = np.linalg.norm(macros[pratos_do_tipo(cardapio, tipo)] - alvo[j], axis=1)
        prato_no_tipo[0, j] = np.argmin(distancias)
        intervalo[0, j] = intervalo_habito[tipo]
    minimos, maximos = limites_intervalo()
    return Populacao(
        prato_no_tipo, np.clip(intervalo, minimos, maximos), np.zeros((1, N_REFEICOES))
    )
