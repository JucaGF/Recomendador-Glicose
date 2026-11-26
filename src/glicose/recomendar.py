"""Recomenda o plano de D7 a D9 de um participante.

Roda o algoritmo genético com o simulador treinado, mantém D7 e D8 no horário e na
caminhada do hábito e grava o plano em JSON e em uma grade de fotos.

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure, SubFigure
from PIL import Image, ImageOps

from glicose import ag, simulador
from glicose.ag import ConfigAG, Lambdas, Simulador
from glicose.esquema import DIAS_PLANO, MACROS, TIPOS_REFEICAO, pasta_dados
from glicose.plano import (
    N_REFEICOES,
    Populacao,
    falta_proteina,
    fora_da_faixa_kcal,
    habito,
    indices_cardapio,
    para_refeicoes,
)
from glicose.refeicoes import ARTEFATOS

PASTA_PLANOS = ARTEFATOS / "planos"
CONFIG = ConfigAG()
POSICOES_D7_D8 = [j for j in range(N_REFEICOES) if j not in ag.POSICOES_D9]
NOMES_TIPO = {"cafe": "café", "almoco": "almoço", "jantar": "jantar"}
NOMES_GRUPO = {"saudavel": "saudável", "pre_diabetes": "pré-diabetes", "dm2": "DM2"}
LADO_MINIATURA = 400
POLEGADAS_POR_FOTO = 3.2
ALTURA_LEGENDA_POR_FOTO = 0.25
DPI_GRADE = 100
MINUTOS_POR_HORA = 60
COR_SEM_FOTO = "0.9"


def d7_d8_do_habito(planos: Populacao, perfil: pd.Series, cardapio: pd.DataFrame) -> Populacao:
    """Devolve os planos com intervalos e caminhadas de D7 e D8 iguais aos do hábito.

    Só D9 é otimizado, a única parte que a iAUC mede.
    """
    referencia = habito(perfil, cardapio)
    novo = planos.copiar()
    novo.intervalo[:, POSICOES_D7_D8] = referencia.intervalo[0, POSICOES_D7_D8]
    novo.caminhada[:, POSICOES_D7_D8] = referencia.caminhada[0, POSICOES_D7_D8]
    return novo


def sem_caminhada(planos: Populacao) -> Populacao:
    """Cópia dos planos com a caminhada zerada."""
    return Populacao(
        planos.prato_no_tipo.copy(), planos.intervalo.copy(), np.zeros_like(planos.caminhada)
    )


def dias_fora_da_faixa(
    kcal_por_dia: np.ndarray, proteina_por_dia: np.ndarray, perfil: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Conta os dias fora da faixa de kcal e abaixo da proteína mínima.

    Returns:
        Tupla `(dias_fora_kcal, dias_fora_proteina)`.
    """
    fora_kcal = fora_da_faixa_kcal(kcal_por_dia, perfil) > 0
    fora_proteina = falta_proteina(proteina_por_dia, perfil) > 0
    return fora_kcal.sum(axis=-1), fora_proteina.sum(axis=-1)


def medir(
    planos: Populacao,
    simular: Simulador,
    perfil: pd.Series,
    cardapio: pd.DataFrame,
    lambdas: Lambdas,
) -> pd.DataFrame:
    """Métricas de cada plano: iAUC de D9 (com e sem caminhada), distância do hábito,
    aptidão e dias fora da faixa.
    """
    dias_fora_kcal, dias_fora_proteina = dias_fora_da_faixa(
        ag.total_por_dia(planos, cardapio, "kcal"),
        ag.total_por_dia(planos, cardapio, "proteina"),
        perfil,
    )
    return pd.DataFrame(
        {
            "iauc_d9": ag.iauc_d9(planos, simular, perfil, cardapio),
            "iauc_d9_sem_caminhada": ag.iauc_d9(sem_caminhada(planos), simular, perfil, cardapio),
            "distancia_habito": ag.distancia_habito(planos, perfil, cardapio),
            "aptidao": ag.aptidao(planos, simular, perfil, cardapio, lambdas),
            "dias_fora_kcal": dias_fora_kcal,
            "dias_fora_proteina": dias_fora_proteina,
        }
    )


def refeicoes_dos_planos(
    planos: Populacao, simular: Simulador, perfil: pd.Series, cardapio: pd.DataFrame
) -> pd.DataFrame:
    """Uma linha por refeição de cada plano, com prato, foto e iAUC prevista."""
    tabela = para_refeicoes(planos, cardapio, perfil)
    pratos = cardapio.iloc[indices_cardapio(planos, cardapio).ravel()]
    return tabela.assign(
        plano=np.repeat(np.arange(len(planos)), N_REFEICOES),
        prato=pratos["prato"].to_numpy(),
        prato_no_tipo=planos.prato_no_tipo.ravel(),
        intervalo=planos.intervalo.ravel(),
        caminhada=planos.caminhada.ravel(),
        foto=pratos["foto"].to_numpy(),
        iauc=np.asarray(simular(tabela)),
    )


def plano_para_json(
    plano: Populacao,
    simular: Simulador,
    perfil: pd.Series,
    cardapio: pd.DataFrame,
    semente: int,
    lambdas: Lambdas,
) -> dict:
    """Converte um plano em dicionário com métricas e refeições."""
    metricas = medir(plano, simular, perfil, cardapio, lambdas).to_dict(orient="records")[0]
    colunas = [
        "dia",
        "tipo",
        "prato",
        "prato_no_tipo",
        "intervalo",
        "horario",
        "caminhada",
        *MACROS,
        "kcal",
        "foto",
        "iauc",
    ]
    refeicoes = refeicoes_dos_planos(plano, simular, perfil, cardapio)[colunas]
    return {
        "participante": int(perfil["participante"]),
        "grupo": str(perfil["grupo"]),
        "semente": semente,
        **metricas,
        "refeicoes": refeicoes.to_dict(orient="records"),
    }


def json_para_plano(dados: dict) -> Populacao:
    """Reconstrói o plano a partir do dicionário de `plano_para_json`."""
    refeicoes = pd.DataFrame(dados["refeicoes"])
    return Populacao(
        refeicoes["prato_no_tipo"].to_numpy(dtype=np.int64)[None, :],
        refeicoes["intervalo"].to_numpy(dtype=float)[None, :],
        refeicoes["caminhada"].to_numpy(dtype=float)[None, :],
    )


def ler_foto(caminho_relativo: str) -> np.ndarray | None:
    """Lê a foto do prato como miniatura RGB.

    Returns:
        A imagem, ou `None` se o arquivo não existe.
    """
    caminho = pasta_dados() / caminho_relativo
    if not caminho.is_file():
        return None
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(imagem)
        imagem.thumbnail((LADO_MINIATURA, LADO_MINIATURA))
        return np.asarray(imagem.convert("RGB"))


def hora_e_minuto(horario: float) -> str:
    """Formata horas decimais como HH:MM."""
    horas, minutos = divmod(round(horario * MINUTOS_POR_HORA), MINUTOS_POR_HORA)
    return f"{horas:02d}:{minutos:02d}"


def desenhar_grade(figura: Figure | SubFigure, refeicoes: pd.DataFrame, titulo: str) -> None:
    """Desenha a grade 3 × 3 de fotos, com horário, caminhada e macros."""
    eixos = figura.subplots(len(DIAS_PLANO), len(TIPOS_REFEICAO))
    for eixo, refeicao in zip(eixos.ravel(), refeicoes.itertuples(), strict=True):
        foto = ler_foto(refeicao.foto)
        if foto is None:
            eixo.set_facecolor(COR_SEM_FOTO)
            eixo.text(0.5, 0.5, "sem foto", ha="center", va="center", transform=eixo.transAxes)
        else:
            eixo.imshow(foto)
        eixo.set_xticks([])
        eixo.set_yticks([])
        eixo.set_title(
            f"D{refeicao.dia} {NOMES_TIPO[refeicao.tipo]} às {hora_e_minuto(refeicao.horario)}"
        )
        eixo.set_xlabel(
            f"caminhada de {refeicao.caminhada:.0f} min\n"
            f"carbo {refeicao.carbo:.0f} g, proteína {refeicao.proteina:.0f} g, "
            f"gordura {refeicao.gordura:.0f} g\n"
            f"fibra {refeicao.fibra:.0f} g, {refeicao.kcal:.0f} kcal"
        )
    figura.suptitle(titulo)


def gravar_grade(refeicoes: pd.DataFrame, titulo: str, caminho: Path) -> None:
    """Grava a grade de refeições em um PNG."""
    figura = Figure(
        figsize=(
            POLEGADAS_POR_FOTO * len(TIPOS_REFEICAO),
            POLEGADAS_POR_FOTO * len(DIAS_PLANO) * (1 + ALTURA_LEGENDA_POR_FOTO),
        ),
        layout="constrained",
    )
    desenhar_grade(figura, refeicoes, titulo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=DPI_GRADE)


def recomendar(
    perfil: pd.Series,
    cardapio: pd.DataFrame,
    simular: Simulador,
    semente: int,
    pasta: Path = PASTA_PLANOS,
    config: ConfigAG = CONFIG,
) -> dict:
    """Roda o algoritmo genético para uma pessoa e grava o plano.

    Args:
        perfil: Linha da tabela `perfis`.
        cardapio: Tabela `cardapio`.
        simular: Função que prevê a iAUC de cada refeição.
        semente: Semente do algoritmo.
        pasta: Onde gravar o JSON e o PNG.
        config: Hiperparâmetros do AG.

    Returns:
        O plano como dicionário, o mesmo gravado em JSON.
    """
    inicio = time.perf_counter()
    resultado = ag.rodar(config, simular, perfil, cardapio, semente)
    segundos = time.perf_counter() - inicio
    recomendado = d7_d8_do_habito(resultado.melhor, perfil, cardapio)
    plano = plano_para_json(recomendado, simular, perfil, cardapio, semente, config.lambdas)
    plano["avaliacoes"] = resultado.avaliacoes
    plano["segundos"] = segundos
    plano["iauc_d9_habito"] = float(
        ag.iauc_d9(habito(perfil, cardapio), simular, perfil, cardapio)[0]
    )

    nome = f"{plano['participante']:03d}"
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / f"{nome}.json").write_text(json.dumps(plano, ensure_ascii=False, indent=2))
    titulo = (
        f"Plano de D7 a D9 do participante {nome} ({NOMES_GRUPO[plano['grupo']]})\n"
        f"iAUC prevista de D9: {plano['iauc_d9']:.0f} com o AG e "
        f"{plano['iauc_d9_habito']:.0f} no hábito (mg/dL·min)"
    )
    gravar_grade(pd.DataFrame(plano["refeicoes"]), titulo, pasta / f"{nome}.png")
    return plano


def imprimir(plano: dict, pasta: Path) -> None:
    """Imprime o resumo do plano."""
    nome = f"{plano['participante']:03d}"
    print(f"participante {nome} ({plano['grupo']}), semente {plano['semente']}")
    print(f"AG: {plano['avaliacoes']} avaliacoes em {plano['segundos']:.1f} s")
    print("iAUC prevista de D9 em mg/dL·min")
    print(f"  habito: {plano['iauc_d9_habito']:.1f}")
    print(f"  AG: {plano['iauc_d9']:.1f}")
    print(f"  AG sem a caminhada: {plano['iauc_d9_sem_caminhada']:.1f}")
    print(f"aptidao do AG: {plano['aptidao']:.6f}")
    print(f"distancia do habito: {plano['distancia_habito']:.2f} desvios-padrao")
    print(
        f"dias fora da faixa: kcal {plano['dias_fora_kcal']}, "
        f"proteina {plano['dias_fora_proteina']}"
    )
    colunas = ["dia", "tipo", "horario", "prato", "caminhada", "carbo", "proteina", "kcal", "iauc"]
    print(pd.DataFrame(plano["refeicoes"])[colunas].round(1).to_string(index=False))
    print(f"plano gravado em {pasta / f'{nome}.json'} e {pasta / f'{nome}.png'}")


def main() -> None:
    """Lê os artefatos, recomenda o plano do participante pedido e imprime o resultado."""
    parser = argparse.ArgumentParser(description="Recomenda refeições e caminhadas de D7 a D9.")
    parser.add_argument("--participante", type=int, required=True)
    parser.add_argument("--semente", type=int, default=0)
    args = parser.parse_args()

    perfis = pd.read_parquet(ARTEFATOS / "perfis.parquet").set_index("participante", drop=False)
    if args.participante not in perfis.index:
        parser.error(f"participante {args.participante} não tem perfil de D1 a D6")
    perfil = perfis.loc[args.participante]
    cardapio = pd.read_parquet(ARTEFATOS / "cardapio.parquet")
    sim = simulador.carregar()
    plano = recomendar(perfil, cardapio, lambda r: sim.prever(perfil, r), args.semente)
    imprimir(plano, PASTA_PLANOS)


if __name__ == "__main__":
    main()
