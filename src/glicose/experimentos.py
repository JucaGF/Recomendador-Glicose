import argparse
import os
import time
from functools import cache

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from glicose import ag, nsga2, otimizadores, simulador
from glicose.ag import ConfigAG, Lambdas
from glicose.esquema import DIAS_PLANO, ENTRADA_SIMULADOR, TIPOS_REFEICAO, conformar
from glicose.plano import LIMITES_CAMINHADA, Populacao
from glicose.recomendar import (
    CONFIG,
    d7_d8_do_habito,
    dias_fora_da_faixa,
    medir,
    refeicoes_dos_planos,
)
from glicose.refeicoes import ARTEFATOS

ARQUIVO_RESULTADOS = ARTEFATOS / "resultados.parquet"
ARQUIVO_CURVAS = ARTEFATOS / "curvas.parquet"

SEMENTES_VITRINE = 10
SEMENTES_DEMAIS = 10
DIA_ALVO = DIAS_PLANO[-1]
THREADS_POR_PROCESSO = 1

CAMINHADA_MAXIMA = LIMITES_CAMINHADA[1]


def menos_carbo_caminhando(perfil: pd.Series, cardapio: pd.DataFrame) -> Populacao:
    plano = otimizadores.menos_carbo(perfil, cardapio)
    plano.caminhada[:, ag.POSICOES_D9] = CAMINHADA_MAXIMA
    return plano


DETERMINISTICAS = {
    "habito": otimizadores.habito,
    "menos_carbo": otimizadores.menos_carbo,
    "menos_carbo_caminhada": menos_carbo_caminhando,
}
ESTRATEGIAS = (
    "real",
    "habito",
    "menos_carbo",
    "menos_carbo_caminhada",
    "aleatoria",
    "ag",
    "nsga2_joelho",
)
FRENTE = "nsga2_frente"

RESULTADOS = {
    "participante": "int64",
    "grupo": "str",
    "estrategia": "str",
    "semente": "Int64",
    "plano": "int64",
    "dia": "int64",
    "tipo": "str",
    "horario": "float64",
    "prato": "Int64",
    "intervalo": "float64",
    "caminhada": "float64",
    "carbo": "float64",
    "proteina": "float64",
    "gordura": "float64",
    "fibra": "float64",
    "kcal": "float64",
    "foto": "str",
    "iauc": "float64",
    "iauc_medida": "float64",
    "iauc_d9": "float64",
    "iauc_d9_sem_caminhada": "float64",
    "distancia_habito": "float64",
    "aptidao": "float64",
    "dias_fora_kcal": "int64",
    "dias_fora_proteina": "int64",
    "avaliacoes": "Int64",
    "segundos": "float64",
}

CURVAS = {
    "participante": "int64",
    "grupo": "str",
    "estrategia": "str",
    "semente": "int64",
    "geracao": "Int64",
    "avaliacoes": "int64",
    "melhor": "float64",
    "media": "float64",
}


def conformar_com_ausentes(tabela: pd.DataFrame, colunas: dict[str, str]) -> pd.DataFrame:
    return conformar(tabela.reindex(columns=list(colunas)), colunas)


@cache
def dados() -> dict:
    tf.config.threading.set_intra_op_parallelism_threads(THREADS_POR_PROCESSO)
    tf.config.threading.set_inter_op_parallelism_threads(THREADS_POR_PROCESSO)
    tabelas = {
        nome: pd.read_parquet(ARTEFATOS / f"{nome}.parquet")
        for nome in ("refeicoes", "dias", "perfis", "cardapio")
    }
    tabelas["perfis"] = tabelas["perfis"].set_index("participante", drop=False)
    tabelas["simulador"] = simulador.carregar()
    return tabelas


def elegiveis(dias: pd.DataFrame, perfis: pd.DataFrame) -> pd.Series:
    dias_plano = dias[dias["dia"].isin(DIAS_PLANO)].groupby("participante")["dia"].nunique()
    completos = dias_plano.index[dias_plano == len(DIAS_PLANO)]
    grupos = perfis.set_index("participante")["grupo"]
    return grupos[grupos.index.isin(completos)]


def vitrine(grupos: pd.Series) -> list[int]:
    return sorted(grupos.reset_index().groupby("grupo")["participante"].min())


def linhas_dos_planos(
    planos: Populacao,
    estrategia: str,
    semente: int | None,
    perfil,
    simular,
    cardapio,
    lambdas: Lambdas = CONFIG.lambdas,
    **extra,
) -> pd.DataFrame:
    refeicoes = refeicoes_dos_planos(planos, simular, perfil, cardapio)
    metricas = medir(planos, simular, perfil, cardapio, lambdas)
    return refeicoes.join(metricas, on="plano").assign(
        grupo=perfil["grupo"], estrategia=estrategia, semente=semente, **extra
    )


def d9_comparavel_com_um_plano(reais: pd.DataFrame, dia_alvo: pd.DataFrame) -> bool:
    so_tres_refeicoes = dia_alvo["n_refeicoes"].tolist() == [len(TIPOS_REFEICAO)]
    uma_limpa_de_cada_tipo = sorted(reais["tipo"]) == sorted(TIPOS_REFEICAO)
    atividade_medida = reais["minutos_ativos_60"].notna().all()
    return so_tres_refeicoes and uma_limpa_de_cada_tipo and atividade_medida


def comportamento_real(perfil, refeicoes: pd.DataFrame, dias: pd.DataFrame, simular):
    participante = perfil["participante"]
    reais = refeicoes[(refeicoes["participante"] == participante) & (refeicoes["dia"] == DIA_ALVO)]
    dia_alvo = dias[(dias["participante"] == participante) & (dias["dia"] == DIA_ALVO)]
    if not d9_comparavel_com_um_plano(reais, dia_alvo):
        return None
    reais = reais.sort_values("horario")
    entrada = conformar(reais.assign(glicose_pre=perfil["glicose_jejum_cgm"]), ENTRADA_SIMULADOR)
    iauc = np.asarray(simular(entrada))
    sem_caminhada = np.asarray(simular(entrada.assign(minutos_ativos_60=0.0)))
    dias_plano = dias[(dias["participante"] == participante) & dias["dia"].isin(DIAS_PLANO)]
    dias_plano = dias_plano.sort_values("dia")
    dias_fora_kcal, dias_fora_proteina = dias_fora_da_faixa(
        dias_plano["kcal"].to_numpy(), dias_plano["proteina"].to_numpy(), perfil
    )
    return entrada.assign(
        grupo=perfil["grupo"],
        estrategia="real",
        plano=0,
        caminhada=entrada["minutos_ativos_60"],
        foto=reais["foto"].to_numpy(),
        iauc=iauc,
        iauc_medida=reais["iauc_dexcom"].to_numpy(),
        iauc_d9=iauc.sum(),
        iauc_d9_sem_caminhada=sem_caminhada.sum(),
        dias_fora_kcal=dias_fora_kcal,
        dias_fora_proteina=dias_fora_proteina,
    )


def curva(historico: pd.DataFrame, estrategia: str, semente: int, perfil) -> pd.DataFrame:
    return historico.assign(
        participante=perfil["participante"],
        grupo=perfil["grupo"],
        estrategia=estrategia,
        semente=semente,
    )


def rodar_deterministicas(perfil, tabelas: dict, simular) -> list[pd.DataFrame]:
    partes = [
        linhas_dos_planos(
            montar(perfil, tabelas["cardapio"]), nome, None, perfil, simular, tabelas["cardapio"]
        )
        for nome, montar in DETERMINISTICAS.items()
    ]
    real = comportamento_real(perfil, tabelas["refeicoes"], tabelas["dias"], simular)
    return partes if real is None else [real, *partes]


def rodar_semente(perfil, tabelas: dict, simular, semente: int, config: ConfigAG):
    cardapio = tabelas["cardapio"]
    inicio = time.perf_counter()
    resultado_ag = ag.rodar(config, simular, perfil, cardapio, semente)
    segundos_ag = time.perf_counter() - inicio

    inicio = time.perf_counter()
    aleatoria = otimizadores.busca_aleatoria(
        resultado_ag.avaliacoes,
        simular,
        perfil,
        cardapio,
        config.lambdas,
        semente,
        config.horario_livre,
    )
    segundos_aleatoria = time.perf_counter() - inicio

    inicio = time.perf_counter()
    resultado_nsga = nsga2.rodar(config, simular, perfil, cardapio, semente)
    segundos_nsga = time.perf_counter() - inicio
    joelho = nsga2.ponto_de_joelho(resultado_nsga.objetivos)

    def linhas(planos, estrategia, avaliacoes, segundos):
        return linhas_dos_planos(
            d7_d8_do_habito(planos, perfil, cardapio),
            estrategia,
            semente,
            perfil,
            simular,
            cardapio,
            config.lambdas,
            avaliacoes=avaliacoes,
            segundos=segundos,
        )

    resultados = [
        linhas(resultado_ag.melhor, "ag", resultado_ag.avaliacoes, segundos_ag),
        linhas(aleatoria.melhor, "aleatoria", aleatoria.avaliacoes, segundos_aleatoria),
        linhas(resultado_nsga.frente, FRENTE, resultado_nsga.avaliacoes, segundos_nsga),
        linhas(
            resultado_nsga.frente[joelho], "nsga2_joelho", resultado_nsga.avaliacoes, segundos_nsga
        ),
    ]
    geracao = resultado_ag.historico["geracao"]
    historico_ag = resultado_ag.historico.assign(
        avaliacoes=(geracao + 1) * config.tamanho_populacao
    )
    curvas = [
        curva(historico_ag, "ag", semente, perfil),
        curva(aleatoria.historico, "aleatoria", semente, perfil),
    ]
    return resultados, curvas


def rodar_tarefa(participante: int, semente: int | None, config: ConfigAG = CONFIG):
    tabelas = dados()
    perfil = tabelas["perfis"].loc[participante]
    sim = tabelas["simulador"]

    def simular(refeicoes):
        return sim.prever(perfil, refeicoes)

    if semente is None:
        resultados, curvas = rodar_deterministicas(perfil, tabelas, simular), []
    else:
        resultados, curvas = rodar_semente(perfil, tabelas, simular, semente, config)
    tabela_curvas = conformar_com_ausentes(pd.concat(curvas), CURVAS) if curvas else None
    return conformar_com_ausentes(pd.concat(resultados), RESULTADOS), tabela_curvas


def tarefas(grupos: pd.Series, sementes_demais: int) -> list[tuple[int, int | None]]:
    destaque = vitrine(grupos)
    lista = []
    for participante in grupos.index:
        n_sementes = SEMENTES_VITRINE if participante in destaque else sementes_demais
        lista += [(participante, None), *((participante, s) for s in range(n_sementes))]
    return lista


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara as estratégias de D7 a D9.")
    parser.add_argument("--sementes", type=int, default=SEMENTES_DEMAIS)
    parser.add_argument("--processos", type=int, default=os.cpu_count())
    parser.add_argument("--participantes", type=int, nargs="*")
    args = parser.parse_args()

    perfis = pd.read_parquet(ARTEFATOS / "perfis.parquet")
    grupos = elegiveis(pd.read_parquet(ARTEFATOS / "dias.parquet"), perfis)
    ineligiveis = sorted(set(args.participantes or []) - set(grupos.index))
    if ineligiveis:
        parser.error(f"sem D7 a D9 válidos ou sem perfil: {ineligiveis}")
    if args.participantes:
        grupos = grupos.loc[args.participantes]
    lista = tarefas(grupos, args.sementes)

    # Com python -m este arquivo é o __main__, e os processos filhos não acham funções do
    # __main__. Importado pelo nome do pacote, rodar_tarefa chega aos filhos como referência.
    from glicose.experimentos import rodar_tarefa as tarefa

    inicio = time.perf_counter()
    saidas = joblib.Parallel(n_jobs=args.processos)(
        joblib.delayed(tarefa)(participante, semente) for participante, semente in lista
    )
    segundos = time.perf_counter() - inicio

    resultados = pd.concat([r for r, _ in saidas], ignore_index=True)
    curvas = pd.concat([c for _, c in saidas if c is not None], ignore_index=True)
    resultados.to_parquet(ARQUIVO_RESULTADOS, index=False)
    curvas.to_parquet(ARQUIVO_CURVAS, index=False)

    print(f"{len(grupos)} participantes, vitrine {vitrine(grupos)}")
    print(f"{len(lista)} tarefas em {args.processos} processos, {segundos:.0f} s")
    planos_ag = resultados.drop_duplicates(["participante", "estrategia", "semente"])
    tempo_ag = planos_ag.loc[planos_ag["estrategia"] == "ag", "segundos"]
    print(f"uma execução do AG: mediana {tempo_ag.median():.1f} s, máximo {tempo_ag.max():.1f} s")
    print(f"gravado em {ARQUIVO_RESULTADOS} ({len(resultados)} linhas) e {ARQUIVO_CURVAS}")


if __name__ == "__main__":
    main()
