import numpy as np
import pytest

from glicose.ag_hiper import (
    GENES,
    ConfigBusca,
    corte_keras,
    cruzar,
    mutar,
    preparar_dados,
    rodar_buscas,
    sortear,
)
from glicose.esquema import DIAS_HABITO, DIAS_PLANO, dados_sinteticos
from glicose.simulador import ConfigRede, particao_do_estudo

N_PARTICIPANTES = 15
N_SORTEIOS = 2000
EPOCAS_TESTE = 4
BUSCA_RAPIDA = ConfigBusca(
    tamanho_populacao=2,
    n_geracoes=1,
    tamanho_torneio=2,
    processos=1,
    treino=ConfigRede(epocas=EPOCAS_TESTE),
)


@pytest.fixture(scope="module")
def sinteticos():
    return dados_sinteticos(N_PARTICIPANTES, semente=0)


@pytest.fixture(scope="module")
def dados(sinteticos):
    return preparar_dados(sinteticos["refeicoes"], sinteticos["perfis"], semente=0)


@pytest.fixture(scope="module")
def resultado(dados):
    return rodar_buscas(dados, BUSCA_RAPIDA, semente=0)


def dentro_dos_limites(config):
    for nome, gene in GENES.items():
        valor = config[nome]
        assert gene.minimo <= valor <= gene.maximo, (nome, valor)
        if gene.inteiro:
            assert isinstance(valor, int) or float(valor).is_integer(), (nome, valor)


def test_toda_configuracao_gerada_respeita_os_limites_dos_genes(resultado):
    rng = np.random.default_rng(0)
    sorteadas = sortear(N_SORTEIOS, semente=0)
    extremos = [
        {nome: getattr(gene, ponta) for nome, gene in GENES.items()}
        for ponta in ("minimo", "maximo")
    ]
    pais = sorteadas + extremos
    busca_que_sempre_muta = ConfigBusca(taxa_mutacao=1.0, sigma_relativo=1.0)
    filhos = [
        mutar(cruzar(pais[i], pais[-1 - i], 0.5, rng), busca_que_sempre_muta, rng)
        for i in range(len(pais))
    ]

    for config in [*sorteadas, *filhos, *resultado.populacoes.to_dict("records")]:
        dentro_dos_limites(config)
    for config in resultado.treinos.to_dict("records"):
        dentro_dos_limites(config)

    taxas = np.array([c["taxa_aprendizado"] for c in sorteadas])
    taxa = GENES["taxa_aprendizado"]
    assert np.median(taxas) == pytest.approx(np.sqrt(taxa.minimo * taxa.maximo), rel=0.2)


def test_as_duas_buscas_fazem_o_mesmo_numero_de_treinos(resultado, dados):
    por_busca = resultado.treinos.groupby("busca").size()
    assert set(por_busca.index) == {"genetica", "aleatoria"}
    assert por_busca["genetica"] == por_busca["aleatoria"]

    genetica = resultado.treinos[resultado.treinos["busca"] == "genetica"]
    configuracoes_unicas = genetica[list(GENES)].drop_duplicates()
    assert por_busca["genetica"] == len(configuracoes_unicas) * len(dados.divisoes)
    assert len(configuracoes_unicas) > BUSCA_RAPIDA.tamanho_populacao


def test_cada_treino_dentro_da_busca_roda_mais_de_uma_epoca(resultado):
    assert (resultado.treinos["epocas"] > 1).all()
    assert (resultado.treinos["epocas"] <= EPOCAS_TESTE).all()


def test_folds_separam_participantes_e_dias(dados, sinteticos):
    refeicoes = particao_do_estudo(sinteticos["refeicoes"], sinteticos["perfis"]).refeicoes
    participantes = refeicoes["participante"].to_numpy()
    dias = refeicoes["dia"].to_numpy()
    testados = []
    for treino, teste in dados.divisoes:
        assert set(participantes[treino]).isdisjoint(participantes[teste])
        assert np.isin(dias[treino], DIAS_HABITO).all()
        assert np.isin(dias[teste], DIAS_PLANO).all()

        corte = corte_keras(len(treino))
        ajuste, validacao = participantes[treino[:corte]], participantes[treino[corte:]]
        assert len(validacao) > 0
        assert set(ajuste).isdisjoint(validacao)
        testados.extend(participantes[teste])
    assert sorted(set(testados)) == sorted(sinteticos["perfis"]["participante"])
