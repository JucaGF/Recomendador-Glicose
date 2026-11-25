import numpy as np
import pytest

from glicose import plano
from glicose.esquema import ENTRADA_SIMULADOR, TIPOS_REFEICAO, dados_sinteticos


def test_para_refeicoes_segue_o_contrato_do_simulador():
    dados = dados_sinteticos(1, semente=2)
    perfil = dados["perfis"].iloc[0]
    pop = plano.aleatoria(4, dados["cardapio"], np.random.default_rng(0))

    refeicoes = plano.para_refeicoes(pop, dados["cardapio"], perfil)

    assert refeicoes.dtypes.astype(str).to_dict() == ENTRADA_SIMULADOR
    assert len(refeicoes) == 4 * plano.N_REFEICOES
    assert (refeicoes["glicose_pre"] == perfil["glicose_jejum_cgm"]).all()
    assert refeicoes["minutos_ativos_60"].tolist() == pop.caminhada.ravel().tolist()
    assert set(refeicoes["dia"]) == {7, 8, 9}


def test_habito_representa_os_horarios_do_participante():
    dados = dados_sinteticos(5, semente=3)
    for _, perfil in dados["perfis"].iterrows():
        horarios = plano.horarios(plano.habito(perfil, dados["cardapio"]))[0, :3]
        esperado = [perfil[f"horario_{tipo}"] for tipo in TIPOS_REFEICAO]
        assert horarios == pytest.approx(esperado)
