import numpy as np
import pandas as pd
import pytest

from glicose.esquema import DIAS_HABITO, DIAS_PLANO, dados_sinteticos
from glicose.perfil import montar_perfis


def bio_do_csv(perfis: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": perfis["participante"],
            "Age": perfis["idade"],
            "Gender": perfis["sexo"],
            "BMI": perfis["imc"],
            "A1c PDL (Lab)": perfis["hba1c"],
            "Fasting GLU - PDL (Lab)": perfis["glicose_jejum_lab"],
            "Insulin ": perfis["insulina_jejum"],
        }
    )


def refeicao(**colunas) -> dict:
    return {
        "participante": 1,
        "dia": 1,
        "tipo": "almoco",
        "horario": 12.0,
        "carbo": 50.0,
        "proteina": 20.0,
        "gordura": 10.0,
        "fibra": 5.0,
        "kcal": 370.0,
        "grupo": "saudavel",
        "foto": "CGMacros/CGMacros-001/photos/a.jpg",
        "iauc_dexcom": 2000.0,
        **colunas,
    }


def test_mudar_os_dias_do_plano_nao_muda_o_perfil():
    dados = dados_sinteticos(n_participantes=6, semente=0)
    bio = bio_do_csv(dados["perfis"])
    refeicoes, dias = dados["refeicoes"], dados["dias"]
    antes = montar_perfis(refeicoes, dias, bio)

    futuro_refeicoes = refeicoes["dia"].isin(DIAS_PLANO)
    refeicoes = refeicoes.copy()
    refeicoes.loc[futuro_refeicoes, ["carbo", "proteina", "horario"]] *= 3
    refeicoes.loc[futuro_refeicoes, "iauc_dexcom"] = 99_999.0
    futuro_dias = dias["dia"].isin(DIAS_PLANO)
    dias = dias.copy()
    dias.loc[futuro_dias, ["kcal", "proteina", "glicose_media", "glicose_jejum_cgm"]] = 999.0
    depois = montar_perfis(refeicoes, dias, bio)

    assert futuro_refeicoes.any() and futuro_dias.any()
    pd.testing.assert_frame_equal(antes, depois)


def test_sensibilidade_e_a_constante_quando_a_iauc_e_proporcional_ao_carbo():
    inclinacao = 37.5
    gramas = [20.0, 45.0, 90.0]
    refeicoes = pd.DataFrame(
        [
            refeicao(dia=dia, carbo=carbo, iauc_dexcom=inclinacao * carbo)
            for dia, carbo in enumerate(gramas, 1)
        ]
    )
    dias = pd.DataFrame(
        {
            "participante": 1,
            "dia": [1, 2, 3],
            "kcal": 1800.0,
            "proteina": 80.0,
            "glicose_media": [110.0, np.nan, 120.0],
            "glicose_jejum_cgm": 95.0,
        }
    )
    bio = bio_do_csv(
        pd.DataFrame(
            {
                "participante": [1],
                "idade": [40.0],
                "sexo": ["F"],
                "imc": [25.0],
                "hba1c": [5.4],
                "glicose_jejum_lab": [90.0],
                "insulina_jejum": [8.0],
            }
        )
    )

    perfil = montar_perfis(refeicoes, dias, bio).iloc[0]

    assert perfil["sensibilidade_carbo"] == pytest.approx(inclinacao)
    assert perfil["glicose_media"] == pytest.approx(115.0)


def test_menos_de_3_refeicoes_nao_dao_sensibilidade():
    dados = dados_sinteticos(n_participantes=3, semente=1)
    refeicoes = dados["refeicoes"]
    habito_do_1 = (refeicoes["participante"] == 1) & refeicoes["dia"].isin(DIAS_HABITO)
    refeicoes = pd.concat([refeicoes[~habito_do_1], refeicoes[habito_do_1].head(2)])

    perfis = montar_perfis(refeicoes, dados["dias"], bio_do_csv(dados["perfis"]))

    sensibilidade = perfis.set_index("participante")["sensibilidade_carbo"]
    assert np.isnan(sensibilidade[1])
    assert sensibilidade[[2, 3]].notna().all()

