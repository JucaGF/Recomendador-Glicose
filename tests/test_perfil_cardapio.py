"""Testes dos perfis (`perfil`) e do cardápio (`cardapio`).

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

import numpy as np
import pandas as pd
import pytest

from glicose.cardapio import montar_cardapio
from glicose.esquema import DIAS_HABITO, DIAS_PLANO, MACROS, dados_sinteticos
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
    linha = {
        "participante": 1,
        "dia": 1,
        "tipo": "almoco",
        "horario": 12.0,
        "carbo": 50.0,
        "proteina": 20.0,
        "gordura": 10.0,
        "fibra": 5.0,
        "grupo": "saudavel",
        "foto": "CGMacros/CGMacros-001/photos/a.jpg",
        "iauc_dexcom": 2000.0,
        "minutos_ativos_60": 10.0,
        **colunas,
    }
    kcal_dos_macros = 4 * linha["carbo"] + 4 * linha["proteina"] + 9 * linha["gordura"]
    return {"kcal": kcal_dos_macros, **linha}


def perfis_das_refeicoes(refeicoes: pd.DataFrame) -> pd.DataFrame:
    participantes = refeicoes["participante"].unique()
    dias = refeicoes[["participante", "dia"]].drop_duplicates()
    dias = dias.assign(kcal=1800.0, proteina=80.0, glicose_media=110.0, glicose_jejum_cgm=95.0)
    bio = bio_do_csv(
        pd.DataFrame(
            {
                "participante": participantes,
                "idade": 40.0,
                "sexo": "F",
                "imc": 25.0,
                "hba1c": 5.4,
                "glicose_jejum_lab": 90.0,
                "insulina_jejum": 8.0,
            }
        )
    )
    return montar_perfis(refeicoes, dias, bio).set_index("participante")


def test_mudar_os_dias_do_plano_nao_muda_o_perfil():
    dados = dados_sinteticos(n_participantes=6, semente=0)
    bio = bio_do_csv(dados["perfis"])
    refeicoes, dias = dados["refeicoes"], dados["dias"]
    antes = montar_perfis(refeicoes, dias, bio)

    futuro_refeicoes = refeicoes["dia"].isin(DIAS_PLANO)
    refeicoes = refeicoes.copy()
    refeicoes.loc[futuro_refeicoes, ["carbo", "proteina", "horario", "minutos_ativos_60"]] *= 3
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


def test_caminhada_habitual_e_a_media_dos_dias_1_a_6():
    refeicoes = pd.DataFrame(
        [
            refeicao(dia=1, carbo=20.0, minutos_ativos_60=10.0),
            refeicao(dia=2, carbo=45.0, minutos_ativos_60=20.0),
            refeicao(dia=3, carbo=90.0, minutos_ativos_60=np.nan),
            refeicao(dia=7, carbo=60.0, minutos_ativos_60=60.0),
        ]
    )

    perfil = perfis_das_refeicoes(refeicoes).loc[1]

    assert perfil["caminhada_almoco"] == pytest.approx(15.0)


def test_sem_caminhada_medida_recebe_a_mediana_do_grupo_ou_a_de_todos():
    refeicoes = pd.DataFrame(
        [
            refeicao(participante=1, minutos_ativos_60=10.0),
            refeicao(participante=2, minutos_ativos_60=30.0),
            refeicao(participante=3, minutos_ativos_60=np.nan),
            refeicao(participante=4, grupo="dm2", minutos_ativos_60=50.0),
            refeicao(participante=5, grupo="pre_diabetes", minutos_ativos_60=np.nan),
        ]
    )

    caminhada = perfis_das_refeicoes(refeicoes)["caminhada_almoco"]

    assert caminhada[3] == pytest.approx(20.0)
    assert caminhada[5] == pytest.approx(30.0)
    assert caminhada.notna().all()


def test_quem_nao_jantou_tem_a_caminhada_do_jantar_do_grupo():
    refeicoes = pd.DataFrame(
        [
            refeicao(participante=1, tipo="jantar", minutos_ativos_60=4.0),
            refeicao(participante=2, tipo="jantar", minutos_ativos_60=8.0),
            refeicao(participante=3, tipo="almoco", minutos_ativos_60=40.0),
        ]
    )

    perfil = perfis_das_refeicoes(refeicoes).loc[3]

    assert perfil["caminhada_jantar"] == pytest.approx(6.0)
    assert np.isnan(perfil["carbo_jantar"])


def test_fibra_acima_do_carbo_fica_fora_da_media_da_fibra_mas_o_carbo_conta():
    refeicoes = pd.DataFrame(
        [
            refeicao(dia=1, tipo="jantar", carbo=50.0, fibra=5.0),
            refeicao(dia=2, tipo="jantar", carbo=40.0, fibra=570.0),
        ]
    )

    perfil = perfis_das_refeicoes(refeicoes).loc[1]

    assert perfil["fibra_jantar"] == pytest.approx(5.0)
    assert perfil["carbo_jantar"] == pytest.approx(45.0)


def test_menos_de_3_refeicoes_nao_dao_sensibilidade():
    dados = dados_sinteticos(n_participantes=3, semente=1)
    refeicoes = dados["refeicoes"]
    habito_do_1 = (refeicoes["participante"] == 1) & refeicoes["dia"].isin(DIAS_HABITO)
    refeicoes = pd.concat([refeicoes[~habito_do_1], refeicoes[habito_do_1].head(2)])

    perfis = montar_perfis(refeicoes, dados["dias"], bio_do_csv(dados["perfis"]))

    sensibilidade = perfis.set_index("participante")["sensibilidade_carbo"]
    assert np.isnan(sensibilidade[1])
    assert sensibilidade[[2, 3]].notna().all()


def test_jantar_zerado_fica_fora_do_cardapio():
    refeicoes = pd.DataFrame(
        [
            refeicao(tipo="jantar", carbo=0.0, proteina=0.0, gordura=0.0, fibra=0.0, kcal=0.0),
            refeicao(tipo="jantar", carbo=30.0),
        ]
    )

    cardapio = montar_cardapio(refeicoes)

    assert cardapio["carbo"].tolist() == [30.0]


def test_prato_repetido_vira_um_prato_com_foto():
    refeicoes = pd.DataFrame(
        [
            refeicao(participante=2, tipo="jantar", foto=None),
            refeicao(participante=3, tipo="jantar", foto="CGMacros/CGMacros-003/photos/b.jpg"),
            refeicao(participante=4, tipo="jantar", foto="CGMacros/CGMacros-004/photos/c.jpg"),
            refeicao(participante=5, tipo="jantar", carbo=10.0, foto=None),
            refeicao(participante=1, tipo="cafe", carbo=70.0),
        ]
    )

    cardapio = montar_cardapio(refeicoes)

    assert cardapio["tipo"].tolist() == ["cafe", "jantar"]
    assert cardapio["prato"].tolist() == [0, 1]
    jantar = cardapio.iloc[1]
    assert jantar["foto"] == "CGMacros/CGMacros-003/photos/b.jpg"
    assert jantar["participante_origem"] == 3


def test_quem_nao_jantou_nos_dias_de_habito_tem_perfil_sem_habito_de_jantar():
    dados = dados_sinteticos(n_participantes=3, semente=2)
    refeicoes = dados["refeicoes"]
    jantar_do_1 = (
        (refeicoes["participante"] == 1)
        & refeicoes["dia"].isin(DIAS_HABITO)
        & (refeicoes["tipo"] == "jantar")
    )

    perfis = montar_perfis(refeicoes[~jantar_do_1], dados["dias"], bio_do_csv(dados["perfis"]))

    perfil = perfis.set_index("participante").loc[1]
    habito_jantar = [f"{coluna}_jantar" for coluna in (*MACROS, "horario")]
    assert jantar_do_1.any()
    assert perfil[habito_jantar].isna().all()
    outras = perfil.drop([*habito_jantar, "grupo", "sexo"]).astype(float)
    assert np.isfinite(outras).all()


def test_kcal_que_nao_bate_com_os_macros_fica_fora_do_cardapio():
    refeicoes = pd.DataFrame(
        [
            refeicao(tipo="jantar", carbo=0.0, proteina=10.0, gordura=0.0, kcal=420.0),
            refeicao(tipo="jantar", carbo=761.0, proteina=20.0, gordura=10.0, kcal=1030.0),
            refeicao(tipo="jantar", carbo=60.0, kcal=500.0),
        ]
    )

    cardapio = montar_cardapio(refeicoes)

    assert cardapio["carbo"].tolist() == [60.0]


def test_fibra_acima_do_carbo_fica_fora_do_cardapio():
    refeicoes = pd.DataFrame(
        [
            refeicao(tipo="jantar", carbo=40.0, fibra=2199.0),
            refeicao(tipo="jantar", carbo=30.0, fibra=30.0),
        ]
    )

    cardapio = montar_cardapio(refeicoes)

    assert cardapio["carbo"].tolist() == [30.0]


def test_prato_quase_vazio_fica_fora_do_cardapio():
    refeicoes = pd.DataFrame(
        [
            refeicao(tipo="jantar", carbo=0.0, proteina=0.0, gordura=0.0, fibra=1.0, kcal=10.0),
            refeicao(tipo="jantar", carbo=10.0, proteina=0.0, gordura=0.0),
            refeicao(tipo="jantar", carbo=30.0),
        ]
    )

    cardapio = montar_cardapio(refeicoes)

    assert cardapio["carbo"].tolist() == [30.0]


def test_gemeo_de_digitacao_vira_o_prato_mais_registrado():
    foto_mais_vista = "CGMacros/CGMacros-002/photos/b.jpg"
    refeicoes = pd.DataFrame(
        [
            refeicao(participante=1, fibra=6.0, foto="CGMacros/CGMacros-001/photos/a.jpg"),
            *[refeicao(participante=2, foto=foto_mais_vista) for _ in range(3)],
            refeicao(participante=3, carbo=53.0),
            refeicao(participante=4, tipo="jantar", fibra=6.0),
        ]
    )

    cardapio = montar_cardapio(refeicoes)

    almoco = cardapio[cardapio["tipo"] == "almoco"]
    assert almoco["carbo"].tolist() == [50.0, 53.0]
    assert almoco.iloc[0]["fibra"] == 5.0
    assert almoco.iloc[0]["foto"] == foto_mais_vista
    assert almoco.iloc[0]["participante_origem"] == 2
    assert (cardapio["tipo"] == "jantar").sum() == 1


def test_mesmos_macros_com_duas_kcal_ficam_com_a_mais_registrada():
    refeicoes = pd.DataFrame([refeicao(kcal=380.0), refeicao(kcal=370.0), refeicao(kcal=370.0)])

    cardapio = montar_cardapio(refeicoes)

    assert cardapio["kcal"].tolist() == [370.0]
