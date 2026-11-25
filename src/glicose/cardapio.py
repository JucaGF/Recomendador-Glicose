import pandas as pd

from glicose.esquema import CARDAPIO, MACROS, TIPOS_REFEICAO, conformar


def montar_cardapio(refeicoes: pd.DataFrame) -> pd.DataFrame:
    macros = list(MACROS)
    com_macros = refeicoes.dropna(subset=macros)
    # O CGMacros não traz nome de prato. Sem foto, não há como mostrar o que comer.
    com_foto = com_macros[com_macros["foto"].notna()]
    nao_zerados = com_foto[com_foto[macros].ne(0).any(axis=1)]

    pratos = nao_zerados.drop_duplicates(["tipo", *macros]).rename(
        columns={"participante": "participante_origem"}
    )
    pratos["tipo"] = pd.Categorical(pratos["tipo"], categories=TIPOS_REFEICAO, ordered=True)
    pratos = pratos.sort_values(["tipo", *macros]).reset_index(drop=True)
    pratos["prato"] = pratos.index
    return conformar(pratos, CARDAPIO)
