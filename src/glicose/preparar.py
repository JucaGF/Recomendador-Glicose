import pandas as pd

from glicose.cardapio import montar_cardapio
from glicose.esquema import TIPOS_REFEICAO, pasta_dados
from glicose.perfil import montar_perfis
from glicose.refeicoes import ARTEFATOS, montar_tabelas


def main() -> None:
    refeicoes, dias = montar_tabelas()
    bio = pd.read_csv(pasta_dados() / "CGMacros" / "bio.csv")
    tabelas = {
        "refeicoes": refeicoes,
        "dias": dias,
        "perfis": montar_perfis(refeicoes, dias, bio),
        "cardapio": montar_cardapio(refeicoes),
    }
    ARTEFATOS.mkdir(exist_ok=True)
    for nome, tabela in tabelas.items():
        tabela.to_parquet(ARTEFATOS / f"{nome}.parquet", index=False)

    pratos = tabelas["cardapio"]["tipo"].value_counts()
    contagem = " ".join(f"{tipo} {pratos.get(tipo, 0)}" for tipo in TIPOS_REFEICAO)
    print(f"perfis {len(tabelas['perfis'])}, cardapio {contagem}")


if __name__ == "__main__":
    main()
