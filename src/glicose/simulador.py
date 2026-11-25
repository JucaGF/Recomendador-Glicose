from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from glicose.baselines_modelo import ALVO, BASELINES, Particao
from glicose.esquema import DIAS_HABITO, DIAS_PLANO, TIPOS_REFEICAO
from glicose.refeicoes import ARTEFATOS

ATRIBUTOS_REFEICAO = (
    "horario",
    "carbo",
    "proteina",
    "gordura",
    "fibra",
    "kcal",
    "glicose_pre",
    "minutos_ativos_60",
)
ATRIBUTOS_PERFIL = (
    "idade",
    "imc",
    "hba1c",
    "glicose_jejum_lab",
    "insulina_jejum",
    "glicose_media",
    "glicose_jejum_cgm",
    "sensibilidade_carbo",
)
ATRIBUTOS = (
    *ATRIBUTOS_REFEICAO,
    *ATRIBUTOS_PERFIL,
    "atividade_ausente",
    "sexo_masculino",
    *(f"tipo_{tipo}" for tipo in TIPOS_REFEICAO),
)

DIAS_ESTUDO = (*DIAS_HABITO, *DIAS_PLANO)
CONJUNTOS = {"d7_d9": DIAS_PLANO, "d1_d6": DIAS_HABITO}
N_FOLDS = 5
SEMENTE = 0
SEMENTES = (0, 1, 2)
FRACAO_VALIDACAO = 0.2
IAUC_MINIMA = 0.0

ARQUIVO_MODELO = "simulador.keras"
ARQUIVO_PREPARO = "simulador_preparo.joblib"


@dataclass(frozen=True)
class ConfigRede:
    n_camadas: int = 2
    n_neuronios: int = 64
    taxa_aprendizado: float = 3e-3
    dropout: float = 0.2
    epocas: int = 300
    tamanho_lote: int = 32
    paciencia: int = 10


@dataclass(frozen=True)
class Alvo:
    ida: Callable[[np.ndarray], np.ndarray]
    volta: Callable[[np.ndarray], np.ndarray]


ALVOS = {
    "iauc": Alvo(ida=np.asarray, volta=np.asarray),
    "log1p": Alvo(ida=np.log1p, volta=np.expm1),
}
MODELOS = (*(f"rede_{alvo}" for alvo in ALVOS), *BASELINES)


def montar_atributos(refeicoes: pd.DataFrame, perfis: pd.DataFrame) -> pd.DataFrame:
    perfil = perfis.set_index("participante").loc[refeicoes["participante"]]
    atributos = pd.DataFrame(
        {
            **{coluna: refeicoes[coluna].to_numpy() for coluna in ATRIBUTOS_REFEICAO},
            **{coluna: perfil[coluna].to_numpy() for coluna in ATRIBUTOS_PERFIL},
            "atividade_ausente": refeicoes["minutos_ativos_60"].isna().to_numpy(),
            "sexo_masculino": (perfil["sexo"] == "M").to_numpy(),
            **{f"tipo_{tipo}": (refeicoes["tipo"] == tipo).to_numpy() for tipo in TIPOS_REFEICAO},
        },
        index=refeicoes.index,
    ).astype("float64")
    # A fibra faz parte do carbo total. Acima dele o registro está errado e vira valor ausente.
    atributos["fibra"] = atributos["fibra"].where(atributos["fibra"] <= atributos["carbo"])
    return atributos


def construir_modelo(
    n_entradas: int, n_camadas: int, n_neuronios: int, taxa_aprendizado: float, dropout: float
) -> keras.Model:
    modelo = keras.Sequential([keras.Input(shape=(n_entradas,))])
    for _ in range(n_camadas):
        modelo.add(keras.layers.Dense(n_neuronios, activation="relu"))
        modelo.add(keras.layers.Dropout(dropout))
    modelo.add(keras.layers.Dense(1))
    # O notebook da aula usa mse. A nota aqui é o MAE, e a perda MAE errou menos que a mse em
    # todas as sementes testadas.
    modelo.compile(loss="mae", optimizer=keras.optimizers.SGD(taxa_aprendizado))
    return modelo


@dataclass(frozen=True)
class Simulador:
    modelo: keras.Model
    preparo: Pipeline
    escala_alvo: StandardScaler
    alvo: str
    iauc_maxima: float

    def entradas(self, atributos: pd.DataFrame) -> np.ndarray:
        return self.preparo.transform(atributos).astype("float32")

    def para_iauc(self, saida: np.ndarray) -> np.ndarray:
        escalado = self.escala_alvo.inverse_transform(np.asarray(saida).reshape(-1, 1))
        return np.clip(ALVOS[self.alvo].volta(escalado.ravel()), IAUC_MINIMA, self.iauc_maxima)

    def prever_atributos(self, atributos: pd.DataFrame) -> np.ndarray:
        return self.para_iauc(self.modelo.predict_on_batch(self.entradas(atributos)))

    def prever(self, perfil: pd.Series, refeicoes: pd.DataFrame) -> np.ndarray:
        return self.prever_atributos(montar_atributos(refeicoes, pd.DataFrame([perfil])))


def separar_validacao(treino: Particao, semente: int) -> tuple[np.ndarray, np.ndarray]:
    divisor = GroupShuffleSplit(n_splits=1, test_size=FRACAO_VALIDACAO, random_state=semente)
    return next(divisor.split(treino.atributos, groups=treino.refeicoes["participante"]))


def treinar(
    treino: Particao, alvo: str, config: ConfigRede, semente: int
) -> tuple[Simulador, keras.callbacks.History]:
    keras.utils.set_random_seed(semente)
    preparo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    preparo.fit(treino.atributos)
    alvo_transformado = ALVOS[alvo].ida(treino.iauc).reshape(-1, 1)
    escala_alvo = StandardScaler().fit(alvo_transformado)
    modelo = construir_modelo(
        len(ATRIBUTOS),
        config.n_camadas,
        config.n_neuronios,
        config.taxa_aprendizado,
        config.dropout,
    )
    simulador = Simulador(modelo, preparo, escala_alvo, alvo, float(treino.iauc.max()))
    x = simulador.entradas(treino.atributos)
    y = escala_alvo.transform(alvo_transformado)
    ajuste, validacao = separar_validacao(treino, semente)
    historico = modelo.fit(
        x[ajuste],
        y[ajuste],
        validation_data=(x[validacao], y[validacao]),
        epochs=config.epocas,
        batch_size=config.tamanho_lote,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=config.paciencia, restore_best_weights=True)
        ],
        verbose=0,
    )
    return simulador, historico


def prever_rede(
    treino: Particao, teste: Particao, alvo: str, config: ConfigRede, semente: int
) -> np.ndarray:
    simulador, _ = treinar(treino, alvo, config, semente)
    return simulador.prever_atributos(teste.atributos)


def particao_do_estudo(refeicoes: pd.DataFrame, perfis: pd.DataFrame) -> Particao:
    estudo = refeicoes[refeicoes["dia"].isin(DIAS_ESTUDO)]
    return Particao(estudo, montar_atributos(estudo, perfis))


def dividir(refeicoes: pd.DataFrame, perfis: pd.DataFrame) -> Iterator[tuple[Particao, Particao]]:
    estudo = particao_do_estudo(refeicoes, perfis)
    habito = estudo.refeicoes["dia"].isin(DIAS_HABITO).to_numpy()
    grupos = estudo.refeicoes["participante"]
    for treino, teste in GroupKFold(N_FOLDS).split(estudo.atributos, groups=grupos):
        yield estudo.recortar(treino[habito[treino]]), estudo.recortar(teste)


def validar(
    refeicoes: pd.DataFrame,
    perfis: pd.DataFrame,
    config: ConfigRede = ConfigRede(),
    semente: int = SEMENTE,
) -> pd.DataFrame:
    redes = {
        f"rede_{alvo}": partial(prever_rede, alvo=alvo, config=config, semente=semente)
        for alvo in ALVOS
    }
    modelos = {**redes, **BASELINES}
    partes = []
    for fold, (treino, teste) in enumerate(dividir(refeicoes, perfis)):
        previsto = {nome: modelo(treino, teste) for nome, modelo in modelos.items()}
        partes.append(teste.refeicoes.assign(fold=fold, **previsto))
    return pd.concat(partes, ignore_index=True)


def resumir(previsoes: pd.DataFrame) -> pd.DataFrame:
    linhas = {}
    for nome in MODELOS:
        linha = {}
        for conjunto, dias in CONJUNTOS.items():
            parte = previsoes[previsoes["dia"].isin(dias)]
            erro = (parte[ALVO] - parte[nome]).abs()
            linha[f"mae_{conjunto}"] = erro.mean()
            linha[f"desvio_folds_{conjunto}"] = erro.groupby(parte["fold"]).mean().std()
            linha[f"pearson_{conjunto}"] = parte[ALVO].corr(parte[nome])
        linhas[nome] = linha
    return pd.DataFrame.from_dict(linhas, orient="index")


def mae_por(previsoes: pd.DataFrame, modelo: str, coluna: str) -> pd.Series:
    plano = previsoes[previsoes["dia"].isin(DIAS_PLANO)]
    return (plano[ALVO] - plano[modelo]).abs().groupby(plano[coluna]).mean()


def media_das_sementes(partes: list) -> pd.DataFrame | pd.Series:
    return pd.concat(partes).groupby(level=0, sort=False).mean()


def escolher_alvo(tabela: pd.DataFrame) -> str:
    return min(ALVOS, key=lambda alvo: tabela.loc[f"rede_{alvo}", "mae_d7_d9"])


def treinar_final(
    refeicoes: pd.DataFrame,
    perfis: pd.DataFrame,
    alvo: str,
    config: ConfigRede = ConfigRede(),
    semente: int = SEMENTE,
) -> Simulador:
    estudo = particao_do_estudo(refeicoes, perfis)
    habito = estudo.refeicoes["dia"].isin(DIAS_HABITO).to_numpy()
    simulador, _ = treinar(estudo.recortar(habito), alvo, config, semente)
    return simulador


def teste_d7_d9(
    simulador: Simulador, refeicoes: pd.DataFrame, perfis: pd.DataFrame
) -> pd.DataFrame:
    estudo = particao_do_estudo(refeicoes, perfis)
    plano = estudo.recortar(estudo.refeicoes["dia"].isin(DIAS_PLANO).to_numpy())
    return plano.refeicoes.assign(previsto=simulador.prever_atributos(plano.atributos))


def salvar(simulador: Simulador, pasta: Path = ARTEFATOS) -> None:
    pasta.mkdir(exist_ok=True)
    simulador.modelo.save(pasta / ARQUIVO_MODELO)
    preparo = {
        "preparo": simulador.preparo,
        "escala_alvo": simulador.escala_alvo,
        "alvo": simulador.alvo,
        "iauc_maxima": simulador.iauc_maxima,
    }
    joblib.dump(preparo, pasta / ARQUIVO_PREPARO)


def carregar(pasta: Path = ARTEFATOS) -> Simulador:
    preparo = joblib.load(pasta / ARQUIVO_PREPARO)
    return Simulador(modelo=keras.models.load_model(pasta / ARQUIVO_MODELO), **preparo)


def main() -> None:
    refeicoes = pd.read_parquet(ARTEFATOS / "refeicoes.parquet")
    perfis = pd.read_parquet(ARTEFATOS / "perfis.parquet")

    validacoes = [validar(refeicoes, perfis, semente=semente) for semente in SEMENTES]
    tabela = media_das_sementes([resumir(previsoes) for previsoes in validacoes])
    alvo = escolher_alvo(tabela)
    print(f"GroupKFold({N_FOLDS}) por participante, média de {len(SEMENTES)} sementes")
    print("MAE em mg/dL·min")
    print(tabela.to_string(float_format="%.2f"))
    print(f"\nalvo escolhido: {alvo}")
    for coluna in ("tipo", "grupo"):
        print(f"\nMAE D7 a D9 da rede_{alvo} por {coluna}")
        mae_coluna = [mae_por(previsoes, f"rede_{alvo}", coluna) for previsoes in validacoes]
        print(media_das_sementes(mae_coluna).round(1).to_string())

    simulador = treinar_final(refeicoes, perfis, alvo)
    salvar(simulador)
    teste = teste_d7_d9(simulador, refeicoes, perfis)
    mae_teste = (teste[ALVO] - teste["previsto"]).abs().mean()
    print(f"\nmodelo final, MAE nas refeições de D7 a D9 dos mesmos participantes: {mae_teste:.1f}")
    print(f"gravado em {ARTEFATOS / ARQUIVO_MODELO} e {ARTEFATOS / ARQUIVO_PREPARO}")


if __name__ == "__main__":
    main()
