from collections.abc import Iterator
from dataclasses import dataclass, replace
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
COLUNA_CARBO = ATRIBUTOS.index("carbo")
COLUNA_CAMINHADA = ATRIBUTOS.index("minutos_ativos_60")
FORA_DO_CONTEXTO = ("carbo", "minutos_ativos_60", "kcal")
COLUNAS_CONTEXTO = [i for i, nome in enumerate(ATRIBUTOS) if nome not in FORA_DO_CONTEXTO]

DIAS_ESTUDO = (*DIAS_HABITO, *DIAS_PLANO)
CONJUNTOS = {"d7_d9": DIAS_PLANO, "d1_d6": DIAS_HABITO}
MODELOS = ("rede", *BASELINES)
N_FOLDS = 5
SEMENTE = 0
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
    n_redes: int = 5


def montar_atributos(refeicoes: pd.DataFrame, perfis: pd.DataFrame) -> pd.DataFrame:
    perfil = perfis.set_index("participante").loc[refeicoes["participante"]]
    return pd.DataFrame(
        {
            **{coluna: refeicoes[coluna].to_numpy() for coluna in ATRIBUTOS_REFEICAO},
            **{coluna: perfil[coluna].to_numpy() for coluna in ATRIBUTOS_PERFIL},
            "atividade_ausente": refeicoes["minutos_ativos_60"].isna().to_numpy(),
            "sexo_masculino": (perfil["sexo"] == "M").to_numpy(),
            **{f"tipo_{tipo}": (refeicoes["tipo"] == tipo).to_numpy() for tipo in TIPOS_REFEICAO},
        },
        index=refeicoes.index,
    ).astype("float64")


def construir_modelo(config: ConfigRede, nome: str = "rede") -> keras.Model:
    entrada = keras.Input(shape=(len(ATRIBUTOS),))
    contexto = keras.ops.take(entrada, COLUNAS_CONTEXTO, axis=1)
    carbo = keras.ops.take(entrada, [COLUNA_CARBO], axis=1)
    caminhada = keras.ops.take(entrada, [COLUNA_CAMINHADA], axis=1)
    oculta = contexto
    for _ in range(config.n_camadas):
        oculta = keras.layers.Dense(config.n_neuronios, activation="relu")(oculta)
        # Cada treino vê menos de mil refeições. Sem Dropout a rede decoraria essas refeições.
        oculta = keras.layers.Dropout(config.dropout)(oculta)
    base = keras.layers.Dense(1)(oculta)
    sensibilidade = keras.layers.Dense(1, activation="softplus")(oculta)
    beneficio_caminhada = keras.layers.Dense(1, activation="softplus")(oculta)
    saida = base + sensibilidade * carbo - beneficio_caminhada * caminhada
    modelo = keras.Model(entrada, saida, name=nome)
    modelo.compile(loss="mae", optimizer=keras.optimizers.SGD(config.taxa_aprendizado))
    return modelo


def media_das_redes(redes: list[keras.Model]) -> keras.Model:
    entrada = keras.Input(shape=(len(ATRIBUTOS),))
    return keras.Model(entrada, keras.layers.Average()([rede(entrada) for rede in redes]))


@dataclass(frozen=True)
class Simulador:
    modelo: keras.Model
    preparo: Pipeline
    escala_alvo: StandardScaler
    iauc_maxima: float

    def entradas(self, atributos: pd.DataFrame) -> np.ndarray:
        return self.preparo.transform(atributos).astype("float32")

    def para_iauc(self, saida: np.ndarray) -> np.ndarray:
        iauc = self.escala_alvo.inverse_transform(np.asarray(saida).reshape(-1, 1)).ravel()
        # A saída é uma reta no carbo e na caminhada. Nos extremos ela sai da faixa de iAUC do
        # treino, onde não há dado. O AG minimiza a previsão. Sem o piso, ele buscaria iAUC
        # negativa com pouco carbo e muita caminhada.
        return np.clip(iauc, IAUC_MINIMA, self.iauc_maxima)

    def prever_atributos(self, atributos: pd.DataFrame) -> np.ndarray:
        return self.para_iauc(self.modelo.predict_on_batch(self.entradas(atributos)))

    def prever(self, perfil: pd.Series, refeicoes: pd.DataFrame) -> np.ndarray:
        return self.prever_atributos(montar_atributos(refeicoes, pd.DataFrame([perfil])))


def escalas(treino: Particao) -> tuple[Pipeline, StandardScaler]:
    preparo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    # Com a perda MAE o passo do SGD não cresce com o erro. A iAUC chega a milhares de
    # mg/dL·min, e a rede levaria muitos passos pequenos até lá. Sem padronizar o alvo, ela
    # para cedo e erra mais na validação cruzada.
    escala_alvo = StandardScaler()
    return preparo.fit(treino.atributos), escala_alvo.fit(treino.iauc.reshape(-1, 1))


def sementes_das_redes(config: ConfigRede, semente: int) -> range:
    return range(semente * config.n_redes, (semente + 1) * config.n_redes)


def separar_validacao(treino: Particao, semente: int) -> tuple[np.ndarray, np.ndarray]:
    divisor = GroupShuffleSplit(n_splits=1, test_size=FRACAO_VALIDACAO, random_state=semente)
    return next(divisor.split(treino.atributos, groups=treino.refeicoes["participante"]))


def treinar(treino: Particao, config: ConfigRede, semente: int) -> tuple[Simulador, list[int]]:
    preparo, escala_alvo = escalas(treino)
    x = preparo.transform(treino.atributos).astype("float32")
    y = escala_alvo.transform(treino.iauc.reshape(-1, 1))
    redes, melhores_epocas = [], []
    for semente_rede in sementes_das_redes(config, semente):
        keras.utils.set_random_seed(semente_rede)
        rede = construir_modelo(config, nome=f"rede_{semente_rede}")
        ajuste, validacao = separar_validacao(treino, semente_rede)
        historico = rede.fit(
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
        redes.append(rede)
        melhores_epocas.append(int(np.argmin(historico.history["val_loss"])) + 1)
    simulador = Simulador(media_das_redes(redes), preparo, escala_alvo, float(treino.iauc.max()))
    return simulador, melhores_epocas


def particao_do_estudo(refeicoes: pd.DataFrame, perfis: pd.DataFrame) -> Particao:
    estudo = refeicoes[refeicoes["dia"].isin(DIAS_ESTUDO)]
    estudo = estudo.assign(fibra=estudo["fibra"].where(estudo["fibra"] <= estudo["carbo"]))
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
) -> tuple[pd.DataFrame, list[int]]:
    partes, melhores_epocas = [], []
    for fold, (treino, teste) in enumerate(dividir(refeicoes, perfis)):
        simulador, epocas_fold = treinar(treino, config, semente)
        melhores_epocas.extend(epocas_fold)
        previsto = {nome: baseline(treino, teste) for nome, baseline in BASELINES.items()}
        previsto["rede"] = simulador.prever_atributos(teste.atributos)
        partes.append(teste.refeicoes.assign(fold=fold, **previsto))
    return pd.concat(partes, ignore_index=True), melhores_epocas


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


def mae_por(previsoes: pd.DataFrame, coluna: str) -> pd.Series:
    plano = previsoes[previsoes["dia"].isin(DIAS_PLANO)]
    return (plano[ALVO] - plano["rede"]).abs().groupby(plano[coluna]).mean()


def treinar_final(
    refeicoes: pd.DataFrame,
    perfis: pd.DataFrame,
    config: ConfigRede = ConfigRede(),
    semente: int = SEMENTE,
) -> Simulador:
    estudo = particao_do_estudo(refeicoes, perfis)
    habito = estudo.recortar(estudo.refeicoes["dia"].isin(DIAS_HABITO).to_numpy())
    preparo, escala_alvo = escalas(habito)
    x = preparo.transform(habito.atributos).astype("float32")
    y = escala_alvo.transform(habito.iauc.reshape(-1, 1))
    redes = []
    for semente_rede in sementes_das_redes(config, semente):
        keras.utils.set_random_seed(semente_rede)
        rede = construir_modelo(config, nome=f"rede_{semente_rede}")
        rede.fit(x, y, epochs=config.epocas, batch_size=config.tamanho_lote, verbose=0)
        redes.append(rede)
    return Simulador(media_das_redes(redes), preparo, escala_alvo, float(habito.iauc.max()))


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
        "iauc_maxima": simulador.iauc_maxima,
    }
    joblib.dump(preparo, pasta / ARQUIVO_PREPARO)


def carregar(pasta: Path = ARTEFATOS) -> Simulador:
    preparo = joblib.load(pasta / ARQUIVO_PREPARO)
    return Simulador(modelo=keras.models.load_model(pasta / ARQUIVO_MODELO), **preparo)


def main() -> None:
    refeicoes = pd.read_parquet(ARTEFATOS / "refeicoes.parquet")
    perfis = pd.read_parquet(ARTEFATOS / "perfis.parquet")

    config = ConfigRede()
    previsoes, melhores_epocas = validar(refeicoes, perfis, config)
    print(f"GroupKFold({N_FOLDS}) por participante, conjunto de {config.n_redes} redes")
    print("MAE em mg/dL·min")
    print(resumir(previsoes).to_string(float_format="%.2f"))
    for coluna in ("tipo", "grupo"):
        print(f"\nMAE D7 a D9 da rede por {coluna}")
        print(mae_por(previsoes, coluna).round(1).to_string())

    epocas = int(np.median(melhores_epocas))
    simulador = treinar_final(refeicoes, perfis, replace(config, epocas=epocas))
    salvar(simulador)
    teste = teste_d7_d9(simulador, refeicoes, perfis)
    mae_teste = (teste[ALVO] - teste["previsto"]).abs().mean()
    print(f"\nmodelo final: os {perfis['participante'].nunique()} participantes, {epocas} épocas")
    print(f"a época é a mediana das {len(melhores_epocas)} paradas da validação cruzada")
    print(f"MAE em D7 a D9 de quem esteve no treino: {mae_teste:.1f}")
    print("Esse MAE mede pessoas conhecidas. Ele não se compara com o da validação cruzada.")
    print(f"gravado em {ARTEFATOS / ARQUIVO_MODELO} e {ARTEFATOS / ARQUIVO_PREPARO}")


if __name__ == "__main__":
    main()
