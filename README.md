# Recomendador de glicose com CGMacros

Projeto final de Introdução à Inteligência Artificial.
Joaquim Germano Félix
jucagfbr@gmail.com
Miguel Mochizuki Silva
miguelmochizukisilva@gmail.com

O sistema recomenda refeições e caminhadas para os dias D7 a D9 de cada participante do CGMacros.
Uma rede Keras, treinada nos dias D1 a D6, prevê a glicose depois de cada refeição.
Um algoritmo genético escolhe o plano que minimiza essa previsão sem fugir muito do hábito da pessoa.

## Prepare o ambiente

O projeto usa Python 3.12 e o [uv](https://docs.astral.sh/uv/).
O TensorFlow estável ainda não tem versão para Python 3.14, e o Keras usa o TensorFlow como backend.

```sh
uv sync
```

## Baixe os dados

Os dados vêm do [CGMacros 1.0.0 no PhysioNet](https://physionet.org/content/cgmacros/1.0.0/).
O script baixa o zip de 657 MB, confere o SHA-256 oficial e extrai os CSVs e as fotos.

```sh
export GLICOSE_DADOS=~/dados/cgmacros
uv run python scripts/baixar_dados.py
```

A última linha deve ser `45 participantes, 3454 fotos, checksum ok`.
Sem `GLICOSE_DADOS`, os dados vão para `data/`, que o git ignora.
Se o zip já existe com o checksum certo, o script não baixa de novo.
Se o checksum não confere, o script sai com erro e não extrai nada. Nesse caso, apague o zip e rode de novo.

## Rode as etapas

Cada etapa é um módulo do pacote `glicose`, e cada módulo entra no repositório em uma PR própria, nesta ordem.

1. `uv run python -m glicose.refeicoes` monta a tabela de refeições com a iAUC de 2 horas.
2. `uv run python -m glicose.ag --falso` roda o algoritmo genético com um simulador falso.
3. `uv run python -m glicose.preparar` grava refeições, perfis e cardápio em `artefatos/`.
4. `uv run python -m glicose.nsga2 --falso` calcula a frente de Pareto com o NSGA-II.
5. `uv run python -m glicose.simulador` treina a rede Keras e compara com os baselines.
6. `uv run python -m glicose.recomendar --participante 001` recomenda o plano de D7 a D9 de um participante.
7. `uv run python -m glicose.validacao` mede o simulador com intervalos de confiança e testes pareados.

## Valide o simulador

`glicose.validacao` compara a rede com os baselines em D7 a D9, em dois cenários.
No cenário ii (pessoa nova), o GroupKFold por participante é repetido 5 vezes com partições diferentes.
No cenário i (pessoa conhecida), o treino usa D1 a D6 de todos e o teste usa D7 a D9 de todos.

A unidade estatística é o participante.
O intervalo de 95% vem de bootstrap sobre participantes (10 000 reamostragens), e os testes são Wilcoxon pareados com correção de Holm dentro de cada métrica.
O MAE é a métrica principal e o RMSE é uma segunda família.
Viés, erro da soma de D9 e correlação de Pearson são descritivos.
A calibração usa o desvio entre as 5 redes do conjunto.
Uma ablação treina também com perda MSE e Huber no cenário ii.

```sh
uv run python -m glicose.validacao --repeticoes 5
```

Cada perda é independente, então dá para treinar em processos separados e juntar depois:

```sh
uv run python -m glicose.validacao --etapa treinar --perdas mse
uv run python -m glicose.validacao --etapa analisar
```

O relatório sai no terminal, e as tabelas e as figuras `contrastes.png` e `calibracao.png` ficam em `artefatos/validacao/`.
Uma execução com semente 0 e 5 repetições leva cerca de 20 minutos com as três perdas em paralelo, na CPU.

## Use o contrato de dados

`src/glicose/esquema.py` define as colunas e os tipos das tabelas `refeicoes`, `dias`, `perfis` e `cardapio`.
`esquema.dados_sinteticos(n_participantes, semente)` gera tabelas pequenas nesse formato, para testar o algoritmo genético sem os dados reais.

## Organização do projeto

```
Recomendador-Glicose/
├── .ruff_cache/
│   ├── 0.17.6/
│   │   ├── 13958010919202077989
│   │   ├── 17322104889019514870
│   │   └── 3906197522659392227
│   ├── .gitignore
│   └── CACHEDIR.TAG
├── scripts/      # download dos dados
│   └── baixar_dados.py
├── src/          # código-fonte
│   └── glicose/  # pacote do projeto
│       ├── __init__.py
│       ├── ag.py
│       ├── ag_hiper.py
│       ├── baselines_modelo.py
│       ├── cardapio.py
│       ├── diferenciais.py
│       ├── esquema.py
│       ├── experimentos.py
│       ├── figuras.py
│       ├── leitura.py
│       ├── nsga2.py
│       ├── otimizadores.py
│       ├── perfil.py
│       ├── plano.py
│       ├── preparar.py
│       ├── recomendar.py
│       ├── refeicoes.py
│       ├── simulador.py
│       └── validacao.py
├── tests/        # testes com pytest
│   ├── test_ag.py
│   ├── test_ag_hiper.py
│   ├── test_baixar_dados.py
│   ├── test_diferenciais.py
│   ├── test_otimizadores.py
│   ├── test_perfil_cardapio.py
│   ├── test_recomendar.py
│   ├── test_refeicoes.py
│   ├── test_simulador.py
│   └── test_validacao.py
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml
└── README.md
```

Cada módulo de `src/glicose/` tem um docstring de abertura que diz o que ele faz e quem o escreveu.

## Ferramentas

- Python 3.12, gerenciado com [uv](https://docs.astral.sh/uv/).
- Keras 3 com TensorFlow para a rede do simulador, e XGBoost e scikit-learn para os baselines e a validação.
- pandas, NumPy e PyArrow para as tabelas e os parquets em `artefatos/`. Matplotlib para as figuras.
- pytest para os testes e ruff para o lint, com linhas de até 100 caracteres.

## Rode os testes

```sh
uv run ruff check
uv run pytest -q
```

## Convenções de documentação

Os docstrings seguem o estilo Google, em português, e todo módulo termina o docstring com a linha `Autores:`.
O modelo completo e um exemplo estão em [CONTRIBUTING.md](CONTRIBUTING.md).

## Autoria

O projeto foi escrito por Joaquim Germano Félix.
Miguel Mochizuki Silva é coautor dos arquivos que modificou e autor do que escreveu depois.
