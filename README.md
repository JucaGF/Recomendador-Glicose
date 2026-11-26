# Recomendador de glicose com CGMacros

Projeto final de Introdução à Inteligência Artificial.
Joaquim Germano Félix
jucagfbr@gmail.com

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

## Use o contrato de dados

`src/glicose/esquema.py` define as colunas e os tipos das tabelas `refeicoes`, `dias`, `perfis` e `cardapio`.
`esquema.dados_sinteticos(n_participantes, semente)` gera tabelas pequenas nesse formato, para testar o algoritmo genético sem os dados reais.

## Rode os testes

```sh
uv run ruff check
uv run pytest -q
```
