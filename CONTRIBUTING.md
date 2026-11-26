# Contribuindo

## Documentation Conventions

### Python

Docstrings em estilo Google, escritas em português. Os type hints carregam o tipo, e o docstring carrega o significado, então não repita o tipo no texto.

- Todo módulo começa com um docstring de uma linha de resumo, um parágrafo opcional e a linha `Autores:`.
- Arquivos de Joaquim Germano Félix que forem modificados ganham Miguel Mochizuki Silva como coautor. Arquivos novos levam só o autor que os escreveu.
- Toda classe e toda função pública levam docstring. Funções triviais de uma linha podem ter só o resumo. Funções com `_` no início ficam sem docstring, a menos que o porquê não seja óbvio.
- Use `Args:`, `Returns:` e `Raises:` só quando acrescentam algo além da assinatura.
- Comentários `#` continuam reservados ao porquê de uma decisão, como já estão no código.

Exemplo:

```python
"""Simulador de iAUC: rede Keras e baselines.

Treina a rede nos dias D1 a D6 e prevê a iAUC de 2 horas de cada refeição.

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""


def validar(refeicoes, perfis, config=ConfigRede(), semente=0):
    """Valida o simulador com GroupKFold por participante.

    Args:
        refeicoes: Tabela de refeições, no esquema `refeicoes`.
        perfis: Tabela de perfis por participante.
        config: Hiperparâmetros da rede.
        semente: Semente base das redes.

    Returns:
        As previsões por refeição e a melhor época de cada rede.
    """
```
