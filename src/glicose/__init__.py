"""Recomendador de refeições e caminhadas para controle glicêmico com o CGMacros.

Define o backend do Keras antes de qualquer importação, então importe sempre pelo pacote.

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
