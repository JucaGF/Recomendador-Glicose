import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import baixar_dados
from glicose import esquema

SCRIPT = Path(__file__).parents[1] / "scripts" / "baixar_dados.py"


def zip_falso(pasta: Path) -> tuple[Path, str]:
    caminho = pasta / "origem.zip"
    with zipfile.ZipFile(caminho, "w") as arquivo_zip:
        arquivo_zip.writestr("CGMacros/bio.csv", "subject\n1\n")
        arquivo_zip.writestr("CGMacros/CGMacros-001/CGMacros-001.csv", "Timestamp\n")
        arquivo_zip.writestr("CGMacros/CGMacros-001/photos/a.jpg", b"\xff\xd8\xff")
        arquivo_zip.writestr("CGMacros/CGMacros-002/CGMacros-002.csv", "Timestamp\n")
        arquivo_zip.writestr("CGMacros/CGMacros-002/photos/b.jpg", b"\xff\xd8\xff")
    return caminho, hashlib.sha256(caminho.read_bytes()).hexdigest()


def trocar_um_byte(caminho: Path) -> None:
    dados = bytearray(caminho.read_bytes())
    dados[len(dados) // 2] ^= 0xFF
    caminho.write_bytes(bytes(dados))


def test_zip_existente_com_checksum_certo_extrai_sem_baixar(tmp_path):
    origem, sha = zip_falso(tmp_path)
    pasta = tmp_path / "dados"
    pasta.mkdir()
    origem.rename(pasta / baixar_dados.NOME_ZIP)

    linha = baixar_dados.preparar(pasta, url="http://nao-deve-ser-acessado.invalid", sha256=sha)

    assert linha == "2 participantes, 2 fotos, checksum ok"
    assert (pasta / "CGMacros" / "bio.csv").exists()


def test_zip_com_um_byte_trocado_falha_sem_extrair(tmp_path):
    origem, sha = zip_falso(tmp_path)
    pasta = tmp_path / "dados"
    pasta.mkdir()
    trocar_um_byte(origem)
    origem.rename(pasta / baixar_dados.NOME_ZIP)

    with pytest.raises(baixar_dados.ChecksumErrado, match="checksum"):
        baixar_dados.preparar(pasta, sha256=sha)

    assert not (pasta / "CGMacros").exists()


def test_script_sai_com_erro_no_zip_corrompido(tmp_path):
    origem, _ = zip_falso(tmp_path)
    trocar_um_byte(origem)
    origem.rename(tmp_path / baixar_dados.NOME_ZIP)

    resultado = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**os.environ, "GLICOSE_DADOS": str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert resultado.returncode != 0
    assert "checksum" in resultado.stderr
    assert f"Apague {tmp_path / baixar_dados.NOME_ZIP} e rode de novo" in resultado.stderr
    assert not (tmp_path / "CGMacros").exists()


def test_falha_de_rede_vira_uma_linha_de_erro(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GLICOSE_DADOS", str(tmp_path))
    ausente = (tmp_path / "nao_existe.zip").as_uri()

    codigo = baixar_dados.main(url=ausente)

    erro = capsys.readouterr().err
    assert codigo != 0
    assert erro.startswith(f"erro: não foi possível baixar {ausente}")
    assert erro.count("\n") == 1
    assert list(tmp_path.iterdir()) == []


def test_download_confere_checksum_antes_de_gravar_o_zip(tmp_path):
    origem, sha = zip_falso(tmp_path)
    pasta = tmp_path / "dados"

    linha = baixar_dados.preparar(pasta, url=origem.as_uri(), sha256=sha)
    assert linha == "2 participantes, 2 fotos, checksum ok"

    outra = tmp_path / "outra"
    with pytest.raises(baixar_dados.ChecksumErrado):
        baixar_dados.preparar(outra, url=origem.as_uri(), sha256="0" * 64)
    assert list(outra.iterdir()) == []


@pytest.mark.parametrize("semente", [0, 1])
def test_dados_sinteticos_seguem_o_contrato(semente):
    tabelas = esquema.dados_sinteticos(n_participantes=5, semente=semente)

    assert tabelas.keys() == esquema.TABELAS.keys()
    for nome, tabela in tabelas.items():
        contrato = esquema.TABELAS[nome]
        assert list(tabela.columns) == list(contrato), nome
        assert {coluna: str(tipo) for coluna, tipo in tabela.dtypes.items()} == contrato, nome
        assert not tabela.isna().any().any(), nome
    assert tabelas["perfis"]["participante"].tolist() == [1, 2, 3, 4, 5]
    assert set(tabelas["refeicoes"]["tipo"]) == set(esquema.TIPOS_REFEICAO)
    assert tabelas["cardapio"]["prato"].tolist() == list(range(len(tabelas["cardapio"])))


def test_dados_sinteticos_repetem_com_a_mesma_semente():
    primeira = esquema.dados_sinteticos(3, semente=7)
    segunda = esquema.dados_sinteticos(3, semente=7)
    for nome in esquema.TABELAS:
        assert primeira[nome].equals(segunda[nome]), nome
