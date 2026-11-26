"""Baixa o CGMacros 1.0.0 do PhysioNet, confere o SHA-256 e extrai os arquivos.

Autores: Joaquim Germano Félix, Miguel Mochizuki Silva
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from glicose.esquema import pasta_dados

URL = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
SHA256 = "05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d"
NOME_ZIP = "CGMacros_dateshifted365.zip"
MEGABYTE = 1024 * 1024


class ChecksumErrado(Exception):
    """O SHA-256 do arquivo não bate com o esperado."""
    pass


class FalhaNoDownload(Exception):
    """O servidor não respondeu ou a conexão caiu."""
    pass


def sha256_arquivo(caminho: Path) -> str:
    """SHA-256 do arquivo, lido em blocos de 1 MB."""
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while bloco := arquivo.read(MEGABYTE):
            resumo.update(bloco)
    return resumo.hexdigest()


def conferir(caminho: Path, sha256: str) -> None:
    """Confere o SHA-256 do arquivo.

    Raises:
        ChecksumErrado: Se o valor não bate.
    """
    obtido = sha256_arquivo(caminho)
    if obtido != sha256:
        raise ChecksumErrado(f"checksum errado em {caminho}: esperado {sha256}, obtido {obtido}")


def baixar(url: str, destino: Path, sha256: str) -> None:
    """Baixa a URL para um arquivo `.parcial`, confere o checksum e só então renomeia.

    Raises:
        FalhaNoDownload: Se a rede falhar.
        ChecksumErrado: Se o arquivo baixado estiver corrompido. O parcial é apagado.
    """
    parcial = destino.with_name(destino.name + ".parcial")
    print(f"baixando {url}")
    try:
        with urllib.request.urlopen(url) as resposta, parcial.open("wb") as arquivo:
            shutil.copyfileobj(resposta, arquivo, MEGABYTE)
    except OSError as erro:
        parcial.unlink(missing_ok=True)
        motivo = getattr(erro, "reason", erro)
        raise FalhaNoDownload(f"não foi possível baixar {url}: {motivo}") from erro
    try:
        conferir(parcial, sha256)
    except ChecksumErrado:
        parcial.unlink()
        raise
    parcial.replace(destino)


def extrair(caminho_zip: Path, pasta: Path) -> None:
    """Extrai o zip na pasta, pulando os arquivos que já existem com o mesmo tamanho."""
    with zipfile.ZipFile(caminho_zip) as arquivo_zip:
        for membro in arquivo_zip.infolist():
            alvo = pasta / membro.filename
            if membro.is_dir() or (alvo.exists() and alvo.stat().st_size == membro.file_size):
                continue
            arquivo_zip.extract(membro, pasta)


def resumo(pasta: Path) -> str:
    """Linha final do script, com a contagem de participantes e fotos."""
    raiz = pasta / "CGMacros"
    participantes = sum(1 for p in raiz.glob("CGMacros-*") if p.is_dir())
    fotos = sum(1 for _ in raiz.rglob("*.jpg"))
    return f"{participantes} participantes, {fotos} fotos, checksum ok"


def preparar(pasta: Path, url: str = URL, sha256: str = SHA256) -> str:
    """Garante que o zip esteja na pasta, conferido e extraído.

    Args:
        pasta: Destino dos dados.
        url: Endereço do zip.
        sha256: Checksum oficial esperado.

    Returns:
        A linha de resumo.

    Raises:
        ChecksumErrado: Se o zip que já estava na pasta não confere.
    """
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_zip = pasta / NOME_ZIP
    if caminho_zip.exists():
        try:
            conferir(caminho_zip, sha256)
        except ChecksumErrado as erro:
            raise ChecksumErrado(f"{erro}. Apague {caminho_zip} e rode de novo.") from None
    else:
        baixar(url, caminho_zip, sha256)
    extrair(caminho_zip, pasta)
    return resumo(pasta)


def main(url: str = URL) -> int:
    """Executa o download.

    Returns:
        Código de saída, 0 em caso de sucesso e 1 se houver erro.
    """
    try:
        print(preparar(pasta_dados(), url))
    except (ChecksumErrado, FalhaNoDownload) as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
