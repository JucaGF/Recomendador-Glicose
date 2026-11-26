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
    pass


class FalhaNoDownload(Exception):
    pass


def sha256_arquivo(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while bloco := arquivo.read(MEGABYTE):
            resumo.update(bloco)
    return resumo.hexdigest()


def conferir(caminho: Path, sha256: str) -> None:
    obtido = sha256_arquivo(caminho)
    if obtido != sha256:
        raise ChecksumErrado(f"checksum errado em {caminho}: esperado {sha256}, obtido {obtido}")


def baixar(url: str, destino: Path, sha256: str) -> None:
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
    with zipfile.ZipFile(caminho_zip) as arquivo_zip:
        for membro in arquivo_zip.infolist():
            alvo = pasta / membro.filename
            if membro.is_dir() or (alvo.exists() and alvo.stat().st_size == membro.file_size):
                continue
            arquivo_zip.extract(membro, pasta)


def resumo(pasta: Path) -> str:
    raiz = pasta / "CGMacros"
    participantes = sum(1 for p in raiz.glob("CGMacros-*") if p.is_dir())
    fotos = sum(1 for _ in raiz.rglob("*.jpg"))
    return f"{participantes} participantes, {fotos} fotos, checksum ok"


def preparar(pasta: Path, url: str = URL, sha256: str = SHA256) -> str:
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
    try:
        print(preparar(pasta_dados(), url))
    except (ChecksumErrado, FalhaNoDownload) as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
