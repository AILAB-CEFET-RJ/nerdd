"""
Build a local political-person alias lexicon from TSE open data.

Primary use case:
    Create a CSV with canonical names and public ballot names (nome de urna)
    for city-council candidates/vereadores, for use in rule/lexicon-based NER.

Data source:
    TSE Dados Abertos - consulta_cand_{year}.zip
    https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand/consulta_cand_{year}.zip

The script is deliberately conservative: it records data provenance and avoids
inferring anything about accusations or events in the downstream corpus.
"""

from __future__ import annotations

import csv
import io
import os
import re
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

TSE_CONSULTA_CAND_URL = (
    "https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand/"
    "consulta_cand_{year}.zip"
)

DEFAULT_ELECTED_STATUSES = {
    "ELEITO",
    "ELEITO POR QP",
    "ELEITO POR MEDIA",  # normalized form of MÉDIA
    "ELEITO POR MÉDIA",
}

COMMON_FIRST_NAMES = {
    # Small, intentionally incomplete stoplist to avoid very weak aliases.
    "ANA", "ANTONIO", "ANTÔNIO", "CARLOS", "DANIEL", "EDUARDO", "FERNANDO",
    "FRANCISCO", "JOAO", "JOÃO", "JORGE", "JOSE", "JOSÉ", "LUIZ", "LUCAS",
    "MARCELO", "MARCOS", "MARIA", "MARIO", "MÁRIO", "MAURO", "PAULO", "PEDRO",
    "RAFAEL", "RICARDO", "ROBERTO", "RODRIGO", "SANDRA", "SERGIO", "SÉRGIO",
}


@dataclass(frozen=True)
class LexiconRow:
    municipio: str
    uf: str
    ano_eleicao: str
    cargo: str
    nome_completo: str
    nome_urna: str
    alias: str
    partido: str
    numero_candidato: str
    sq_candidato: str
    situacao_candidatura: str
    situacao_totalizacao: str
    fonte: str
    campo_fonte: str
    confianca: str


def strip_accents(text: str) -> str:
    """Remove accents while preserving ASCII characters."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def normalize_key(text: object) -> str:
    """Normalize text for matching municipality, cargo and status fields."""
    if text is None:
        return ""
    text = str(text).strip()
    text = strip_accents(text)
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def clean_text(text: object) -> str:
    """Normalize whitespace, but preserve accents and casing from the source."""
    if text is None:
        return ""
    text = str(text).replace("\ufeff", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_any(row: dict[str, str], candidates: Sequence[str]) -> str:
    """Return the first non-empty value among possible column names."""
    for col in candidates:
        if col in row and clean_text(row[col]):
            return clean_text(row[col])
    return ""


def make_tse_url(year: int) -> str:
    return TSE_CONSULTA_CAND_URL.format(year=year)


def download_tse_zip(year: int, cache_dir: str | Path = "data/raw") -> Path:
    """Download the TSE consulta_cand ZIP for a given election year if needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"consulta_cand_{year}.zip"

    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path

    url = make_tse_url(year)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    return zip_path


def find_candidate_csv_members(zip_path: str | Path, uf: str | None = None) -> list[str]:
    """
    Find candidate CSV files inside the TSE ZIP.

    Recent files are usually named like consulta_cand_2024_BRASIL.csv.
    Some older releases may include UF-specific CSVs. This function supports both.
    """
    uf_key = normalize_key(uf or "")
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]

    candidate_names = [
        name for name in names
        if Path(name).name.lower().startswith("consulta_cand_")
    ]

    if uf_key:
        uf_specific = [
            name for name in candidate_names
            if f"_{uf_key}." in normalize_key(Path(name).name)
        ]
        # Prefer UF-specific files when present, otherwise use the national file.
        if uf_specific:
            return uf_specific

    national = [
        name for name in candidate_names
        if "BRASIL" in normalize_key(Path(name).name)
    ]
    return national or candidate_names


def iter_tse_candidate_rows(zip_path: str | Path, uf: str | None = None) -> Iterator[dict[str, str]]:
    """Yield rows from candidate CSV files inside a TSE consulta_cand ZIP."""
    members = find_candidate_csv_members(zip_path, uf=uf)
    if not members:
        raise FileNotFoundError(f"No consulta_cand CSV found inside {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        for member in members:
            with zf.open(member) as raw:
                # TSE CSV files are commonly encoded as Latin-1 and separated by semicolon.
                text = io.TextIOWrapper(raw, encoding="latin-1", newline="")
                reader = csv.DictReader(text, delimiter=";")
                for row in reader:
                    yield row


def alias_confidence(alias: str, nome_completo: str, source_field: str) -> str:
    """Assign a simple confidence label for NER use."""
    alias_norm = normalize_key(alias)
    full_norm = normalize_key(nome_completo)

    if not alias_norm:
        return "descartar"
    if alias_norm == full_norm:
        return "alta_nome_completo"
    if source_field == "NM_URNA_CANDIDATO":
        # Multi-token ballot names and names containing locality markers are useful for NER.
        if len(alias_norm.split()) >= 2:
            return "alta_nome_urna"
        if len(alias_norm) >= 5 and alias_norm not in COMMON_FIRST_NAMES:
            return "media_nome_urna_curto"
        return "baixa_nome_urna_curto"
    if source_field == "DERIVED_FROM_NM_URNA_CANDIDATO":
        return "media_derivada"
    return "media"


def generate_aliases(nome_completo: str, nome_urna: str, include_derived_aliases: bool = True) -> list[tuple[str, str, str]]:
    """
    Generate aliases from canonical name and ballot name.

    Returns tuples: (alias, source_field, confidence).
    """
    aliases: list[tuple[str, str, str]] = []

    for alias, source_field in [
        (nome_completo, "NM_CANDIDATO"),
        (nome_urna, "NM_URNA_CANDIDATO"),
    ]:
        alias = clean_text(alias)
        if alias:
            aliases.append((alias, source_field, alias_confidence(alias, nome_completo, source_field)))

    if include_derived_aliases and nome_urna:
        # Example: "Maurinho do Paiol" -> "Maurinho" as a weaker alias.
        first_token = clean_text(nome_urna).split()[0] if clean_text(nome_urna) else ""
        if first_token:
            first_norm = normalize_key(first_token)
            if len(first_norm) >= 5 and first_norm not in COMMON_FIRST_NAMES:
                aliases.append((first_token, "DERIVED_FROM_NM_URNA_CANDIDATO", "media_derivada"))

    # Deduplicate aliases preserving order.
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for alias, field, conf in aliases:
        key = normalize_key(alias)
        if key not in seen:
            seen.add(key)
            out.append((alias, field, conf))
    return out


def build_political_person_lexicon(
    municipios: Optional[Iterable[str]],
    anos_eleicao: Iterable[int],
    uf: str = "RJ",
    cargo: str | Iterable[str] = "VEREADOR",
    cache_dir: str | Path = "data/raw",
    elected_only: bool = False,
    elected_statuses: Optional[set[str]] = None,
    include_derived_aliases: bool = True,
) -> list[LexiconRow]:
    """
    Build a political-person alias lexicon from TSE candidate data.

    Parameters
    ----------
    municipios:
        Municipality names to keep, e.g. ["Nilópolis", "Mesquita"].
        Use None or an empty iterable to disable municipality filtering. This is
        appropriate for statewide offices such as DEPUTADO FEDERAL and
        DEPUTADO ESTADUAL.
    anos_eleicao:
        Election years, e.g. [2020, 2024]. For city council, municipal years are expected.
    uf:
        State abbreviation.
    cargo:
        Target office or offices. Examples: "VEREADOR" or
        ["DEPUTADO FEDERAL", "DEPUTADO ESTADUAL"]. Default: "VEREADOR".
    elected_only:
        If True, keep only rows whose totalization status indicates elected candidates.
        If False, keep all candidates for the office. For NER lexicons, False can be useful.
    include_derived_aliases:
        If True, create weaker derived aliases, such as the first token of the ballot name.

    Returns
    -------
    list[LexiconRow]
    """
    municipios_norm = {normalize_key(m) for m in municipios} if municipios else set()
    uf_norm = normalize_key(uf)
    if isinstance(cargo, str):
        cargos_norm = {normalize_key(cargo)}
    else:
        cargos_norm = {normalize_key(c) for c in cargo}
    elected_statuses_norm = {normalize_key(s) for s in (elected_statuses or DEFAULT_ELECTED_STATUSES)}

    rows: list[LexiconRow] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for year in anos_eleicao:
        zip_path = download_tse_zip(year, cache_dir=cache_dir)
        source_url = make_tse_url(year)

        for raw in iter_tse_candidate_rows(zip_path, uf=uf):
            row_uf = get_any(raw, ["SG_UF"])
            row_municipio = get_any(raw, ["NM_UE", "NM_MUNICIPIO", "NM_MUNICÍPIO"])
            row_cargo = get_any(raw, ["DS_CARGO"])

            if normalize_key(row_uf) != uf_norm:
                continue
            if municipios_norm and normalize_key(row_municipio) not in municipios_norm:
                continue
            if normalize_key(row_cargo) not in cargos_norm:
                continue

            situacao_totalizacao = get_any(raw, ["DS_SIT_TOT_TURNO", "DS_SITUACAO_TOTALIZACAO"])
            if elected_only and normalize_key(situacao_totalizacao) not in elected_statuses_norm:
                continue

            nome_completo = get_any(raw, ["NM_CANDIDATO", "NM_CANDIDATO_URNA"])
            nome_urna = get_any(raw, ["NM_URNA_CANDIDATO", "NM_URNA"])
            partido = get_any(raw, ["SG_PARTIDO"])
            numero = get_any(raw, ["NR_CANDIDATO"])
            sq_candidato = get_any(raw, ["SQ_CANDIDATO"])
            situacao_candidatura = get_any(raw, ["DS_SITUACAO_CANDIDATURA", "DS_DETALHE_SITUACAO_CAND"])
            ano_eleicao = get_any(raw, ["ANO_ELEICAO"])

            for alias, source_field, confidence in generate_aliases(
                nome_completo=nome_completo,
                nome_urna=nome_urna,
                include_derived_aliases=include_derived_aliases,
            ):
                key = (
                    normalize_key(row_municipio),
                    str(year),
                    normalize_key(nome_completo),
                    normalize_key(alias),
                    source_field,
                )
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    LexiconRow(
                        municipio=row_municipio,
                        uf=row_uf,
                        ano_eleicao=ano_eleicao or str(year),
                        cargo=row_cargo,
                        nome_completo=nome_completo,
                        nome_urna=nome_urna,
                        alias=alias,
                        partido=partido,
                        numero_candidato=numero,
                        sq_candidato=sq_candidato,
                        situacao_candidatura=situacao_candidatura,
                        situacao_totalizacao=situacao_totalizacao,
                        fonte=source_url,
                        campo_fonte=source_field,
                        confianca=confidence,
                    )
                )

    rows.sort(key=lambda r: (normalize_key(r.municipio), r.ano_eleicao, normalize_key(r.alias)))
    return rows

def write_lexicon_csv(rows: Sequence[LexiconRow], output_csv: str | Path) -> None:
    """Write lexicon rows to CSV in UTF-8."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(LexiconRow.__dataclass_fields__.keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def read_municipalities_file(path: str | Path) -> list[str]:
    """
    Read municipalities from a text file, one municipality per line.
    Empty lines and lines starting with # are ignored.
    """
    municipios: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                municipios.append(line)
    return municipios
