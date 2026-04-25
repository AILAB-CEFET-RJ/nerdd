#!/usr/bin/env python3
"""Build conservative multilabel pseudolabels from a metadata-anchored Location candidate pool."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_dense_tips import read_json_or_jsonl, write_jsonl

NAME_TOKEN = r"[A-ZÀ-Ú][A-Za-zÀ-ÿ]+"
NAME_SEP = r"[ \t]+"
FULL_NAME = rf"{NAME_TOKEN}(?:{NAME_SEP}{NAME_TOKEN}){{1,4}}"
ROLE_NAME = rf"{NAME_TOKEN}(?:{NAME_SEP}{NAME_TOKEN}){{0,2}}"
ALIAS_TOKEN = r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]+"
ALIAS_SPAN = rf"{ALIAS_TOKEN}(?:{NAME_SEP}[A-Za-zÀ-ÿ]*[0-9][A-Za-zÀ-ÿ0-9_-]*)?"
PERSON_ROLE_WORDS = (
    "Sargento",
    "Cabo",
    "Soldado",
    "Tenente",
    "Capitao",
    "Capitão",
    "Major",
    "Coronel",
    "Delegado",
    "Investigador",
    "Agente",
    "Inspetor",
    "Detetive",
)
PERSON_ALIAS_MARKERS = (
    "alcunha",
    "apelidado",
    "apelido",
    "chamado",
    "conhecido",
    "vulgo",
)
WEAK_PERSON_PATTERNS = {
    "weak_person_known_as": re.compile(
        rf"(?i:\b(?:conhecido|conhecida|chamado|chamada|apelidado|apelidada)\s+(?:como|de)\s+)(?P<name>{NAME_TOKEN})"
    ),
    "weak_person_vulgo_alias": re.compile(
        rf"(?i:\bvulgo\s+)(?P<name>{NAME_TOKEN})"
    ),
    "weak_person_after_action": re.compile(
        rf"(?i:\b(?:matou|assassinou|executou|esfaqueou|baleou|atirou\s+em)\s+)(?P<name>{FULL_NAME})"
    ),
    "weak_person_victim_context": re.compile(
        rf"(?i:\b(?:vitima\s+(?:e|é|foi)\s*:?\s*|morte\s+d[eo]\s+|vida\s+de\s+))(?P<name>{FULL_NAME})"
    ),
}
ORG_LITERAL_PATTERNS = (
    r"\b\d{1,3}[ªa°º]?\s*DP\b(?:\s+de\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+(?:\s+[A-ZÀ-Ú][A-Za-zÀ-ÿ]+){0,3})?",
    r"\b\d{1,3}[ªa°º]?\s*BPM\b",
    r"\b(?:BOPE|DRACO|GAECO|CEDAE|Light)\b",
    r"\b(?:Comando Vermelho|Terceiro Comando|Liga da Justica|Liga da Justiça)\b",
    r"\b(?:TCP|ADA|CV|PCC)\b",
    r"\b[Mm]il[ií]c(?:ia|ía)\s+(?:de|da|do)\s+(?-i:[A-ZÀ-Ú][A-Za-zÀ-ÿ]+)(?:\s+(?-i:[A-ZÀ-Ú][A-Za-zÀ-ÿ]+)){0,3}",
    r"\b\d{1,3}\s*(?:Batalh[aã]o|Batalhao)\b",
    r"\bDH\b",
    r"\b(?:batalhao|batalhão)\s+de\s+choque\b",
    r"\bPol[ií]cia\s+Militar\b",
)
PERSON_FORBIDDEN_SUBSTRINGS = (
    "avenida",
    "bar",
    "bairro",
    "casa",
    "chamado",
    "de nome",
    "estrada",
    "justica",
    "justiça",
    "local",
    "operacao",
    "operação",
    "policia",
    "polícia",
    "por",
    "praca",
    "praça",
    "rua",
    "que",
    "traficante",
)
PERSON_FORBIDDEN_TOKENS = {
    "area",
    "familia",
    "filho",
    "frio",
    "mando",
    "mandaram",
    "morte",
    "moradores",
    "nome",
    "pm",
    "raiva",
    "realizando",
    "sem",
}
PERSON_FORBIDDEN_SINGLE_ALIAS_NAMES = {
    "caonze",
    "chatuba",
    "chapadao",
    "coreia",
    "mesquita",
    "mundel",
    "pavuna",
    "predinhos",
}
PERSON_FORBIDDEN_START_TOKENS = {
    "a",
    "as",
    "essa",
    "essas",
    "esse",
    "esses",
    "favela",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "outro",
    "outra",
    "um",
    "uma",
}
PERSON_STRIPPABLE_PREFIX_TOKENS = {"essa", "essas", "esse", "esses", "vagabundo"}
PERSON_FOLLOWING_VERBS = {
    "ajuda",
    "ajudar",
    "ajudando",
    "esta",
    "está",
    "fazer",
    "fechar",
    "fugiu",
    "mandou",
    "mantem",
    "mantém",
    "manter",
    "matou",
    "realiza",
    "realizar",
    "realizando",
}
PERSON_ALL_CAPS_FORBIDDEN_TOKENS = PERSON_FORBIDDEN_TOKENS | {
    "bahia",
    "bope",
    "na",
    "no",
}
LOCATION_FORBIDDEN_SUBSTRINGS = (
    " bar ",
    " com ",
)
LOCATION_MARKERS = (
    "alameda",
    "av",
    "avenida",
    "estrada",
    "praca",
    "praça",
    "rodovia",
    "rua",
    "travessa",
    "trav",
    "trv",
)
LOGRADOURO_MARKER_ALIASES = {
    "alameda": "alameda",
    "av": "avenida",
    "avenida": "avenida",
    "beco": "beco",
    "estrada": "estrada",
    "ladeira": "ladeira",
    "praca": "praca",
    "praça": "praca",
    "rodovia": "rodovia",
    "rua": "rua",
    "ruas": "rua",
    "trav": "travessa",
    "travessa": "travessa",
    "trv": "travessa",
}
LOGRADOURO_MARKER_RE = re.compile(
    r"\b(?:alameda|av\.?|avenida|beco|estrada|ladeira|pra[cç]a|rodovia|ruas?|travessa|trav\.?|trv\.?)\b",
    re.IGNORECASE,
)
LOGRADOURO_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
LOGRADOURO_CONNECTOR_TOKENS = {"d", "da", "das", "de", "do", "dos", "e"}
LOGRADOURO_STOP_TOKENS = {
    "bar",
    "alto",
    "casa",
    "com",
    "escadao",
    "escadão",
    "esquina",
    "fica",
    "ficam",
    "ha",
    "há",
    "na",
    "localidade",
    "no",
    "num",
    "numa",
    "perto",
    "para",
    "ponte",
    "proximo",
    "próximo",
    "subida",
    "tem",
    "trailer",
    "vulgo",
}
LOGRADOURO_SIMPLIFY_DROP_TOKENS = {"d", "da", "das", "de", "do", "dos"}
BARE_LOGRADOURO_CONTEXT_RE = re.compile(
    r"\b(?:esquina|proximo[ \t]+a|pr[oó]ximo[ \t]+a|perto[ \t]+d[aoe])[ \t]+"
    r"(?P<name>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ]+(?:[ \t]+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ]+){1,5})",
    re.IGNORECASE,
)
KNOWN_COMMUNITIES_RE = re.compile(
    r"\b(?:"
    r"(?:(?:complexo|morro|comunidade|favela)[ \t]+(?:d[oa][ \t]+)?)?chapa[ \t]*d[aã]o"
    r"|(?:comunidade[ \t]+d[ae][ \t]+)?coreia"
    r"|pavuna"
    r"|caonze"
    r"|chatuba"
    r"|k11"
    r"|vila[ \t]+lage"
    r"|morro[ \t]+final[ \t]+feliz"
    r")\b",
    re.IGNORECASE,
)
KNOWN_BAIRROS_BY_CITY = {
    "nilopolis": {"olinda", "paiol de polvora"},
}
BAIRRO_CONTEXT_RE = re.compile(
    r"\bbairro\s+(?:d[eo]\s+)?(?P<name>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ]+(?:[ \t]+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ]+){0,4})",
    re.IGNORECASE,
)
LOGRADOURO_INTERSECTION_RE = re.compile(
    rf"\besquina\s+d(?:a|e|o)s?\s+ruas?\s+"
    rf"(?P<first>{NAME_TOKEN}(?:{NAME_SEP}{NAME_TOKEN}){{0,4}})\s+"
    rf"(?:com|e)\s+"
    rf"(?P<second>{NAME_TOKEN}(?:{NAME_SEP}{NAME_TOKEN}){{0,4}})"
    rf"(?=\s+(?:ficam?|tem|h[áa]|,|\.|\n|\r|$))",
    re.IGNORECASE,
)
CITY_ALIASES = {
    "sao joao meriti": "sao joao de meriti",
}

PERSON_PATTERNS = {
    "person_firstname_vulgo_alias": re.compile(
        rf"(?P<name>{NAME_TOKEN})\s*,?\s+(?i:vulgo)\s+(?P<alias>{ALIAS_SPAN})",
    ),
    "person_called_fullname": re.compile(
        rf"(?i:\bchamad[oa]\s+)(?P<fullname>{FULL_NAME})(?=\s*(?:,|\.|\n|\r|$|\s+(?:vulgo|conhecid[ao]|hoje|fica|ficam|é|e\b)))",
    ),
    "person_known_as": re.compile(
        rf"(?i:\b(?:de\s+nome\s+)?)(?P<fullname>{NAME_TOKEN}(?:{NAME_SEP}{NAME_TOKEN}){{0,4}})\s+"
        rf"(?i:conhecid[ao]\s+como|chamad[ao]\s+de)\s+(?P<alias>{ALIAS_SPAN})",
    ),
    "person_vulgo_alias_fullname": re.compile(
        rf"(?i:\bvulgo\s+)(?P<alias>{ALIAS_SPAN})\s+(?P<fullname>{FULL_NAME})",
    ),
    "person_article_vulgo_alias": re.compile(
        rf"(?i:\bo\s+vulgo\s+)(?P<name>{ALIAS_SPAN})",
    ),
    "person_apelido_alias": re.compile(
        rf"(?i:\bapelido\s+)(?P<name>{ALIAS_SPAN})",
    ),
    "person_nome_fullname": re.compile(
        rf"(?i:\bnome\s+)(?P<fullname>{FULL_NAME})\s*,?\s+(?i:(?:de\s+)?vulgo)\s+(?P<alias>{ALIAS_SPAN})",
    ),
    "person_vulgo_fullname": re.compile(
        rf"(?P<fullname>{FULL_NAME})\s*,?\s+(?i:(?:de\s+)?vulgo)\s+(?P<alias>{ALIAS_SPAN})",
    ),
    "person_role_name": re.compile(
        rf"\b(?i:(?:{'|'.join(PERSON_ROLE_WORDS)}))\s+(?P<name>{ROLE_NAME})",
    ),
}
PERSON_COORD_PATTERNS = {
    "person_sibling_coordination": re.compile(
        rf"(?P<n1>{FULL_NAME})\s*,\s*(?P<n2>{FULL_NAME})\s+(?:[Ss][aã]o|[Ee])\s+irm[aã]os?\b",
    ),
}
PERSON_LEXICON_FULL_NAME_RE = re.compile(rf"\b(?P<fullname>{NAME_TOKEN}{NAME_SEP}{NAME_TOKEN})\b")
PERSON_LEXICON_LIST_RE = re.compile(
    rf"(?P<n1>{NAME_TOKEN}{NAME_SEP}{NAME_TOKEN})\s*,\s*(?P<n2>{NAME_TOKEN}{NAME_SEP}{NAME_TOKEN})"
    rf"(?=\s+(?:[Ss][aã]o|ficam|est[aã]o|vendem|trabalham|moram)\b)"
)
PERSON_LEXICON_PRE_CONTEXT_RE = re.compile(
    r"(?:^|[^a-zà-ÿ])(?:nome|chamad[oa]|dono|traficante|vagabundo|vapor|gerente)\s+$"
)
PERSON_LEXICON_POST_CONTEXT_RE = re.compile(
    r"^\s*,?\s*(?:vulgo|[Ss][aã]o|fica|ficam|est[aã]o|vende|vendem|atua|atuam|trabalha|trabalham|mora|moram|irm[aã]os?)\b"
)
ORG_PATTERNS = {
    f"organization_pattern_{idx}": re.compile(pattern, re.IGNORECASE)
    for idx, pattern in enumerate(ORG_LITERAL_PATTERNS, start=1)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build conservative Person/Location/Organization pseudolabels from a metadata candidate pool."
    )
    parser.add_argument("--input", required=True, help="Input candidate pool JSONL or JSON.")
    parser.add_argument("--output-pool-jsonl", required=True, help="Full accepted multilabel pool JSONL.")
    parser.add_argument("--output-review-jsonl", required=True, help="Top-N review JSONL with audit metadata.")
    parser.add_argument("--output-pseudolabel-jsonl", required=True, help="Final top-N pseudolabel JSONL.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output.")
    parser.add_argument("--top-n", type=int, default=20, help="How many accepted rows to export for review/refit.")
    parser.add_argument(
        "--require-two-location-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exactly two existing Location seeds in the source row.",
    )
    parser.add_argument(
        "--min-person-seeds",
        type=int,
        default=1,
        help="Minimum number of conservative Person seeds required.",
    )
    parser.add_argument(
        "--min-organization-seeds",
        type=int,
        default=1,
        help="Minimum number of conservative Organization seeds required.",
    )
    parser.add_argument(
        "--drop-incomplete-person-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with strong weak-Person signal not covered by accepted Person seeds.",
    )
    parser.add_argument(
        "--max-uncovered-person-signals",
        type=int,
        default=2,
        help="Maximum weak Person signals not covered by accepted Person seeds before dropping a row.",
    )
    parser.add_argument(
        "--max-rows-per-cluster",
        type=int,
        default=2,
        help="Maximum accepted rows to keep per normalized near-duplicate cluster. Use 0 to disable.",
    )
    parser.add_argument(
        "--logradouro-lexicon-jsonl",
        default="",
        help="Optional IBGE logradouro lexicon JSONL used to add validated Location spans.",
    )
    parser.add_argument(
        "--enable-lexicon-location-expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add Location spans validated against --logradouro-lexicon-jsonl and cidadeLocal.",
    )
    parser.add_argument(
        "--max-uncovered-logradouro-signals",
        type=int,
        default=8,
        help="Drop rows with more than this many unvalidated/uncovered logradouro candidates. Use -1 to disable.",
    )
    parser.add_argument(
        "--prenomes-lexicon",
        default="",
        help="Optional IBGE first-name lexicon TXT/JSON/JSONL used for conservative Person expansion.",
    )
    parser.add_argument(
        "--enable-prenome-person-expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add Person spans from --prenomes-lexicon in strong local contexts.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    return " ".join(text.split())


def normalize_relato_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r",(?=[^\s\d])", ", ", text)
    return text


def normalize_city(text: str) -> str:
    normalized = normalize_text(text)
    return CITY_ALIASES.get(normalized, normalized)


def flexible_literal_pattern(phrase: str) -> re.Pattern | None:
    tokens = [re.escape(token) for token in str(phrase or "").strip().split()]
    if not tokens:
        return None
    return re.compile(r"(?<![A-Za-zÀ-ÿ])" + r"\s+".join(tokens) + r"(?![A-Za-zÀ-ÿ])", re.IGNORECASE)


def accent_flexible_literal_pattern(phrase: str) -> re.Pattern | None:
    accent_classes = {
        "a": "aáàâãä",
        "e": "eéèêë",
        "i": "iíìîï",
        "o": "oóòôõö",
        "u": "uúùûü",
        "c": "cç",
    }
    token_patterns = []
    for token in str(phrase or "").strip().split():
        pieces = []
        for char in token:
            base = normalize_text(char)
            if len(base) == 1 and base in accent_classes:
                chars = accent_classes[base]
                pieces.append("[" + re.escape(chars + chars.upper()) + "]")
            else:
                pieces.append(re.escape(char))
        token_patterns.append("".join(pieces))
    if not token_patterns:
        return None
    return re.compile(
        r"(?<![A-Za-zÀ-ÿ0-9])" + r"\s+".join(token_patterns) + r"(?![A-Za-zÀ-ÿ0-9])",
        re.IGNORECASE,
    )


def get_text(row: dict) -> str:
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_entities(row: dict) -> list[dict]:
    entities = row.get("entities")
    return entities if isinstance(entities, list) else []


def load_prenomes_lexicon(path: str) -> set[str]:
    if not path:
        return set()
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("res"), list):
            rows = payload["res"]
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict) and isinstance(payload[0].get("res"), list):
            rows = payload[0]["res"]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
    elif source.suffix.lower() == ".jsonl":
        rows = read_json_or_jsonl(str(source))
    else:
        rows = [{"nome": line.strip().split()[0]} for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

    names = set()
    for row in rows:
        if isinstance(row, str):
            name = row
        elif isinstance(row, dict):
            name = row.get("nome") or row.get("name") or row.get("prenome") or ""
        else:
            name = ""
        normalized = normalize_text(name).split()
        if len(normalized) == 1 and len(normalized[0]) >= 3:
            names.add(normalized[0])
    return names


def simplify_logradouro_norm(text: str) -> str:
    return " ".join(token for token in normalize_text(text).split() if token not in LOGRADOURO_SIMPLIFY_DROP_TOKENS)


def load_logradouro_lexicon(path: str) -> dict[str, dict[str, set[str]]]:
    if not path:
        return {}
    index: dict[str, dict[str, set[str]]] = {}
    all_index = index.setdefault("__all__", {"canonical": set(), "simplified": set(), "simplified_prefix": set(), "bare": set()})
    for row in read_json_or_jsonl(path):
        city = normalize_city(row.get("municipio_norm") or row.get("municipio") or "")
        name = normalize_text(row.get("canonical_name_norm") or row.get("canonical_name") or "")
        if not city or not name:
            continue
        city_index = index.setdefault(city, {"canonical": set(), "simplified": set(), "simplified_prefix": set(), "bare": set()})
        simplified_name = simplify_logradouro_norm(name)
        city_index["canonical"].add(name)
        city_index["simplified"].add(simplified_name)
        all_index["canonical"].add(name)
        all_index["simplified"].add(simplified_name)
        simplified_tokens = simplified_name.split()
        for end_idx in range(2, len(simplified_tokens)):
            prefix = " ".join(simplified_tokens[:end_idx])
            city_index["simplified_prefix"].add(prefix)
            all_index["simplified_prefix"].add(prefix)
        bare_name = normalize_text(row.get("nome") or "")
        if bare_name:
            city_index["bare"].add(simplify_logradouro_norm(bare_name))
            all_index["bare"].add(simplify_logradouro_norm(bare_name))
    return index


def logradouro_match_origin(
    candidate_norm: str,
    city_lexicon: dict[str, set[str]],
    global_lexicon: dict[str, set[str]] | None,
) -> str:
    simplified = simplify_logradouro_norm(candidate_norm)
    simplified_tokens = simplified.split()
    bare = " ".join(simplified_tokens[1:]) if len(simplified_tokens) > 1 else ""
    if (
        candidate_norm in city_lexicon["canonical"]
        or simplified in city_lexicon["simplified"]
        or simplified in city_lexicon["simplified_prefix"]
        or (bare and bare in city_lexicon["bare"])
    ):
        return "ibge_logradouro_lexicon"
    if global_lexicon and (
        candidate_norm in global_lexicon["canonical"]
        or simplified in global_lexicon["simplified"]
        or simplified in global_lexicon["simplified_prefix"]
        or (bare and bare in global_lexicon["bare"])
    ):
        return "ibge_logradouro_lexicon_global"
    return ""


def bare_logradouro_match_origin(
    bare_norm: str,
    city_lexicon: dict[str, set[str]],
    global_lexicon: dict[str, set[str]] | None,
) -> str:
    if bare_norm in city_lexicon["bare"]:
        return "ibge_logradouro_lexicon"
    if global_lexicon and bare_norm in global_lexicon["bare"]:
        return "ibge_logradouro_lexicon_global"
    return ""


def canonicalize_logradouro_marker(marker: str) -> str:
    normalized = normalize_text(str(marker or "").replace(".", ""))
    return LOGRADOURO_MARKER_ALIASES.get(normalized, normalized)


def normalize_logradouro_candidate(text: str) -> str:
    tokens = normalize_text(text).split()
    if not tokens:
        return ""
    tokens[0] = canonicalize_logradouro_marker(tokens[0])
    return " ".join(tokens)


def candidate_prefixes(candidate: dict, *, min_tokens: int = 2) -> list[dict]:
    raw_tokens = [
        {
            "start": match.start(),
            "end": match.end(),
            "text": match.group(0),
            "norm": normalize_text(match.group(0)),
        }
        for match in LOGRADOURO_TOKEN_RE.finditer(candidate["text"])
    ]
    if len(raw_tokens) < min_tokens:
        return []

    prefixes = []
    for end_idx in range(len(raw_tokens), min_tokens - 1, -1):
        last_norm = raw_tokens[end_idx - 1]["norm"]
        if last_norm in LOGRADOURO_CONNECTOR_TOKENS:
            continue
        start = candidate["start"] + raw_tokens[0]["start"]
        end = candidate["start"] + raw_tokens[end_idx - 1]["end"]
        text = candidate["full_text"][start:end]
        suffix = candidate["full_text"][end : candidate["end"]]
        prefixes.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "norm": normalize_logradouro_candidate(text),
                "suffix": suffix,
            }
        )
    return prefixes


def is_safe_logradouro_prefix(prefix: dict, candidate: dict) -> bool:
    if prefix["end"] == candidate["end"]:
        return True
    suffix = str(prefix.get("suffix", ""))
    if suffix.lstrip().startswith(","):
        return True
    if re.match(r"\s+(?:[A-ZÀ-Ú][A-Za-zÀ-ÿ]+\s+){1,3}(?i:vulgo)\b", suffix):
        return True
    return bool(re.fullmatch(r"[\s,./º°°ªªNn0-9-]+", suffix))


def extract_logradouro_candidates(text: str, *, max_name_tokens: int = 6) -> list[dict]:
    candidates = []
    marker_matches = list(LOGRADOURO_MARKER_RE.finditer(text))
    for idx, marker_match in enumerate(marker_matches):
        next_marker_start = marker_matches[idx + 1].start() if idx + 1 < len(marker_matches) else len(text)
        cursor = marker_match.end()
        token_count = 0
        end = marker_match.end()

        for token_match in LOGRADOURO_TOKEN_RE.finditer(text, cursor, next_marker_start):
            between = text[end : token_match.start()]
            if "\n" in between or ";" in between or "." in between:
                break
            token_norm = normalize_text(token_match.group(0))
            if token_norm in LOGRADOURO_STOP_TOKENS:
                break
            if token_count >= max_name_tokens:
                break
            if token_count >= 2 and re.match(
                r"(?:[A-ZÀ-Ú][A-Za-zÀ-ÿ]+\s+){2,4}(?i:vulgo)\b",
                text[token_match.start() :],
            ):
                break
            token_count += 1
            end = token_match.end()

        if token_count == 0:
            continue
        candidate_text = text[marker_match.start() : end]
        candidates.append(
            {
                "start": marker_match.start(),
                "end": end,
                "text": candidate_text,
                "full_text": text,
            }
        )
    return candidates


def find_lexicon_location_seeds(
    text: str,
    *,
    cidade: str,
    lexicon_index: dict[str, dict[str, set[str]]],
) -> tuple[list[dict], int]:
    city_key = normalize_city(cidade)
    if not city_key or city_key not in lexicon_index:
        return [], 0

    city_lexicon = lexicon_index[city_key]
    global_lexicon = lexicon_index.get("__all__")
    seeds = []
    unresolved = 0
    for candidate in extract_logradouro_candidates(text):
        accepted = None
        accepted_origin = "ibge_logradouro_lexicon"
        for prefix in candidate_prefixes(candidate):
            if not is_safe_logradouro_prefix(prefix, candidate):
                continue
            match_origin = logradouro_match_origin(prefix["norm"], city_lexicon, global_lexicon)
            if match_origin:
                accepted = prefix
                accepted_origin = match_origin
                break
        line_end_candidates = [
            pos for pos in (
                text.find("\n", candidate["start"]),
                text.find(";", candidate["start"]),
                text.find(".", candidate["start"]),
            )
            if pos >= 0
        ]
        line_end = min(line_end_candidates) if line_end_candidates else len(text)
        first_comma = text.find(",", candidate["start"], line_end)
        base_tokens = list(LOGRADOURO_TOKEN_RE.finditer(text[candidate["start"] : first_comma])) if first_comma >= 0 else []
        if first_comma >= 0 and 2 <= len(base_tokens) <= 5:
            marker_norm = canonicalize_logradouro_marker(candidate["text"].split()[0])
            comma_positions = [first_comma] + [match.start() for match in re.finditer(",", text[first_comma + 1 : line_end])]
            comma_positions = [pos if pos == first_comma else first_comma + 1 + pos for pos in comma_positions]
        else:
            comma_positions = []
        for comma_pos in comma_positions:
            item_start = comma_pos + 1
            while item_start < line_end and text[item_start].isspace():
                item_start += 1
            next_comma = text.find(",", item_start, line_end)
            item_end = next_comma if next_comma >= 0 else line_end
            item_text = text[item_start:item_end].strip()
            if not item_text:
                continue
            token_matches = []
            for token_match in LOGRADOURO_TOKEN_RE.finditer(item_text):
                token_norm = normalize_text(token_match.group(0))
                if token_norm in LOGRADOURO_STOP_TOKENS:
                    break
                token_matches.append(token_match)
            if token_matches and canonicalize_logradouro_marker(token_matches[0].group(0)) in LOGRADOURO_MARKER_ALIASES.values():
                continue
            if not token_matches or len(token_matches) > 4:
                break
            end_token = token_matches[-1]
            span_start = item_start + token_matches[0].start()
            span_end = item_start + end_token.end()
            span_text = text[span_start:span_end]
            bare_norm = simplify_logradouro_norm(span_text)
            if len(bare_norm) <= 2 or bare_norm in LOGRADOURO_CONNECTOR_TOKENS:
                break
            full_norm = simplify_logradouro_norm(f"{marker_norm} {span_text}")
            match_origin = ""
            if bare_norm in city_lexicon["bare"] or full_norm in city_lexicon["simplified"]:
                match_origin = "ibge_logradouro_lexicon_comma_continuation"
            elif global_lexicon and (bare_norm in global_lexicon["bare"] or full_norm in global_lexicon["simplified"]):
                match_origin = "ibge_logradouro_lexicon_global_comma_continuation"
            if not match_origin:
                break
            seeds.append(
                {
                    "start": span_start,
                    "end": span_end,
                    "text": span_text,
                    "label": "Location",
                    "seed_origin": match_origin,
                }
            )
        if accepted is None:
            unresolved += 1
            continue
        seeds.append(
            {
                "start": accepted["start"],
                "end": accepted["end"],
                "text": accepted["text"],
                "label": "Location",
                "seed_origin": accepted_origin,
            }
        )
    for match in BARE_LOGRADOURO_CONTEXT_RE.finditer(text):
        raw_name = match.group("name")
        name_tokens = []
        for token_match in LOGRADOURO_TOKEN_RE.finditer(raw_name):
            token_norm = normalize_text(token_match.group(0))
            if token_norm in LOGRADOURO_STOP_TOKENS:
                break
            name_tokens.append(token_match)
        if not name_tokens:
            continue
        start = match.start("name") + name_tokens[0].start()
        end = match.start("name") + name_tokens[-1].end()
        span_text = text[start:end]
        bare_norm = simplify_logradouro_norm(span_text)
        match_origin = bare_logradouro_match_origin(bare_norm, city_lexicon, global_lexicon)
        if not match_origin:
            continue
        seeds.append(
            {
                "start": start,
                "end": end,
                "text": span_text,
                "label": "Location",
                "seed_origin": f"{match_origin}_bare_context",
            }
        )
    return deduplicate_matches(seeds), unresolved


def find_city_location_seeds(text: str, cidade: str) -> list[dict]:
    pattern = flexible_literal_pattern(cidade)
    if pattern is None:
        return []
    seeds = []
    for match in pattern.finditer(text):
        seeds.append(
            {
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "label": "Location",
                "seed_origin": "metadata_literal_cidadeLocal",
            }
        )
    return deduplicate_matches(seeds)


def find_known_community_seeds(text: str, already_covered: list[dict]) -> list[dict]:
    seeds = []
    for match in KNOWN_COMMUNITIES_RE.finditer(text):
        candidate = {
            "start": match.start(),
            "end": match.end(),
            "text": match.group(0),
            "label": "Location",
            "seed_origin": "known_community_pattern",
        }
        if span_is_covered(candidate, already_covered):
            continue
        seeds.append(candidate)
    return deduplicate_matches(seeds)


def find_known_bairro_context_seeds(text: str, cidade: str, already_covered: list[dict]) -> list[dict]:
    known_bairros = KNOWN_BAIRROS_BY_CITY.get(normalize_city(cidade), set())
    if not known_bairros:
        return []

    seeds = []
    for match in BAIRRO_CONTEXT_RE.finditer(text):
        token_matches = list(LOGRADOURO_TOKEN_RE.finditer(match.group("name")))
        for end_idx in range(len(token_matches), 0, -1):
            start = match.start("name") + token_matches[0].start()
            end = match.start("name") + token_matches[end_idx - 1].end()
            span_text = text[start:end]
            if normalize_text(span_text) not in known_bairros:
                continue
            candidate = {
                "start": start,
                "end": end,
                "text": span_text,
                "label": "Location",
                "seed_origin": "known_bairro_context",
            }
            if span_is_covered(candidate, already_covered + seeds):
                break
            seeds.append(candidate)
            break
    return deduplicate_matches(seeds)


def find_logradouro_intersection_seeds(text: str, already_covered: list[dict]) -> list[dict]:
    seeds = []
    for match in LOGRADOURO_INTERSECTION_RE.finditer(text):
        for group_name in ("first", "second"):
            span_text = match.group(group_name).strip()
            if not span_text:
                continue
            start = match.start(group_name)
            end = match.end(group_name)
            candidate = {
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": "Location",
                "seed_origin": "logradouro_intersection_pattern",
            }
            if span_is_covered(candidate, already_covered + seeds):
                continue
            seeds.append(candidate)
    return deduplicate_matches(seeds)


def metadata_cluster_key(
    row: dict,
    *,
    person_seeds: list[dict] | None = None,
    org_seeds: list[dict] | None = None,
) -> str:
    parts = [part for part in (
        normalize_text(row.get("assunto", "")),
        normalize_text(row.get("cidadeLocal", "")),
        normalize_text(row.get("bairroLocal", "")),
    ) if part]
    person_parts = sorted({key for seed in person_seeds or [] if (key := cluster_person_key(seed["text"]))})
    org_parts = sorted({key for seed in org_seeds or [] if (key := cluster_org_key(seed["text"]))})
    if person_parts:
        parts.append("person=" + ",".join(person_parts))
    elif org_parts:
        parts.append("org=" + ",".join(org_parts))
    else:
        logradouro = normalize_text(row.get("logradouroLocal", ""))
        if logradouro:
            parts.append(logradouro)
    if len(parts) < 3:
        text_signature = " ".join(normalize_text(get_text(row)).split()[:80])
        parts.append(text_signature)
    return "|".join(parts)


def cluster_person_key(text: str) -> str:
    tokens = normalize_text(text).split()
    if not tokens:
        return ""
    return tokens[0]


def cluster_org_key(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"\b(\d{1,3})\s+(dp|bpm)\b", r"\1\2", normalized)
    if normalized in {"cv", "tcp", "ada", "pcc"}:
        return ""
    return normalized


def validate_entity_offset(text: str, entity: dict) -> tuple[bool, str]:
    try:
        start = int(entity["start"])
        end = int(entity["end"])
    except (KeyError, TypeError, ValueError):
        return False, "missing_or_non_integer_offset"
    if start < 0 or end <= start or end > len(text):
        return False, "out_of_bounds_offset"

    expected_text = entity.get("text")
    if isinstance(expected_text, str) and expected_text:
        actual_text = text[start:end]
        if expected_text != actual_text:
            return False, "entity_text_mismatch"
    return True, "ok"


def validated_location_entities(text: str, entities: list[dict], counters: Counter) -> list[dict] | None:
    locations = []
    for entity in entities:
        if entity.get("label") != "Location":
            continue
        valid, reason = validate_entity_offset(text, entity)
        if valid:
            locations.append(entity)
            continue

        pattern = flexible_literal_pattern(str(entity.get("text", "")))
        match = pattern.search(text) if pattern is not None else None
        if match is None:
            counters[f"dropped_location_{reason}"] += 1
            return None
        locations.append(
            {
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "label": "Location",
                "seed_origin": entity.get("seed_origin", ""),
            }
        )
    return locations


def _next_token(text: str, end: int) -> str:
    match = re.match(r"\s*([A-Za-zÀ-ÿ]+)", text[end:])
    return normalize_text(match.group(1)) if match else ""


def validate_person_seed(
    full_text: str,
    start: int,
    end: int,
    *,
    allow_single_token: bool,
    allow_lowercase_alias: bool = False,
    min_single_token_len: int = 3,
) -> tuple[bool, str]:
    if start < 0 or end <= start or end > len(full_text):
        return False, "invalid_offset"
    raw = full_text[start:end]
    if raw != raw.strip():
        return False, "edge_whitespace"
    if "\n" in raw or "\r" in raw:
        return False, "linebreak"
    if raw[:1].islower() and not allow_lowercase_alias:
        return False, "starts_lowercase"

    normalized = normalize_text(raw)
    if not normalized:
        return False, "empty"
    tokens = normalized.split()
    raw_tokens = raw.split()
    if not tokens or not raw_tokens:
        return False, "empty"
    if tokens[0] in PERSON_FORBIDDEN_START_TOKENS:
        return False, "starts_with_function_word"
    if any(piece in normalized for piece in PERSON_FORBIDDEN_SUBSTRINGS) and not allow_lowercase_alias:
        return False, "forbidden_substring"
    if any(token in PERSON_FORBIDDEN_TOKENS for token in tokens):
        return False, "forbidden_token"
    if len(tokens) > 5:
        return False, "too_long"
    if not allow_single_token and len(tokens) < 2:
        return False, "too_short"
    if allow_single_token and len(tokens) == 1 and len(tokens[0]) < min_single_token_len:
        return False, "too_short"
    if allow_single_token and len(tokens) == 1 and tokens[0] in LOGRADOURO_CONNECTOR_TOKENS:
        return False, "function_word_alias"
    if any(not token[:1].isupper() for token in raw_tokens) and not allow_lowercase_alias:
        return False, "non_title_token"
    if any(token.isupper() and normalize_text(token) in PERSON_ALL_CAPS_FORBIDDEN_TOKENS for token in raw_tokens):
        return False, "all_caps_context_token"
    previous_context = normalize_text(full_text[max(0, start - 40) : start])
    role_words = "|".join(normalize_text(word) for word in PERSON_ROLE_WORDS)
    location_words = "|".join(LOGRADOURO_MARKER_ALIASES)
    if re.search(rf"(?:^|[^a-zà-ÿ])({location_words}) ({role_words}) $", previous_context + " "):
        return False, "preceded_by_location_role"
    if _next_token(full_text, end) in PERSON_FOLLOWING_VERBS:
        return False, "followed_by_verb"
    return True, "ok"


def is_valid_person_candidate(text: str, *, allow_alias_single_token: bool) -> bool:
    raw = str(text or "").strip()
    if not raw or "\n" in raw or "\r" in raw:
        return False
    if raw[:1].islower():
        return False

    normalized = normalize_text(raw)
    if not normalized:
        return False
    tokens = normalized.split()
    if not tokens:
        return False
    if tokens[0] in PERSON_FORBIDDEN_START_TOKENS:
        return False
    if any(piece in normalized for piece in PERSON_FORBIDDEN_SUBSTRINGS):
        return False
    if allow_alias_single_token:
        return len(tokens) == 1 and len(tokens[0]) >= 3
    return len(tokens) >= 2


def is_valid_location_candidate(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or "\n" in raw or "\r" in raw:
        return False
    normalized = f" {normalize_text(raw)} "
    if any(piece in normalized for piece in LOCATION_FORBIDDEN_SUBSTRINGS):
        return False
    marker_count = sum(normalized.count(f" {marker} ") for marker in LOCATION_MARKERS)
    if marker_count > 1:
        return False
    return True


def deduplicate_matches(matches: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for match in sorted(matches, key=lambda item: (item["start"], item["end"], item["label"], item["text"])):
        key = (match["start"], match["end"], match["label"], normalize_text(match["text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def span_is_covered(candidate: dict, accepted: list[dict]) -> bool:
    for entity in accepted:
        if candidate["start"] < entity["end"] and entity["start"] < candidate["end"]:
            return True
    return False


def prefer_longest_non_overlapping_spans(matches: list[dict]) -> list[dict]:
    accepted = []
    for match in sorted(matches, key=lambda item: (-(item["end"] - item["start"]), item["start"], item["end"])):
        if span_is_covered(match, accepted):
            continue
        accepted.append(match)
    return sorted(accepted, key=lambda item: (item["start"], item["end"], item["label"], item["text"]))


def find_weak_person_signals(text: str) -> list[dict]:
    matches = []
    for rule_name, pattern in WEAK_PERSON_PATTERNS.items():
        for match in pattern.finditer(text):
            name = match.group("name")
            valid, reason = validate_person_seed(
                text,
                match.start("name"),
                match.end("name"),
                allow_single_token=True,
            )
            if not valid:
                continue
            matches.append(
                {
                    "start": match.start("name"),
                    "end": match.end("name"),
                    "text": name,
                    "label": "Person",
                    "seed_origin": rule_name,
                }
            )
    return deduplicate_matches(matches)


def has_dense_alias_signal(text: str) -> bool:
    normalized = normalize_text(text)
    return sum(normalized.count(marker) for marker in PERSON_ALIAS_MARKERS) >= 3


def uncovered_person_signal_count(text: str, accepted_person_seeds: list[dict]) -> int:
    weak_signals = find_weak_person_signals(text)
    uncovered = [signal for signal in weak_signals if not span_is_covered(signal, accepted_person_seeds)]
    if any(
        signal["seed_origin"] in {"weak_person_after_action", "weak_person_victim_context"}
        for signal in uncovered
    ):
        return max(len(uncovered), 3)
    if has_dense_alias_signal(text) and uncovered:
        return max(len(uncovered), 3)
    return len(uncovered)


def add_person_match(
    matches: list[dict],
    rejection_counts: Counter,
    *,
    text: str,
    start: int,
    end: int,
    label_text: str,
    seed_origin: str,
    allow_single_token: bool,
    allow_lowercase_alias: bool = False,
    min_single_token_len: int = 3,
) -> bool:
    valid, reason = validate_person_seed(
        text,
        start,
        end,
        allow_single_token=allow_single_token,
        allow_lowercase_alias=allow_lowercase_alias,
        min_single_token_len=min_single_token_len,
    )
    if not valid:
        rejection_counts[f"person_rejected_{reason}"] += 1
        return False
    matches.append(
        {
            "start": start,
            "end": end,
            "text": label_text,
            "label": "Person",
            "seed_origin": seed_origin,
        }
    )
    return True


def first_token_is_prenome(span_text: str, prenomes: set[str]) -> bool:
    tokens = normalize_text(span_text).split()
    return bool(tokens and tokens[0] in prenomes)


def has_strong_prenome_context(text: str, start: int, end: int) -> bool:
    before = normalize_text(text[max(0, start - 32) : start])
    after = text[end : end + 48]
    return bool(
        PERSON_LEXICON_PRE_CONTEXT_RE.search(before + " ")
        or PERSON_LEXICON_POST_CONTEXT_RE.search(after)
    )


def find_prenome_lexicon_person_seeds(
    text: str,
    prenomes: set[str],
    rejection_counts: Counter,
) -> list[dict]:
    if not prenomes:
        return []

    matches = []
    for match in PERSON_LEXICON_LIST_RE.finditer(text):
        for group_name in ("n1", "n2"):
            span_text = match.group(group_name)
            if not first_token_is_prenome(span_text, prenomes):
                continue
            add_person_match(
                matches,
                rejection_counts,
                text=text,
                start=match.start(group_name),
                end=match.end(group_name),
                label_text=span_text,
                seed_origin="person_prenome_lexicon_list",
                allow_single_token=False,
            )

    for match in PERSON_LEXICON_FULL_NAME_RE.finditer(text):
        span_text = match.group("fullname")
        if not first_token_is_prenome(span_text, prenomes):
            continue
        if not has_strong_prenome_context(text, match.start("fullname"), match.end("fullname")):
            continue
        add_person_match(
            matches,
            rejection_counts,
            text=text,
            start=match.start("fullname"),
            end=match.end("fullname"),
            label_text=span_text,
            seed_origin="person_prenome_lexicon_context",
            allow_single_token=False,
        )
    return deduplicate_matches(matches)


def find_person_seeds(
    text: str,
    rejection_counts: Counter | None = None,
    prenomes: set[str] | None = None,
) -> list[dict]:
    matches = []
    if rejection_counts is None:
        rejection_counts = Counter()
    for rule_name, pattern in PERSON_PATTERNS.items():
        for match in pattern.finditer(text):
            groups = match.groupdict()
            person_added = True
            if "fullname" in groups:
                fullname = groups["fullname"]
                fullname_start = match.start("fullname")
                fullname_end = match.end("fullname")
                fullname_text = fullname
                raw_fullname_tokens = list(re.finditer(r"\S+", fullname))
                if (
                    raw_fullname_tokens
                    and normalize_text(raw_fullname_tokens[0].group(0)) in PERSON_STRIPPABLE_PREFIX_TOKENS
                    and len(raw_fullname_tokens) >= 3
                ):
                    fullname_start += raw_fullname_tokens[1].start()
                    fullname_text = text[fullname_start:fullname_end]
                person_added = add_person_match(
                    matches,
                    rejection_counts,
                    text=text,
                    start=fullname_start,
                    end=fullname_end,
                    label_text=fullname_text,
                    seed_origin=rule_name,
                    allow_single_token=rule_name == "person_known_as",
                )
            if "name" in groups:
                person_name = groups["name"]
                person_start = match.start("name")
                alias_name_rule = rule_name in {"person_article_vulgo_alias", "person_apelido_alias"}
                if rule_name == "person_firstname_vulgo_alias" and re.search(
                    rf"{NAME_TOKEN}[ \t]+$",
                    text[max(0, person_start - 40) : person_start],
                ):
                    person_added = False
                elif (
                    rule_name == "person_firstname_vulgo_alias"
                    and normalize_text(person_name) in PERSON_FORBIDDEN_SINGLE_ALIAS_NAMES
                ):
                    person_added = False
                else:
                    person_added = add_person_match(
                        matches,
                        rejection_counts,
                        text=text,
                        start=person_start,
                        end=match.end("name"),
                        label_text=person_name,
                        seed_origin=rule_name,
                        allow_single_token=True,
                        allow_lowercase_alias=alias_name_rule,
                        min_single_token_len=2 if alias_name_rule else 3,
                    )
            if "alias" in groups and person_added:
                alias = groups["alias"]
                add_person_match(
                    matches,
                    rejection_counts,
                    text=text,
                    start=match.start("alias"),
                    end=match.end("alias"),
                    label_text=alias,
                    seed_origin=f"{rule_name}_alias",
                    allow_single_token=True,
                    allow_lowercase_alias=True,
                    min_single_token_len=2,
                )
    for rule_name, pattern in PERSON_COORD_PATTERNS.items():
        for match in pattern.finditer(text):
            for group_name in ("n1", "n2"):
                person_name = match.group(group_name)
                if not person_name:
                    continue
                add_person_match(
                    matches,
                    rejection_counts,
                    text=text,
                    start=match.start(group_name),
                    end=match.end(group_name),
                    label_text=person_name,
                    seed_origin=rule_name,
                    allow_single_token=False,
                )
    if prenomes:
        matches.extend(find_prenome_lexicon_person_seeds(text, prenomes, rejection_counts))
    return prefer_longest_non_overlapping_spans(deduplicate_matches(matches))


def find_org_seeds(text: str) -> list[dict]:
    matches = []
    for rule_name, pattern in ORG_PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "label": "Organization",
                    "seed_origin": rule_name,
                }
            )
    return deduplicate_matches(matches)


def overlaps(existing: list[dict], candidate: dict) -> bool:
    for entity in existing:
        if candidate["start"] < entity["end"] and entity["start"] < candidate["end"]:
            return True
    return False


def raw_without_first_token(text: str) -> str:
    match = re.match(r"\s*\S+\s+(?P<rest>.+?)\s*$", text or "")
    return match.group("rest") if match else ""


def is_known_single_token_location(entity: dict) -> bool:
    origin = entity.get("seed_origin", "")
    return origin in {
        "metadata_literal_bairroLocal",
        "known_community_pattern",
        "known_bairro_context",
    }


def should_propagate_entity(entity: dict) -> bool:
    normalized = normalize_text(entity.get("text", ""))
    if not normalized:
        return False
    tokens = normalized.split()
    label = entity.get("label")
    origin = entity.get("seed_origin", "")
    if label == "Location":
        if len(tokens) >= 2:
            return True
        return len(normalized) >= 5 and is_known_single_token_location(entity)
    if label == "Person":
        if len(tokens) >= 2:
            return True
        return ("alias" in origin or origin in {"person_article_vulgo_alias", "person_apelido_alias"}) and len(normalized) >= 3
    return False


def propagation_variants(entity: dict) -> list[str]:
    text = str(entity.get("text", "")).strip()
    if not text:
        return []
    variants = [text]
    if entity.get("label") == "Location":
        tokens = normalize_text(text).split()
        if tokens and canonicalize_logradouro_marker(tokens[0]) in LOGRADOURO_MARKER_ALIASES.values():
            bare = raw_without_first_token(text)
            if len(normalize_text(bare).split()) >= 2:
                variants.append(bare)
    return sorted({variant for variant in variants if variant.strip()}, key=len, reverse=True)


def propagate_repeated_entities(text: str, entities: list[dict]) -> list[dict]:
    accepted = list(entities)
    variants = []
    for entity in entities:
        if entity.get("label") not in {"Location", "Person"}:
            continue
        if not should_propagate_entity(entity):
            continue
        for variant in propagation_variants(entity):
            variants.append((entity["label"], variant))

    seen_variants = set()
    for label, variant in sorted(variants, key=lambda item: len(item[1]), reverse=True):
        key = (label, normalize_text(variant))
        if key in seen_variants:
            continue
        seen_variants.add(key)
        pattern = accent_flexible_literal_pattern(variant)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            candidate = {
                "start": match.start(),
                "end": match.end(),
                "text": text[match.start() : match.end()],
                "label": label,
                "seed_origin": "propagated_same_record",
            }
            if overlaps(accepted, candidate):
                continue
            accepted.append(candidate)

    return sorted(accepted, key=lambda item: (item["start"], item["end"], item["label"]))


def merge_entities(
    text: str,
    base_entities: list[dict],
    lexicon_location_seeds: list[dict],
    person_seeds: list[dict],
    org_seeds: list[dict],
) -> list[dict]:
    merged = []
    for entity in sorted(base_entities, key=lambda item: (item["start"], item["end"], item.get("label", ""))):
        entity_text = text[int(entity["start"]) : int(entity["end"])]
        if str(entity.get("label")) == "Location" and not is_valid_location_candidate(entity_text):
            continue
        normalized = {
            "start": int(entity["start"]),
            "end": int(entity["end"]),
            "text": entity_text,
            "label": str(entity["label"]),
            "seed_origin": entity.get("seed_origin", ""),
        }
        merged.append(normalized)

    for seed in sorted(lexicon_location_seeds, key=lambda item: (item["start"], item["end"], item["label"])):
        if overlaps(merged, seed):
            continue
        merged.append(seed)

    for seed in sorted(person_seeds + org_seeds, key=lambda item: (item["start"], item["end"], item["label"])):
        if overlaps(merged, seed):
            continue
        merged.append(seed)

    merged.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    return propagate_repeated_entities(text, merged)


def score_row(row: dict, person_count: int, org_count: int) -> float:
    score = float(row.get("_pilot_meta", {}).get("score", 0.0))
    assunto = str(row.get("assunto", "")).strip()
    cidade = normalize_text(row.get("cidadeLocal", ""))
    lex_matches = row.get("_pilot_meta", {}).get("lexicon_matches", []) or []

    if assunto == "Homicídios":
        score += 0.25
    elif assunto == "Roubos em Geral":
        score += 0.10
    if cidade == "rio de janeiro":
        score += 0.15
    if lex_matches:
        score += 0.10

    score += min(person_count, 3) * 0.20
    score += min(org_count, 3) * 0.15
    return score


def build_multilabel_pool(
    rows: list[dict],
    *,
    require_two_location_seeds: bool,
    min_person_seeds: int,
    min_organization_seeds: int,
    drop_incomplete_person_signal: bool,
    max_uncovered_person_signals: int,
    max_rows_per_cluster: int,
    logradouro_lexicon_index: dict[str, dict[str, set[str]]] | None,
    enable_lexicon_location_expansion: bool,
    max_uncovered_logradouro_signals: int,
    prenomes_lexicon: set[str] | None,
    enable_prenome_person_expansion: bool,
) -> tuple[list[dict], dict]:
    kept = []
    counters = Counter()
    person_rule_counts = Counter()
    org_rule_counts = Counter()
    person_rejection_counts = Counter()
    lexicon_location_counts = Counter()

    for row in rows:
        text = normalize_relato_text(get_text(row))
        if not text:
            counters["dropped_missing_text"] += 1
            continue

        base_entities = get_entities(row)
        location_entities = validated_location_entities(text, base_entities, counters)
        if location_entities is None:
            continue
        if require_two_location_seeds and len(location_entities) != 2:
            counters["dropped_location_seed_count"] += 1
            continue
        if not location_entities:
            counters["dropped_missing_location"] += 1
            continue

        lexicon_location_seeds = []
        uncovered_logradouro_signals = 0
        if enable_lexicon_location_expansion and logradouro_lexicon_index:
            lexicon_location_seeds, uncovered_logradouro_signals = find_lexicon_location_seeds(
                text,
                cidade=str(row.get("cidadeLocal", "")),
                lexicon_index=logradouro_lexicon_index,
            )
            lexicon_location_seeds.extend(find_city_location_seeds(text, str(row.get("cidadeLocal", ""))))
            lexicon_location_seeds.extend(
                find_known_community_seeds(text, location_entities + lexicon_location_seeds)
            )
            lexicon_location_seeds.extend(
                find_known_bairro_context_seeds(
                    text,
                    str(row.get("cidadeLocal", "")),
                    location_entities + lexicon_location_seeds,
                )
            )
            lexicon_location_seeds.extend(
                find_logradouro_intersection_seeds(text, location_entities + lexicon_location_seeds)
            )
            lexicon_location_seeds = deduplicate_matches(lexicon_location_seeds)
            if max_uncovered_logradouro_signals >= 0 and uncovered_logradouro_signals > max_uncovered_logradouro_signals:
                counters["dropped_uncovered_logradouro_signal"] += 1
                continue

        person_seeds = find_person_seeds(
            text,
            person_rejection_counts,
            prenomes=prenomes_lexicon if enable_prenome_person_expansion else None,
        )
        person_seeds = [
            seed
            for seed in person_seeds
            if not span_is_covered(seed, location_entities + lexicon_location_seeds)
        ]
        org_seeds = find_org_seeds(text)
        uncovered_person_signals = uncovered_person_signal_count(text, person_seeds)
        if drop_incomplete_person_signal:
            if uncovered_person_signals > max_uncovered_person_signals:
                counters["dropped_incomplete_person_signal"] += 1
                continue
        if len(person_seeds) < min_person_seeds:
            counters["dropped_missing_person"] += 1
            continue
        if len(org_seeds) < min_organization_seeds:
            counters["dropped_missing_organization"] += 1
            continue

        merged_entities = merge_entities(text, location_entities, lexicon_location_seeds, person_seeds, org_seeds)
        merged_location_count = sum(1 for entity in merged_entities if entity["label"] == "Location")
        if require_two_location_seeds and merged_location_count < 2:
            counters["dropped_filtered_location_seed_count"] += 1
            continue
        if not merged_location_count:
            counters["dropped_location_filtered"] += 1
            continue
        if not any(entity["label"] == "Person" for entity in merged_entities):
            counters["dropped_person_overlap"] += 1
            continue
        if not any(entity["label"] == "Organization" for entity in merged_entities):
            counters["dropped_organization_overlap"] += 1
            continue

        for seed in person_seeds:
            person_rule_counts[seed["seed_origin"]] += 1
        for seed in org_seeds:
            org_rule_counts[seed["seed_origin"]] += 1
        for seed in lexicon_location_seeds:
            if any(entity["start"] == seed["start"] and entity["end"] == seed["end"] and entity["label"] == "Location" for entity in merged_entities):
                if seed.get("seed_origin") == "metadata_literal_cidadeLocal":
                    lexicon_location_counts["accepted_metadata_city_location"] += 1
                elif seed.get("seed_origin") == "ibge_logradouro_lexicon_bare_context":
                    lexicon_location_counts["accepted_ibge_bare_context_location"] += 1
                elif seed.get("seed_origin") == "ibge_logradouro_lexicon_global_bare_context":
                    lexicon_location_counts["accepted_ibge_global_bare_context_location"] += 1
                elif seed.get("seed_origin") == "ibge_logradouro_lexicon_comma_continuation":
                    lexicon_location_counts["accepted_ibge_comma_continuation_location"] += 1
                elif seed.get("seed_origin") == "ibge_logradouro_lexicon_global_comma_continuation":
                    lexicon_location_counts["accepted_ibge_global_comma_continuation_location"] += 1
                elif seed.get("seed_origin") == "known_community_pattern":
                    lexicon_location_counts["accepted_known_community_location"] += 1
                elif seed.get("seed_origin") == "known_bairro_context":
                    lexicon_location_counts["accepted_known_bairro_context_location"] += 1
                elif seed.get("seed_origin") == "logradouro_intersection_pattern":
                    lexicon_location_counts["accepted_logradouro_intersection_location"] += 1
                elif seed.get("seed_origin") == "ibge_logradouro_lexicon_global":
                    lexicon_location_counts["accepted_ibge_global_logradouro_location"] += 1
                else:
                    lexicon_location_counts["accepted_ibge_logradouro_location"] += 1

        enriched = dict(row)
        enriched["text"] = text
        enriched["entities"] = merged_entities
        enriched["_multilabel_meta"] = {
            "cluster_key": metadata_cluster_key(row, person_seeds=person_seeds, org_seeds=org_seeds),
            "person_seed_count": len(person_seeds),
            "organization_seed_count": len(org_seeds),
            "lexicon_location_seed_count": len(lexicon_location_seeds),
            "uncovered_person_signal_count": uncovered_person_signals,
            "uncovered_logradouro_signal_count": uncovered_logradouro_signals,
            "person_seed_rules": sorted({seed["seed_origin"] for seed in person_seeds}),
            "organization_seed_rules": sorted({seed["seed_origin"] for seed in org_seeds}),
            "selection_score": score_row(row, len(person_seeds), len(org_seeds)),
        }
        kept.append(enriched)

    kept.sort(
        key=lambda row: (
            -float(row["_multilabel_meta"]["selection_score"]),
            -int(row["_multilabel_meta"]["person_seed_count"]),
            -int(row["_multilabel_meta"]["organization_seed_count"]),
            row.get("source_id", ""),
        )
    )

    if max_rows_per_cluster > 0:
        clustered = []
        cluster_counts = Counter()
        for row in kept:
            cluster_key = row["_multilabel_meta"]["cluster_key"]
            if cluster_counts[cluster_key] >= max_rows_per_cluster:
                counters["dropped_cluster_limit"] += 1
                continue
            cluster_counts[cluster_key] += 1
            clustered.append(row)
        kept = clustered

    summary = {
        "rows_seen": len(rows),
        "rows_kept": len(kept),
        "dropped_counts": dict(counters),
        "person_rejection_counts": dict(person_rejection_counts),
        "lexicon_location_counts": dict(lexicon_location_counts),
        "person_rule_counts": dict(person_rule_counts),
        "organization_rule_counts": dict(org_rule_counts),
        "assunto_counts": dict(Counter(str(row.get("assunto", "")).strip() for row in kept)),
        "cidade_counts": dict(Counter(str(row.get("cidadeLocal", "")).strip() for row in kept).most_common(20)),
        "cluster_counts": dict(Counter(row["_multilabel_meta"]["cluster_key"] for row in kept).most_common(20)),
        "label_counts_kept": dict(Counter(entity["label"] for row in kept for entity in row["entities"])),
    }
    return kept, summary


def project_for_refit(row: dict) -> dict:
    return {
        "source_id": row.get("source_id", ""),
        "text": get_text(row),
        "entities": [
            {
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["text"],
                "label": entity["label"],
                "seed_origin": entity.get("seed_origin", ""),
            }
            for entity in row.get("entities", [])
        ],
    }


def main():
    args = parse_args()
    rows = read_json_or_jsonl(args.input)
    logradouro_lexicon_index = load_logradouro_lexicon(args.logradouro_lexicon_jsonl) if args.logradouro_lexicon_jsonl else {}
    prenomes_lexicon = load_prenomes_lexicon(args.prenomes_lexicon) if args.prenomes_lexicon else set()
    kept, summary = build_multilabel_pool(
        rows,
        require_two_location_seeds=args.require_two_location_seeds,
        min_person_seeds=args.min_person_seeds,
        min_organization_seeds=args.min_organization_seeds,
        drop_incomplete_person_signal=args.drop_incomplete_person_signal,
        max_uncovered_person_signals=args.max_uncovered_person_signals,
        max_rows_per_cluster=args.max_rows_per_cluster,
        logradouro_lexicon_index=logradouro_lexicon_index,
        enable_lexicon_location_expansion=args.enable_lexicon_location_expansion,
        max_uncovered_logradouro_signals=args.max_uncovered_logradouro_signals,
        prenomes_lexicon=prenomes_lexicon,
        enable_prenome_person_expansion=args.enable_prenome_person_expansion,
    )

    review_rows = kept[: args.top_n]
    pseudolabel_rows = [project_for_refit(row) for row in review_rows]

    write_jsonl(args.output_pool_jsonl, kept)
    write_jsonl(args.output_review_jsonl, review_rows)
    write_jsonl(args.output_pseudolabel_jsonl, pseudolabel_rows)

    payload = {
        "input": str(Path(args.input).resolve()),
        "output_pool_jsonl": str(Path(args.output_pool_jsonl).resolve()),
        "output_review_jsonl": str(Path(args.output_review_jsonl).resolve()),
        "output_pseudolabel_jsonl": str(Path(args.output_pseudolabel_jsonl).resolve()),
        "top_n": args.top_n,
        "require_two_location_seeds": args.require_two_location_seeds,
        "min_person_seeds": args.min_person_seeds,
        "min_organization_seeds": args.min_organization_seeds,
        "drop_incomplete_person_signal": args.drop_incomplete_person_signal,
        "max_uncovered_person_signals": args.max_uncovered_person_signals,
        "max_rows_per_cluster": args.max_rows_per_cluster,
        "logradouro_lexicon_jsonl": str(Path(args.logradouro_lexicon_jsonl).resolve()) if args.logradouro_lexicon_jsonl else "",
        "enable_lexicon_location_expansion": args.enable_lexicon_location_expansion,
        "max_uncovered_logradouro_signals": args.max_uncovered_logradouro_signals,
        "logradouro_lexicon_cities": len([city for city in logradouro_lexicon_index if city != "__all__"]),
        "logradouro_lexicon_entries": sum(
            len(entries.get("canonical", set()))
            for city, entries in logradouro_lexicon_index.items()
            if city != "__all__"
        ),
        "prenomes_lexicon": str(Path(args.prenomes_lexicon).resolve()) if args.prenomes_lexicon else "",
        "enable_prenome_person_expansion": args.enable_prenome_person_expansion,
        "prenomes_lexicon_entries": len(prenomes_lexicon),
        **summary,
        "review_rows": len(review_rows),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Rows seen: {payload['rows_seen']}")
    print(f"Rows kept: {payload['rows_kept']}")
    print(f"Review rows: {payload['review_rows']}")
    print(f"Pool JSONL: {args.output_pool_jsonl}")
    print(f"Review JSONL: {args.output_review_jsonl}")
    print(f"Pseudolabel JSONL: {args.output_pseudolabel_jsonl}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
