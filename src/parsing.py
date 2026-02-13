"""Parse EU AI Act XHTML into structured Provision objects."""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from lxml import etree


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_LANG = "de"

LABELS = {
    "de": {"recital": "Erwägungsgrund", "article": "Artikel", "annex": "ANHANG"},
    "en": {"recital": "Recital", "article": "Article", "annex": "ANNEX"},
}


@dataclass
class Provision:
    id: str
    type: str  # "article", "recital", "annex"
    number: str  # "1", "2", "I", "II", etc.
    title: str
    text: str
    chapter: str = ""
    chapter_title: str = ""
    paragraphs: list[dict] = field(default_factory=list)
    lang: str = DEFAULT_LANG


def _get_text(el) -> str:
    """Extract cleaned text from an element and its descendants."""
    raw = " ".join(el.itertext())
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", raw).strip()
    # Remove stray backticks from source XHTML
    return cleaned.replace("`", "")


def _get_chapter_map(root) -> dict[str, tuple[str, str]]:
    """Build mapping from article_id -> (chapter_number, chapter_title)."""
    chapter_map = {}
    chapters = root.xpath('//*[starts-with(@id, "cpt_") and not(contains(@id, "."))]')
    for ch in chapters:
        ch_id = ch.get("id")
        ch_num = ch_id.replace("cpt_", "")

        # Get chapter title from the eli-title div
        title_el = ch.xpath('.//*[contains(@class, "eli-title")]')
        ch_title = _get_text(title_el[0]) if title_el else ""

        # Find all top-level articles in this chapter
        articles = ch.xpath('.//*[starts-with(@id, "art_")]')
        for art in articles:
            art_id = art.get("id")
            if re.match(r"^art_\d+$", art_id):
                chapter_map[art_id] = (ch_num, ch_title)

    return chapter_map


def _parse_article(el, chapter_map: dict, lang: str = DEFAULT_LANG) -> Provision:
    art_id = el.get("id")
    number = art_id.replace("art_", "")

    # Title from oj-ti-art element
    ti_el = el.xpath('.//*[contains(@class, "oj-ti-art")]')
    fallback_label = LABELS.get(lang, LABELS["de"])["article"]
    art_label = _get_text(ti_el[0]) if ti_el else f"{fallback_label} {number}"

    # Subtitle from eli-title
    title_el = el.xpath('.//*[contains(@class, "eli-title")]')
    subtitle = _get_text(title_el[0]) if title_el else ""

    title = f"{art_label} — {subtitle}" if subtitle else art_label

    # Paragraphs: divs with numeric IDs like 001.001
    paragraphs = []
    for child in el:
        child_id = child.get("id") or ""
        if re.match(r"\d{3}\.\d{3}", child_id):
            para_text = _get_text(child)
            paragraphs.append({"id": child_id, "text": para_text})

    full_text = "\n\n".join(p["text"] for p in paragraphs) if paragraphs else _get_text(el)

    ch_num, ch_title = chapter_map.get(art_id, ("", ""))

    return Provision(
        id=art_id,
        type="article",
        number=number,
        title=title,
        text=full_text,
        chapter=ch_num,
        chapter_title=ch_title,
        paragraphs=paragraphs,
        lang=lang,
    )


def _parse_recital(el, lang: str = DEFAULT_LANG) -> Provision:
    rec_id = el.get("id")
    number = rec_id.replace("rct_", "")
    text = _get_text(el)

    # Strip leading "(N)" pattern from text
    text_clean = re.sub(r"^\(\d+\)\s*", "", text)

    label = LABELS.get(lang, LABELS["de"])["recital"]
    return Provision(
        id=rec_id,
        type="recital",
        number=number,
        title=f"{label} {number}",
        text=text_clean,
        lang=lang,
    )


def _parse_annex(el, lang: str = DEFAULT_LANG) -> Provision:
    anx_id = el.get("id")
    number = anx_id.replace("anx_", "")

    # Title from first oj-doc-ti
    title_els = el.xpath('.//p[contains(@class, "oj-doc-ti")]')
    titles = [_get_text(t) for t in title_els[:2]]
    title = " — ".join(t for t in titles if t)

    text = _get_text(el)

    return Provision(
        id=anx_id,
        type="annex",
        number=number,
        title=title,
        text=text,
        lang=lang,
    )


def parse_eu_ai_act(path: str | Path | None = None, lang: str = DEFAULT_LANG) -> list[Provision]:
    """Parse EU AI Act XHTML file and return list of Provisions.

    If path is None, resolves to data/<lang> relative to project root.
    """
    if path is None:
        path = DATA_DIR / lang

    parser = etree.HTMLParser(encoding="utf-8")
    tree = etree.parse(str(path), parser)
    root = tree.getroot()

    chapter_map = _get_chapter_map(root)
    provisions = []

    # Articles
    for el in root.iter():
        el_id = el.get("id") or ""
        if re.match(r"^art_\d+$", el_id):
            provisions.append(_parse_article(el, chapter_map, lang))

    # Recitals
    for el in root.iter():
        el_id = el.get("id") or ""
        if re.match(r"^rct_\d+$", el_id):
            provisions.append(_parse_recital(el, lang))

    # Annexes
    for el in root.iter():
        el_id = el.get("id") or ""
        if re.match(r"^anx_[A-Z]+$", el_id):
            provisions.append(_parse_annex(el, lang))

    return provisions


def save_provisions(provisions: list[Provision], path: str | Path) -> None:
    """Save provisions to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(p) for p in provisions], f, indent=2, ensure_ascii=False)


def load_provisions(path: str | Path) -> list[Provision]:
    """Load provisions from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [Provision(**d) for d in data]
