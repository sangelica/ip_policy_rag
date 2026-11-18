import pymupdf
import re

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CONFIG, PROJECT_ROOT, get_embeddings
from src.vector_store import get_vector_store

RAW_DIR = PROJECT_ROOT / CONFIG["ingest"]["raw_dir"]
OCR_DIR = PROJECT_ROOT / CONFIG["ingest"]["ocr_dir"]
FILE = RAW_DIR / "PC2024-0013.pdf"

OCR_MIN_CHARS = 50  # fewer characters than this = no usable text layer (scanned page)

# abbreviation -> full form, from config.yaml (empty dict if the section is missing)
ABBREVIATIONS: dict[str, str] = CONFIG["ingest"].get("abbreviations") or {}


def _build_abbrev_pattern(abbrevs: dict[str, str]) -> re.Pattern | None:
    """One regex matching every abbreviation and every full form.

    Each entry gets two named groups, f<i> (full form) and a<i> (abbreviation), so the
    replacement knows which entry matched. Longest alternatives go first, so "Offices"
    is tried before "Office" and "PROs" before "PRO".
    """
    if not abbrevs:
        return None
    alts = []
    for i, (abbr, full) in enumerate(abbrevs.items()):
        a = re.escape(abbr)
        # Full form: any whitespace between words (line breaks, double spaces), any
        # letter case, optionally already followed by "(ABBR)" so it isn't doubled.
        f = r"\s+".join(re.escape(word) for word in full.split())
        alts.append((len(full), rf"(?P<f{i}>\b(?i:{f})\b(?:\s*\({a}\))?)"))
        # Abbreviation: whole word, case-sensitive (so the word "pro" is left alone).
        alts.append((len(abbr), rf"(?P<a{i}>\b{a}\b)"))
    alts.sort(key=lambda x: x[0], reverse=True)
    return re.compile("|".join(pattern for _, pattern in alts))


ABBREV_PATTERN = _build_abbrev_pattern(ABBREVIATIONS)
ABBREV_ITEMS = list(ABBREVIATIONS.items())


def expand_abbreviations(text: str) -> str:
    """Write every abbreviation and full form as "Full Form (ABBR)".

    PRO -> PhilHealth Regional Office (PRO); PhilHealth Regional Offices -> ... (PROs).
    Done in a single pass, so replaced text is never matched again.
    """
    if ABBREV_PATTERN is None:
        return text

    def replace(m: re.Match) -> str:
        abbr, full = ABBREV_ITEMS[int(m.lastgroup[1:])]   # "f1" / "a1" -> entry 1
        return f"{full} ({abbr})"

    return ABBREV_PATTERN.sub(replace, text)


# Annex pages carry a footer like "Page 2 of 4 of Annex A"; main pages say "Page 3 of 8".
ANNEX_FOOTER = re.compile(r"Page\s+\d+\s+of\s+\d+\s+of\s+Annex\b", re.IGNORECASE)
# The first page of an annex starts with its heading: "Annex A: ..." / "Annex I; ..." (OCR often reads ":" as ";").
ANNEX_HEADING = re.compile(r"\s*Annex\s+(?:[A-Z]|[IVX]+)\b\s*[:;.\-–]", re.IGNORECASE)


def doc_part(raw_text: str) -> str:
    """Return "annex" or "main" for one page, from its raw (uncleaned) text.

    Only the footer and the very start of the page are checked: the main body also
    mentions annexes ("...written notice (Annex A; Notice of ...)"), so searching the
    whole page would mislabel it. Pages that can't be identified default to "main".
    """
    if ANNEX_FOOTER.search(raw_text) or ANNEX_HEADING.match(raw_text):
        return "annex"
    return "main"


def page_text(page: pymupdf.Page, cache_dir: Path) -> tuple[str, bool]:
    """Return (text, used_ocr) for one page.

    Uses the PDF's text layer when it has one. Otherwise runs Tesseract OCR and caches
    the result as cache_dir/pNN.txt, so OCR runs once per page, not on every ingest.
    """
    cache_file = cache_dir / f"p{page.number + 1:02d}.txt"
    if cache_file.exists():                      # only OCR'd pages are ever cached
        return cache_file.read_text(encoding="utf-8"), True

    text = page.get_text("text")
    if len(text.strip()) >= OCR_MIN_CHARS:
        return text, False

    tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
    text = page.get_text("text", textpage=tp)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text, True


""" Ingest pdf file"""
def ingest(vector_store: Chroma, file=FILE):
    cache_dir = OCR_DIR / Path(file).stem          # data/ocr/PC2024-0013/
    with pymupdf.open(file) as pdf_file:

        # Built once per file, not per page; sizes come from config.yaml so evals record them.
        if not CONFIG["ingest"]["list_aware_split"]:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CONFIG["ingest"]["chunk_size"],
                chunk_overlap=CONFIG["ingest"]["chunk_overlap"],
            )
        else:

            # Try to cut at paragraphs, then numbered items, then lettered items, then sentences
            # List aware chunking so that middle of an item in list is not cut off
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CONFIG["ingest"]["chunk_size"],
                chunk_overlap=CONFIG["ingest"]["chunk_overlap"],
                separators=[
                    r"\n\n",            # paragraph break
                    r"(?=\s\d+\.\s)",   # before " 3. "  (numbered item)
                    r"(?=\s[a-z]\.\s)", # before " c. "  (lettered sub-item)
                    r"(?<=\.)\s",       # after a full stop (sentence end)
                    r"\s",              # any space
                    "",                 # last resort: any character
                ],
                is_separator_regex=True,
            )

        for page_num in range(len(pdf_file)):
            page = page_num + 1  # PyMuPDF is 0-based; citations use human page numbers
            text, used_ocr = page_text(pdf_file[page_num], cache_dir)

            if not text.strip():  # blank page, or OCR found nothing: nothing to embed
                print(f"skipped page {page}: no text")
                continue

            part = doc_part(text)  # detect before cleanup, while footer/heading line breaks are intact

            #re.sub(r"\s*\n\s*", " ", text)
            text = re.sub(r"[ \t]*\n[ \t]*\n\s*", "\n\n", text)   # 1. tidy paragraph breaks to exactly "\n\n"
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)          # 2. single newline → space
            text = expand_abbreviations(text)                     # 3. "PRO" / full form → "Full Form (PRO)"
            chunks = splitter.split_text(text)

            docs = []
            ids = []
            for i, chunk in enumerate(chunks):

                print("==============")
                print(chunk)

                metadata = {}
                metadata["source"] = Path(file).name
                metadata["page"] = page
                metadata["chunk"] = i
                metadata["ocr"] = used_ocr  # OCR text has character errors: trust it less
                metadata["doc_part"] = part  # "annex" or "main"
                doc = Document(page_content = chunk, metadata=metadata)
                ids.append(f"{Path(file).stem}-p{page}-c{i}")
                docs.append(doc)

            vector_store.add_documents(docs, ids=ids)


# Runs only with `python -m src.ingest`, not when another module imports this file.
if __name__ == "__main__":

    vector_store = get_vector_store()

    for pdf in sorted(RAW_DIR.glob("*.pdf")):    # every PDF in data/raw, not just one
        if pdf.name != "PC2024-0013.pdf": # skip ocr for now
            print(f"ingesting {pdf.name}")
            ingest(vector_store, pdf)

    print(len(vector_store.get()["ids"]))

#store = get_vector_store()
    for part in ("main", "annex"):
        pages = sorted({m["page"] for m in vector_store.get(where={"$and": [{"source": "PC2026-0002.pdf"}, {"doc_part": part}]})["metadatas"]})
        print(part, pages)
