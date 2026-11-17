from src.state_machine import app as IP_AGENT
from src.config import PROJECT_ROOT
from pathlib import Path

import json, time, pytest
import re


# Curly quotes and dashes from the PDF → plain ASCII, so quotes match what the model writes.
_PUNCT = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})

def compact(text: str) -> str:
    """normalize() with all spaces removed. For quote checks: PDF text has breaks
    like "Pre mium" that the model may quote as "Premium"."""
    return normalize(text).replace(" ", "")

def contains(phrase: str, text: str) -> bool:
    """Normalised substring match that won't match a number inside a bigger number."""
    p, t = normalize(phrase), normalize(text)
    return re.search(rf"(?<!\d){re.escape(p)}(?!\d)", t) is not None

def normalize(text: str) -> str:
    """Lowercase and strip formatting so equivalent answers compare equal.

    "₱100,000.00", "P100,000", "PHP 100,000" -> "100000"
    "5.0%" -> "5%"
    """
    text = text.lower().translate(_PUNCT)
    text = text.replace("₱", "")
    text = re.sub(r"\b(?:php|p)\s?(?=\d)", "", text)   # currency prefix before a number
    text = re.sub(r"(?<=\d),(?=\d)", "", text)         # thousands separators: 100,000 -> 100000
    text = re.sub(r"(\d)\.0+\b", r"\1", text)          # trailing zeros: 500.00 -> 500, 5.0 -> 5
    text = re.sub(r"\s+", " ", text).strip()           # collapse spaces/newlines
    return text


# golden test data path
test_data_path = PROJECT_ROOT / "eval" / "golden_set.jsonl"

def load_jsonl():
    data = []
    with open(test_data_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): # skip empty or whitespace only
                continue
            data.append(json.loads(line))
    return data



@pytest.mark.parametrize(
        "test_query",
        load_jsonl()
)


def test_golden_data(test_query):

    #print(test_query)

    start_time = time.perf_counter()
    qs_obj = IP_AGENT.invoke({"query": test_query['q']})
    end_time = time.perf_counter()
    
    #print(qs_obj)

    # get the answer field (Answer obj) from QueryState structure
    answer_obj = qs_obj['answer']
    chunks = qs_obj["chunks"]

    if test_query['answerable']:
        # check actual "answer" of llm from object
        is_contained = False
        if test_query['must_contain_any']:
            for p in test_query["must_contain_any"]:
                if contains(p, answer_obj['answer']):
                    is_contained = True
                    break
            

        answer_found = normalize(test_query['answer']) in normalize(answer_obj['answer'])    

        assert answer_found or is_contained

        actual_pages = [citation['page'] for citation in answer_obj['citations']]

        #print(actual_pages)

        # check if expected page is in actual pages from citations
        assert test_query['expected_page'] in actual_pages

        # grounding: every quote really exists in a retrieved chunk
        #chunk_texts = [compact(d.page_content) for d in chunks]
        #for c in answer_obj['citations']:
        #    assert any(compact(c["quote"]) in t for t in chunk_texts), f"invented quote: {c['quote']!r}"

    else:
        assert answer_obj['citations'] == []
        # how to force llm to reply in specific manner when info is not answerable
    print(f'Q:{test_query['q']}\nAns:{answer_obj['answer']}\nTiming:{end_time - start_time:.2f}s\n')
    #break