import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import PHASE_0_JSON, PHASE_1_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_elements() -> list[dict]:
    with open(PHASE_0_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data["elements"]


def _get_slide_id(el: dict) -> str:
    slide = el.get("slide", "")
    if slide:
        return slide
    sn = el.get("slide_number")
    if sn:
        return f"Slide {sn}"
    return ""


def _is_slide_heading(el: dict) -> bool:
    return el.get("element_type") == "slide_heading"


def merge_related_elements(elements: list[dict]) -> list[dict]:
    merged = []
    i = 0
    while i < len(elements):
        el = elements[i]
        doc_type = el.get("doc_type", "")
        slide_id = _get_slide_id(el)

        if doc_type in ("process_flow", "training") and slide_id:
            group = [el]
            i += 1
            while i < len(elements):
                nxt = elements[i]
                nxt_slide = _get_slide_id(nxt)
                if _is_slide_heading(nxt) and nxt_slide and nxt_slide != slide_id:
                    break
                if nxt_slide:
                    group.append(nxt)
                    i += 1
                else:
                    break
            merged.append(_merge_group(group))
        else:
            merged.append(el)
            i += 1
    return merged


def _merge_group(group: list[dict]) -> dict:
    merged = dict(group[0])
    parts = []
    seen_labels = set()
    for el in group:
        et = el.get("element_type", "")
        if et == "narrator":
            txt = el.get("narrator_text", "")
            if txt and not txt.startswith("Narrator:"):
                parts.append(f"Narrator: {txt}")
            elif txt:
                parts.append(txt)
        elif et == "section_label":
            label = el.get("label", "")
            if label and label not in seen_labels:
                seen_labels.add(label)
                parts.append(f"\n{label}:")
        elif et == "paragraph":
            txt = el.get("text", "")
            if txt:
                parts.append(f"  - {txt}")
        elif et == "slide_heading":
            txt = el.get("text", "") or el.get("slide_title", "")
            if txt:
                parts.append(txt)
        else:
            txt = el.get("text", "")
            if txt:
                parts.append(txt)

    merged["text"] = "\n".join(parts)
    merged["element_type"] = "merged_slide"
    return merged


def make_text(element: dict) -> str:
    et = element.get("element_type", "")
    if et == "term_definition":
        parts = [f"Term: {element.get('term', '')}"]
        if element.get("full_form"):
            parts.append(f"Full Form: {element['full_form']}")
        if element.get("explanation"):
            parts.append(f"Explanation: {element['explanation']}")
        if element.get("example"):
            parts.append(f"Example: {element['example']}")
        if element.get("why_needed"):
            parts.append(f"Why Needed: {element['why_needed']}")
        if element.get("used_by"):
            parts.append(f"Used By: {', '.join(element['used_by'])}")
        return "\n".join(parts)

    if et == "scenario":
        parts = [
            f"Scenario {element.get('scenario_number', '')}: {element.get('title', '')}",
        ]
        if element.get("domain"):
            parts.append(f"Domain: {element['domain']}")
        if element.get("scenario_text"):
            parts.append(f"Scenario: {element['scenario_text']}")
        if element.get("what_went_wrong"):
            parts.append(f"What Went Wrong: {element['what_went_wrong']}")
        if element.get("who_was_involved"):
            parts.append(f"Who Was Involved: {element['who_was_involved']}")
        if element.get("expected_vs_actual"):
            parts.append(f"Expected vs Actual: {element['expected_vs_actual']}")
        if element.get("impact"):
            parts.append(f"Impact: {element['impact']}")
        return "\n".join(parts)

    if et == "qa_pair":
        return f"Q: {element.get('question', '')}\nA: {element.get('answer', '')}"

    if et == "section_heading":
        return element.get("text", element.get("section_title", ""))

    if et == "slide_heading":
        txt = element.get("text", "")
        proc = element.get("process_name", "")
        return f"{proc} - {txt}" if proc else txt

    if et == "narrator":
        return element.get("narrator_text", "")

    if et == "paragraph":
        txt = element.get("text", "")
        proc = element.get("process_name", "")
        slide = element.get("slide", "")
        parts = []
        if proc:
            parts.append(f"[{proc}]")
        if slide:
            parts.append(f"[{slide}]")
        parts.append(txt)
        return " ".join(parts)

    if et == "process_section":
        return element.get("text", element.get("process_name", ""))

    if et == "section_label":
        return element.get("label", "")

    if et == "merged_slide":
        return element.get("text", "")

    return element.get("text", "")


def chunk_text(text: str, max_words: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        step = max_words - overlap
        if step <= 0:
            step = max_words
        start += step
    return chunks


META_BUILDERS: dict[str, callable] = {}


def _meta_term(element: dict) -> dict:
    return {
        "term": element.get("term", ""),
        "domain": "Glossary",
        "section": element.get("section_title", ""),
    }


def _meta_scenario(element: dict) -> dict:
    return {
        "scenario_number": element.get("scenario_number", 0),
        "title": element.get("title", ""),
        "domain": element.get("domain", "Scenario"),
    }


def _meta_qa(element: dict) -> dict:
    return {
        "question_number": element.get("question_number", 0),
        "section": element.get("section", ""),
        "domain": "FAQ",
    }


def _meta_slide(element: dict) -> dict:
    return {
        "process_name": element.get("process_name", ""),
        "slide": element.get("slide", ""),
        "slide_number": element.get("slide_number", 0),
        "domain": "Process Flow",
    }


def _meta_flow(element: dict) -> dict:
    return {
        "process_name": element.get("process_name", ""),
        "slide": element.get("slide", ""),
        "domain": element.get("doc_type", "General"),
    }


def _meta_section_heading(element: dict) -> dict:
    return {
        "section_title": element.get("section_title", ""),
        "domain": "Glossary",
    }


def _meta_merged_slide(element: dict) -> dict:
    return {
        "process_name": element.get("process_name", ""),
        "slide": element.get("slide", ""),
        "slide_number": element.get("slide_number", 0),
        "slide_title": element.get("slide_title", ""),
        "domain": element.get("doc_type", "General"),
    }


META_BUILDERS = {
    "term_definition": _meta_term,
    "scenario": _meta_scenario,
    "qa_pair": _meta_qa,
    "slide_heading": _meta_slide,
    "narrator": _meta_flow,
    "paragraph": _meta_flow,
    "section_label": _meta_flow,
    "process_section": _meta_flow,
    "section_heading": _meta_section_heading,
    "merged_slide": _meta_merged_slide,
}


def build_metadata(element: dict) -> dict:
    meta = {
        "source_doc": element.get("source_doc", ""),
        "doc_type": element.get("doc_type", ""),
        "element_type": element.get("element_type", ""),
    }
    builder = META_BUILDERS.get(element.get("element_type", ""))
    if builder:
        meta.update(builder(element))
    return meta


def chunk_all() -> list[dict]:
    elements = load_elements()
    elements = merge_related_elements(elements)
    PHASE_1_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []

    for element in elements:
        text = make_text(element)
        if not text.strip():
            continue

        meta = build_metadata(element)
        text_chunks = chunk_text(text)
        for i, tc in enumerate(text_chunks):
            m = dict(meta)
            if len(text_chunks) > 1:
                m["chunk_index"] = i
                m["total_chunks"] = len(text_chunks)
            chunks.append({"text": tc, "metadata": m})

    out_path = PHASE_1_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("Chunking complete: %d chunks written to %s", len(chunks), out_path)
    return chunks


if __name__ == "__main__":
    chunk_all()
