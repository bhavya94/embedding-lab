"""Three ways to cut long documents before embedding, from the Advanced RAG course.

1. Fixed: consecutive windows of `size` sentences. Embed and return the same window.
2. Sentence window: embed each sentence alone (sharp match), return it with `window` sentences either side
   (enough context for a reader).
3. Auto-merging: embed small chunks; when most of a parent's children are retrieved, return the parent instead.

Every strategy produces chunks with an `embed_text` (what gets matched) and a span (what gets returned).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start: int  # sentence index, inclusive
    end: int  # exclusive
    embed_text: str
    parent: int | None = None  # auto-merging: index of the parent span in the doc's parent list


def fixed_chunks(doc_id: str, doc: list[str], size: int) -> list[Chunk]:
    return [Chunk(doc_id, start, min(start + size, len(doc)), " ".join(doc[start:start + size]))
            for start in range(0, len(doc), size)]


def sentence_windows(doc_id: str, doc: list[str], window: int) -> list[Chunk]:
    """One chunk per sentence; its span widens to `window` sentences on each side for the returned text."""
    return [Chunk(doc_id, max(0, i - window), min(len(doc), i + window + 1), sentence)
            for i, sentence in enumerate(doc)]


def auto_merging_chunks(doc_id: str, doc: list[str], child: int, parent: int) -> list[Chunk]:
    """Children of `child` sentences, each tagged with its parent block of `parent` sentences."""
    if parent % child:
        raise ValueError(f"parent size {parent} must be a multiple of child size {child}")
    return [Chunk(doc_id, start, min(start + child, len(doc)), " ".join(doc[start:start + child]),
                  parent=start // parent) for start in range(0, len(doc), child)]


def merge_hits(hits: list[Chunk], children: list[Chunk], threshold: float = 0.5) -> list[tuple[str, int, int]]:
    """Ranked child hits to returned spans: a parent replaces its children once more than `threshold` of them hit.

    `children` is every child chunk, so each parent's size and span come from its real children: a short last
    parent can merge too, and no span runs past the end of its document. The parent takes the rank of its best
    child; the merged children drop out.
    """
    family: dict[tuple[str, int], list[Chunk]] = {}
    for chunk in children:
        family.setdefault((chunk.doc_id, chunk.parent), []).append(chunk)
    counts: dict[tuple[str, int], int] = {}
    for hit in hits:
        counts[(hit.doc_id, hit.parent)] = counts.get((hit.doc_id, hit.parent), 0) + 1
    spans, seen = [], set()
    for hit in hits:
        key = (hit.doc_id, hit.parent)
        members = family[key]
        if counts[key] / len(members) > threshold:
            if key not in seen:
                seen.add(key)
                spans.append((hit.doc_id, members[0].start, members[-1].end))
        else:
            spans.append((hit.doc_id, hit.start, hit.end))
    return spans
