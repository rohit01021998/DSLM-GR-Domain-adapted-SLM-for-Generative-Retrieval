import json
import uuid
from typing import List, Dict, Any, Optional


class OCRPostProcessor:
    """
    Processes raw OCR extraction JSON output to prepare it for semantic chunking.
    Removes noise (headers/footers), groups text by section, and chunks large
    sections while preserving section context and structural integrity
    (tables/formulas are never split mid-structure).
    """

    def __init__(self, max_chunk_size: int = 1500, overlap: int = 150):
        self.max_chunk_size = max_chunk_size
        # NOTE: `overlap` is intentionally not used for character-level text
        # duplication between chunks anymore (see chunk_group). This chunker
        # only splits at existing block boundaries, so there's no mid-flow
        # content to protect with a text-overlap window, and slicing raw
        # text at an arbitrary offset produced broken fragments. Kept as a
        # constructor arg for API compatibility; chunks are instead linked
        # via `prev_chunk_id`/`next_chunk_id` in metadata.
        self.overlap = overlap

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Group by section while cleaning out noise
        grouped_sections = self.group_by_section(data)

        # 2. Merge heading-only groups forward into their following section.
        #    A heading with no body is not a retrievable unit on its own.
        grouped_sections = self.merge_heading_only_groups(grouped_sections)

        # 3. Chunk each group's blocks directly (no lossy markdown round-trip)
        final_chunks = []
        for group in grouped_sections:
            final_chunks.extend(self.chunk_group(group))

        # 4. Link chunks sequentially ACROSS THE WHOLE DOCUMENT (not per
        # section group — a per-group chain resets at every section
        # boundary, which is wrong: a chunk at the end of one section has a
        # real, meaningful neighbor at the start of the next one). Chains
        # only break between different source documents. This replaces
        # character-level text overlap: a retriever can pull in the
        # adjacent chunk on demand without any chunk's own text ever being
        # duplicated or mutated.
        self.link_chunks(final_chunks)

        return final_chunks

    def link_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Set prev_chunk_id/next_chunk_id in place, resetting only at
        document boundaries."""
        for i, c in enumerate(chunks):
            prev_c = chunks[i - 1] if i > 0 else None
            next_c = chunks[i + 1] if i < len(chunks) - 1 else None

            same_doc_prev = (
                prev_c is not None
                and prev_c["metadata"]["document"] == c["metadata"]["document"]
            )
            same_doc_next = (
                next_c is not None
                and next_c["metadata"]["document"] == c["metadata"]["document"]
            )

            c["metadata"]["prev_chunk_id"] = (
                prev_c["metadata"]["chunk_id"] if same_doc_prev else None
            )
            c["metadata"]["next_chunk_id"] = (
                next_c["metadata"]["chunk_id"] if same_doc_next else None
            )

    # ------------------------------------------------------------------ #
    # Grouping
    # ------------------------------------------------------------------ #
    def group_by_section(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups = []
        current_group = None

        for block in data:
            section = block.get("section", "Uncategorized")
            document = block.get("document", "Unknown")
            page = block.get("page", 0)

            cleaned_content = self.clean_content(block)
            if not cleaned_content:
                continue

            if (
                current_group is None
                or current_group["section"] != section
                or current_group["document"] != document
            ):
                if current_group is not None:
                    groups.append(current_group)
                current_group = {
                    "section": section,
                    "document": document,
                    "pages": set(),
                    "blocks": [],
                }

            current_group["pages"].add(page)
            current_group["blocks"].extend(cleaned_content)

        if current_group is not None:
            groups.append(current_group)

        return groups

    def merge_heading_only_groups(
        self, groups: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        A group whose blocks are ALL headings (no text/table/image/formula
        body) is not a usable retrieval unit. Fold it forward into the next
        group in the same document so the heading still appears as a lead-in
        line, instead of being emitted as its own near-empty chunk.
        """
        merged: List[Dict[str, Any]] = []
        pending_heading_blocks: List[Dict[str, Any]] = []
        pending_pages: set = set()

        def is_heading_only(group: Dict[str, Any]) -> bool:
            return all(b["type"] == "Heading" for b in group["blocks"])

        for group in groups:
            if is_heading_only(group):
                pending_heading_blocks.extend(group["blocks"])
                pending_pages |= group["pages"]
                continue

            if pending_heading_blocks:
                group["blocks"] = pending_heading_blocks + group["blocks"]
                group["pages"] = pending_pages | group["pages"]
                pending_heading_blocks = []
                pending_pages = set()

            merged.append(group)

        # Trailing heading-only group with nothing after it (end of doc) —
        # keep it rather than silently drop it, but it's now the last chunk.
        if pending_heading_blocks:
            if merged and merged[-1]["document"] == groups[-1]["document"]:
                merged[-1]["blocks"].extend(pending_heading_blocks)
                merged[-1]["pages"] |= pending_pages
            else:
                merged.append(
                    {
                        "section": groups[-1]["section"],
                        "document": groups[-1]["document"],
                        "pages": pending_pages,
                        "blocks": pending_heading_blocks,
                    }
                )

        return merged

    # ------------------------------------------------------------------ #
    # Content cleaning
    # ------------------------------------------------------------------ #
    def clean_content(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        page = block.get("page", 0)
        block_type = block.get("type")
        if block_type == "section":
            return [{"type": "Heading", "data": block.get("content", ""), "page": page}]
        elif block_type == "image":
            return [{"type": "Image", "data": block.get("content", ""), "page": page}]
        elif block_type == "table":
            content = block.get("content")
            if isinstance(content, list):
                return self._filter_mixed(content, page)
            return [{"type": "Table", "data": content, "page": page}]
        elif block_type == "formula":
            content = block.get("content")
            if isinstance(content, list):
                return self._filter_mixed(content, page)
            return [{"type": "Formula", "data": content, "page": page}]
        elif block_type == "mixed":
            return self._filter_mixed(block.get("content", []), page)

        return []

    # Source data is inconsistent about type casing: a top-level `table`/
    # `formula` block gets a capitalized type when its content isn't a list,
    # but the same content nested inside a `mixed` block keeps whatever
    # casing the OCR layer emitted (observed: lowercase "formula"). Left
    # unnormalized, this silently breaks any downstream filter or the
    # atomic-chunking check for tables/formulas. Canonicalize on ingest.
    _TYPE_CANONICAL = {
        "table": "Table",
        "formula": "Formula",
        "heading": "Heading",
        "image": "Image",
    }

    def _canonical_type(self, raw_type: Optional[str]) -> str:
        if raw_type is None:
            return "Text"
        return self._TYPE_CANONICAL.get(raw_type.lower(), raw_type)

    def _filter_mixed(
        self, contents: List[Dict[str, Any]], page: int
    ) -> List[Dict[str, Any]]:
        cleaned = []
        for item in contents:
            if item.get("type") not in ["Page-header", "Page-footer"]:
                cleaned.append(
                    {
                        "type": self._canonical_type(item.get("type")),
                        "data": item.get("data", ""),
                        "page": page,
                    }
                )
        return cleaned

    # ------------------------------------------------------------------ #
    # Block -> markdown rendering (per-block, not per-group)
    # ------------------------------------------------------------------ #
    def render_block(self, block: Dict[str, Any]) -> Optional[str]:
        """Render a single block to its markdown fragment. Returns None for
        blocks that shouldn't contribute embeddable text (kept in metadata
        instead)."""
        t = block.get("type")
        d = block.get("data", "")

        if t == "Image":
            # Don't dump the raw file path into embeddable text — it carries
            # no semantic signal and dilutes the chunk. Path is preserved in
            # chunk metadata (see chunk_group) for traceability instead.
            return "[Image]"
        elif t == "List-item":
            return f"- {d}"
        else:
            # Heading, Text, Caption, Footnote, Table, Formula
            return str(d)

    def image_ref(self, block: Dict[str, Any]) -> Optional[str]:
        if block.get("type") == "Image":
            return block.get("data")
        return None

    # ------------------------------------------------------------------ #
    # Chunking (operates on the block list directly)
    # ------------------------------------------------------------------ #
    def chunk_group(self, group: Dict[str, Any]) -> List[Dict[str, Any]]:
        section = group["section"]
        document = group["document"]
        header = f"**Section Context:** {section}"

        chunks: List[Dict[str, Any]] = []
        buffer_blocks: List[Dict[str, Any]] = []
        buffer_len = 0

        def flush(extra_block: Optional[Dict[str, Any]] = None):
            nonlocal buffer_blocks, buffer_len
            if not buffer_blocks and not extra_block:
                return

            blocks_to_emit = buffer_blocks + ([extra_block] if extra_block else [])
            body_parts = []
            content_types = set()
            image_refs = []
            chunk_pages = set()

            for b in blocks_to_emit:
                rendered = self.render_block(b)
                if rendered:
                    body_parts.append(rendered)
                content_types.add(b["type"])
                ref = self.image_ref(b)
                if ref:
                    image_refs.append(ref)
                if "page" in b:
                    chunk_pages.add(b["page"])

            body = "\n\n".join(p for p in body_parts if p.strip())
            full_text = f"{header}\n\n{body}".strip()

            # Report only the pages actually represented by this chunk's own
            # blocks (each block carries its source page from clean_content)
            # rather than the whole section group's page range — a large
            # section split into several chunks should not have every chunk
            # claiming to span pages it doesn't contain, since that's a real
            # citation-accuracy problem downstream.
            chunks.append(
                {
                    "page_content": full_text,
                    "metadata": {
                        "chunk_id": str(uuid.uuid4()),
                        "document": document,
                        "pages": sorted(chunk_pages),
                        "section": section,
                        "content_types": sorted(content_types),
                        "image_refs": image_refs,
                        "token_count_est": len(full_text) // 4,
                    },
                }
            )

            buffer_blocks = []
            buffer_len = 0

        for block in group["blocks"]:
            rendered = self.render_block(block) or ""
            block_len = len(rendered)

            # Tables/formulas are never split mid-structure. If a single
            # block alone exceeds max_chunk_size, flush whatever's pending
            # first, then emit the block as its own atomic chunk (still
            # carrying full section context) rather than truncating it.
            if block_len > self.max_chunk_size and block["type"] in (
                "Table",
                "Formula",
            ):
                # Merge whatever's pending (often just a heading) INTO the
                # atomic table/formula chunk rather than flushing it alone
                # first — a heading-only chunk immediately followed by a
                # context-free table is the exact bug this rewrite fixes.
                flush(extra_block=block)
                continue

            if buffer_len + block_len > self.max_chunk_size and buffer_blocks:
                flush()

            buffer_blocks.append(block)
            buffer_len += block_len

        flush()

        return chunks


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "ocr_extraction.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "chunks_fixed.json"

    processor = OCRPostProcessor(max_chunk_size=1500, overlap=150)
    chunks = processor.process_file(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(chunks)} chunks to {output_path}")