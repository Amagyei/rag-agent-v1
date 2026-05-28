import os
os.environ["DOCLING_DEVICE"] = "cpu"
import io
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import Node, TextNode, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv
import uuid
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
import ollama as ollama_client
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem

load_dotenv()

EMBED_DIM        = 768
LOADER_BACKEND   = os.getenv("LOADER_BACKEND", "docling")
VISION_MODEL     = os.getenv("VISION_MODEL", "moondream:latest")
MIN_IMAGE_PX     = 150
TABLE_CHAR_BUDGET = 6000


# ─────────────────────────────────────────────
# NODE PARSERS
# ─────────────────────────────────────────────

parent_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128, include_prev_next_rel=True)
child_splitter  = SentenceSplitter(chunk_size=128, chunk_overlap=32,  include_prev_next_rel=True)
prose_parser = HierarchicalNodeParser(
    chunk_sizes=[512, 128],
    node_parser_ids=["parent_chunk", "child_chunk"],
    node_parser_map={"parent_chunk": parent_splitter, "child_chunk": child_splitter},
)


# ─────────────────────────────────────────────
# IMAGE DESCRIPTION
# ─────────────────────────────────────────────

def _describe_image(pil_image, page_context: str = "") -> str:
    w, h = pil_image.size
    if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
        print(f"  [vision] Skipping small image ({w}×{h}px) — likely icon/logo")
        return ""

    # ── Convert PIL → bytes ───────────────────────────────────────────────────
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    prompt = (
        "This image is from a departmental procedures manual. "
        f"It appears under the section: '{page_context}'.\n\n"
        "Describe it as a structured process if it seems to have a flow or an object. "
        "Transcribe ALL visible text exactly as written. "
        "Include every decision point (yes/no branches), every role or actor "
        "responsible for each step, and the start and end conditions. "
        "Format each step as:  Step N [ROLE]: action → outcome. "
        "If it is a table or chart, describe each row and column clearly."
    )

    try:
        response = ollama_client.Client(host="http://localhost:11434").chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_bytes], 
            }]
        )
        return response.message.content.strip()

    except Exception as e:
        print(f"  [vision] WARNING: Vision LLM failed: {e}")
        print(f"  [vision]   Make sure model is pulled: ollama pull {VISION_MODEL}")
        return f"[Image in section '{page_context}' — vision description pending]"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _make_structural_node(text: str, metadata: dict) -> TextNode:
    return TextNode(text=text, id_=str(uuid.uuid4()), metadata=metadata)


def _split_table_by_char_budget(
    markdown_table: str,
    section_heading: str,
    source_file: str,
    budget: int = TABLE_CHAR_BUDGET,
) -> list[TextNode]:
    """
    Split a markdown table string into multiple TextNodes, each under `budget`
    characters, with the section heading and column headers prepended to every
    chunk so context is preserved across splits.

    Handles the edge case where a single data row exceeds the budget by
    truncating at the nearest pipe (|) before the limit so markdown stays valid.

    Returns a list of TextNodes tagged element_type='table'.
    """
    lines = markdown_table.splitlines()

    # ── Identify table structure lines ───────────────────────────────────────
    header_row = ""
    separator_row = ""
    data_rows: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not header_row and stripped.startswith("|"):
            header_row = line
        elif not separator_row and set(stripped.replace("|", "").replace("-", "").replace(" ", "")) == set():
            separator_row = line
        elif stripped.startswith("|"):
            data_rows.append(line)

    # Fallback: if Docling didn't produce a clear header/separator, treat all
    # lines as data rows and skip the header-repeat logic.
    if not header_row:
        header_row = ""
        separator_row = ""

    prefix = (f"## {section_heading}\n\n" if section_heading else "")
    header_block = f"{header_row}\n{separator_row}\n" if (header_row and separator_row) else ""

    nodes: list[TextNode] = []
    current_rows: list[str] = []
    current_len = len(prefix) + len(header_block)

    def _flush(rows: list[str]) -> None:
        if not rows:
            return
        chunk_text = prefix + header_block + "\n".join(rows)
        nodes.append(_make_structural_node(
            text=chunk_text,
            metadata={
                "element_type": "table",
                "section": section_heading,
                "source_file": source_file,
                "chunk_index": len(nodes),
            }
        ))

    for row in data_rows:
        row_len = len(row) + 1  # +1 for the newline

        if current_len + row_len <= budget:
            # Normal case: row fits in current chunk
            current_rows.append(row)
            current_len += row_len
        elif row_len > budget:
            # Edge case: single row exceeds budget — flush current, emit truncated row
            _flush(current_rows)
            current_rows = []
            current_len = len(prefix) + len(header_block)

            # Truncate at the nearest pipe before the budget
            max_chars = budget - len(prefix) - len(header_block)
            truncated = row[:max_chars]
            last_pipe = truncated.rfind("|")
            if last_pipe > 0:
                truncated = truncated[:last_pipe] + "|"  # close the cell
            print(f"  [table] WARNING: Single row exceeded budget ({row_len} chars), truncated to {len(truncated)}")
            nodes.append(_make_structural_node(
                text=prefix + header_block + truncated,
                metadata={
                    "element_type": "table",
                    "section": section_heading,
                    "source_file": source_file,
                    "chunk_index": len(nodes),
                    "truncated": True,
                }
            ))
        else:
            # Row fits but not in current chunk — flush and start new chunk
            _flush(current_rows)
            current_rows = [row]
            current_len = len(prefix) + len(header_block) + row_len

    _flush(current_rows)  # flush remaining rows
    return nodes


# ─────────────────────────────────────────────
# DOCLING LOADER
# ─────────────────────────────────────────────
image_count= 0
def load_pdf_docling(path: str):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(path)
    doc    = result.document

    prose_chunks     = []
    structural_nodes = []
    current_heading  = ""

    for element, _level in doc.iterate_items():

        # ── PictureItem: describe with vision LLM ─────────────────────────────
        if isinstance(element, PictureItem):
            pil_image = element.get_image(doc)
            if pil_image is not None:
                description = _describe_image(pil_image, page_context=current_heading)
                
                if description:
                    structural_nodes.append(_make_structural_node(
                        text=f"[Diagram — {current_heading}]\n{description}",
                        metadata={
                            "element_type": "diagram",
                            "section":      current_heading,
                            "source_file":  os.path.basename(path),
                            "source_type":  "vision_description",
                        }
                    ))
            # continue
            continue

        # ── All other items: export_to_markdown(doc) ──────────────────────────
        if hasattr(element, "export_to_markdown"):
            text = element.export_to_markdown(doc)
        else:
            text = str(element)

        if not text or not text.strip():
            continue

        label = getattr(element, "label", "paragraph")

        # Heading: track context, don't create a node
        if label in ("section_header", "title"):
            current_heading = text.strip()
            continue

        # Table: split by character budget, headers repeated on every chunk
        if isinstance(element, TableItem):
            table_chunks = _split_table_by_char_budget(
                markdown_table=text,
                section_heading=current_heading,
                source_file=os.path.basename(path),
            )
            if table_chunks:
                structural_nodes.extend(table_chunks)
                print(f"  [table] '{current_heading[:60]}' → {len(table_chunks)} chunk(s)")
            continue

        # Prose: accumulate for sentence-splitting
        paragraph_text = f"{current_heading}: {text}" if current_heading else text
        prose_chunks.append(paragraph_text)

    # ── Sentence-split accumulated prose ─────────────────────────────────────
    prose_nodes = []
    if prose_chunks:
        combined    = "\n\n".join(prose_chunks)
        prose_docs  = [Document(text=combined)]
        prose_nodes = prose_parser.get_nodes_from_documents(prose_docs)
        for node in prose_nodes:
            node.metadata["element_type"] = "prose"
            node.metadata["source_file"]  = os.path.basename(path)

    all_nodes = structural_nodes + prose_nodes
    print(
        f"[data_loader] Docling produced {len(all_nodes)} nodes "
        f"({len(structural_nodes)} structural + {len(prose_nodes)} prose + {image_count} images)"
    )
    return all_nodes


# ─────────────────────────────────────────────
# LEGACY LOADER
# ─────────────────────────────────────────────

def load_pdf_legacy(path: str):
    docs  = PDFReader().load_data(file=path)
    nodes = prose_parser.get_nodes_from_documents(docs)
    return nodes


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def load_and_create_nodes_from_pdf(path: str):
    if LOADER_BACKEND == "docling":
        print(f"[data_loader] Using Docling backend for: {path}")
        return load_pdf_docling(path)
    else:
        print(f"[data_loader] Using legacy PyPDF backend for: {path}")
        return load_pdf_legacy(path)


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:v1.5",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"num_ctx": 8192}
)

def embed_nodes(nodes: list[Node]):
    for n in nodes:
        if len(n.text) > 8000:
            print(f"  [embed] WARNING: Non-table node exceeded budget ({len(n.text)} chars), hard-truncating.")
            n.text = n.text[:8000]

    embeddings = embed_model.get_text_embedding_batch([n.text for n in nodes])
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb
    return nodes

def embed_query(query: str):
    return embed_model.get_text_embedding(query)