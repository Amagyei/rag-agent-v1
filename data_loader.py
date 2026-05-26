"""
data_loader.py — Hybrid structure-aware PDF ingestion
=====================================================

Fixes applied vs previous version:
  - _describe_image now converts PIL Image → bytes before passing to ollama
    (ollama rejects PIL objects; accepts bytes | str | Path only)
  - Images smaller than MIN_IMAGE_PX are skipped (logos/icons, not diagrams)
  - Diagram node is only created when description is non-empty
"""

import os
import io
os.environ["DOCLING_DEVICE"] = "cpu"

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
)
from llama_index.core.schema import Node, TextNode, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv
import uuid

from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
import ollama as ollama_client
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem

load_dotenv()

EMBED_DIM    = 768
LOADER_BACKEND = os.getenv("LOADER_BACKEND", "docling")
VISION_MODEL   = os.getenv("VISION_MODEL", "moondream:latest")

# Images below this size (either dimension) are skipped — they are logos/icons.
# Meaningful flowcharts and diagrams are typically 200px+ on their shortest side.
MIN_IMAGE_PX = 150


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
    """
    Describe a PIL Image using a local Ollama vision model.

    FIX: converts PIL Image → bytes via io.BytesIO.
         ollama's images= parameter only accepts bytes | str | Path,
         NOT PIL Image objects (Pydantic raises a validation error otherwise).

    Returns empty string for images too small to be meaningful diagrams,
    so the caller can skip creating a node for them.
    """

    # Skip tiny images — logos, icons, decorative elements
    w, h = pil_image.size
    if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
        print(f"  [vision] Skipping small image ({w}×{h}px) — likely icon/logo")
        return ""

    # ── Convert PIL → bytes ───────────────────────────────────────────────────
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()            # bytes ← what ollama actually accepts

    prompt = (
        "This image is from a departmental procedures manual. "
        f"It appears under the section: '{page_context}'.\n\n"
        "Describe it as a structured, numbered step-by-step process. "
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


# ─────────────────────────────────────────────
# DOCLING LOADER
# ─────────────────────────────────────────────

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
            # pil_image = element.get_image(doc)
            # if pil_image is not None:
            #     description = _describe_image(pil_image, page_context=current_heading)
            #     # Only create a node if description is non-empty
            #     # (empty = image was too small / skipped)
            #     if description:
            #         structural_nodes.append(_make_structural_node(
            #             text=f"[Diagram — {current_heading}]\n{description}",
            #             metadata={
            #                 "element_type": "diagram",
            #                 "section":      current_heading,
            #                 "source_file":  os.path.basename(path),
            #                 "source_type":  "vision_description",
            #             }
            #         ))
            # continue
            image_count= 0
            print(f"image found and skipped ", image_count)
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

        # Table: whole structural node
        if isinstance(element, TableItem):
            table_text = f"## {current_heading}\n\n{text}" if current_heading else text
            structural_nodes.append(_make_structural_node(
                text=table_text,
                metadata={
                    "element_type": "table",
                    "section":      current_heading,
                    "source_file":  os.path.basename(path),
                }
            ))
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
        f"({len(structural_nodes)} structural + {len(prose_nodes)} prose)"
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
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

def embed_nodes(nodes: list[Node]):
    embeddings = embed_model.get_text_embedding_batch([n.text for n in nodes])
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb
    return nodes

def embed_query(query: str):
    return embed_model.get_text_embedding(query)