"""
normalise_nodes.py
==================
LlamaIndex node normalisation **engine**.

This module is document-agnostic — all document-specific settings (header
regex patterns, whitelist terms, scoring thresholds, etc.) live in a
YAML config file loaded at runtime.

What the pipeline does:
  1. Removes repeating page header blocks that appear on every page
  2. Removes table-of-contents / dot-leader fragments
  3. Drops nodes that become empty or too short after cleaning
  4. Preserves legitimate repeated content (acronym definitions, policy terms)
     using BOTH frequency AND context diversity scoring — never frequency alone

How the boilerplate detection works:
  Every line's "boilerplate score" is:
      (frequency / total_nodes) × log(line_length + 1) × context_similarity

  - frequency alone is not enough: "SSNIT" appears many times but in diverse
    contexts, so its context_similarity is low and its score stays small.
  - A true boilerplate line appears often AND always surrounded by near-identical
    text (the same page header). Both conditions must be true.

Usage:
  python normalise_nodes.py --config configs/c-a_manual.yaml \\
                            --input data.json --output clean_nodes.json

  Or import directly:
    from normalise_nodes import load_config, normalise_pipeline
    cfg = load_config("configs/c-a_manual.yaml")
    clean_nodes, discard_log = normalise_pipeline(nodes, cfg)
"""

import json
import re
import math
import argparse
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field

import yaml


def _node_to_dict(node) -> dict:
    """Accept LlamaIndex TextNode objects or JSON-style dicts."""
    if isinstance(node, dict):
        return node
    if hasattr(node, "to_dict"):
        return node.to_dict()
    raise TypeError(
        f"Expected dict or LlamaIndex node, got {type(node).__name__}"
    )


# ─────────────────────────────────────────────
# CONFIGURATION DATACLASS
# ─────────────────────────────────────────────

# Mapping from YAML flag names to `re` module constants
_RE_FLAGS = {
    "DOTALL":     re.DOTALL,
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE":  re.MULTILINE,
    "VERBOSE":    re.VERBOSE,
}


def _compile_flags(names: list[str]) -> int:
    """OR-combine a list of regex flag names into an ``re`` flags int."""
    result = 0
    for name in names:
        flag = _RE_FLAGS.get(name.upper())
        if flag is None:
            raise ValueError(
                f"Unknown regex flag '{name}'. "
                f"Valid flags: {sorted(_RE_FLAGS)}"
            )
        result |= flag
    return result


def _compile_patterns(entries: list[dict]) -> list[re.Pattern]:
    """Compile a list of ``{pattern, flags}`` dicts into ``re.Pattern`` objects."""
    compiled = []
    for entry in entries:
        raw = entry["pattern"]
        flags = _compile_flags(entry.get("flags", []))
        compiled.append(re.compile(raw, flags))
    return compiled


@dataclass
class NormaliserConfig:
    """
    Runtime configuration for the normalisation pipeline.
    Every field maps 1-to-1 to a YAML key — no business logic here.
    """

    # ── Phase 0: structural patterns (compiled) ──────────────────────
    header_patterns:       list[re.Pattern] = field(default_factory=list)
    orphan_line_patterns:  list[re.Pattern] = field(default_factory=list)

    # ── Phase 1: dynamic scoring ─────────────────────────────────────
    min_frequency:                int   = 10
    boilerplate_score_threshold:  float = 0.15
    min_line_length:              int   = 15
    context_sample_size:          int   = 6

    # ── Phase 2: node-level filters ──────────────────────────────────
    min_clean_text_length:  int   = 80
    max_dot_fraction:       float = 0.12
    dot_leader_min_run:     int   = 4

    # ── Whitelist ────────────────────────────────────────────────────
    whitelist:  list[str] = field(default_factory=list)

    # ── Idempotency ──────────────────────────────────────────────────
    idempotency_signal:  str = ""


def load_config(path: str | Path) -> NormaliserConfig:
    """
    Load a YAML config file and return a ``NormaliserConfig``.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError``
    on invalid flag names.
    """
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)

    scoring = raw.get("scoring", {})
    filters = raw.get("filters", {})

    return NormaliserConfig(
        # Compile regex patterns from YAML
        header_patterns=_compile_patterns(raw.get("header_patterns", [])),
        orphan_line_patterns=_compile_patterns(raw.get("orphan_line_patterns", [])),

        # Scoring
        min_frequency=scoring.get("min_frequency", 10),
        boilerplate_score_threshold=scoring.get("boilerplate_score_threshold", 0.15),
        min_line_length=scoring.get("min_line_length", 15),
        context_sample_size=scoring.get("context_sample_size", 6),

        # Filters
        min_clean_text_length=filters.get("min_clean_text_length", 80),
        max_dot_fraction=filters.get("max_dot_fraction", 0.12),
        dot_leader_min_run=filters.get("dot_leader_min_run", 4),

        # Whitelist & idempotency
        whitelist=raw.get("whitelist", []),
        idempotency_signal=raw.get("idempotency_signal", ""),
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def tokenize(text: str) -> set:
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def avg_pairwise_similarity(texts: list, cfg: NormaliserConfig) -> float:
    """
    Average Jaccard similarity between sampled node texts.
    High → contexts are similar → boilerplate.
    Low  → contexts are diverse → legitimate repetition.
    """
    sample = texts[:cfg.context_sample_size]
    sets = [tokenize(t) for t in sample]
    total, pairs = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            total += jaccard(sets[i], sets[j])
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def boilerplate_score(freq: int, line_length: int,
                       ctx_sim: float, total_nodes: int) -> float:
    """
    Score = (freq / total_nodes) × log(length + 1) × context_similarity

    High frequency + long text + similar context = boilerplate.
    Any one factor being low protects legitimate content.

    SSNIT example: freq=23, ctx_sim=0.30 → score ≈ 0.004 → KEEP
    "Schedule Officer": freq=248, ctx_sim=1.0 → score ≈ 1.2 → REMOVE
    """
    return (freq / total_nodes) * math.log(line_length + 1) * ctx_sim


def is_whitelisted(line: str, cfg: NormaliserConfig) -> bool:
    lower = line.lower()
    return any(w.lower() in lower for w in cfg.whitelist)


def is_dot_leader_node(text: str, cfg: NormaliserConfig) -> bool:
    runs = re.findall(r'\.{' + str(cfg.dot_leader_min_run) + r',}', text)
    dot_chars = sum(len(r) for r in runs)
    return (dot_chars / max(len(text), 1)) > cfg.max_dot_fraction


# ─────────────────────────────────────────────
# PHASE 0: STRUCTURAL HEADER STRIP
# ─────────────────────────────────────────────

def strip_structural_headers(text: str, cfg: NormaliserConfig) -> str:
    """
    Remove the repeating page header/footer block as a unit using regex.
    This is the first pass — faster and more complete than line-by-line scoring
    for known structural patterns.
    """
    for pattern in cfg.header_patterns:
        text = pattern.sub('', text)
    for pattern in cfg.orphan_line_patterns:
        text = pattern.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────
# PHASE 1: DYNAMIC BOILERPLATE SCORING
# ─────────────────────────────────────────────

def build_boilerplate_registry(nodes: list, cfg: NormaliserConfig) -> set:
    """
    Discover any remaining boilerplate lines not caught by the structural strip.
    Uses the frequency × length × context_similarity score.
    Returns a set of line strings to remove.
    """
    total_nodes = len(nodes)
    line_to_node_texts = defaultdict(list)

    for node in nodes:
        seen = set()
        for raw_line in node["text"].split("\n"):
            line = raw_line.strip()
            if len(line) < cfg.min_line_length:
                continue
            if line not in seen:
                line_to_node_texts[line].append(node["text"])
                seen.add(line)

    boilerplate: set = set()

    for line, node_texts in line_to_node_texts.items():
        freq = len(node_texts)
        if freq < cfg.min_frequency:
            continue
        if is_whitelisted(line, cfg):
            continue

        ctx_sim = avg_pairwise_similarity(node_texts, cfg)
        score = boilerplate_score(freq, len(line), ctx_sim, total_nodes)

        if score >= cfg.boilerplate_score_threshold:
            boilerplate.add(line)

    return boilerplate


# ─────────────────────────────────────────────
# PHASE 2: CLEAN INDIVIDUAL NODES
# ─────────────────────────────────────────────

def clean_node(node: dict, boilerplate_lines: set,
               cfg: NormaliserConfig) -> str:
    """
    Apply both cleaning passes to a single node.
    Returns the cleaned text string.
    """
    text = node["text"]

    # Pass A: regex structural strip
    text = strip_structural_headers(text, cfg)

    # Pass B: remove any remaining boilerplate lines found by scoring
    cleaned = []
    for raw_line in text.split("\n"):
        if raw_line.strip() not in boilerplate_lines:
            cleaned.append(raw_line)
    text = "\n".join(cleaned)

    # Final whitespace cleanup
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def should_discard(cleaned_text: str, cfg: NormaliserConfig) -> tuple:
    """Returns (discard: bool, reason: str)"""
    if len(cleaned_text) < cfg.min_clean_text_length:
        return True, f"too_short ({len(cleaned_text)} chars)"
    if is_dot_leader_node(cleaned_text, cfg):
        return True, "dot_leader_toc_fragment"
    return False, ""


# ─────────────────────────────────────────────
# PHASE 3: ORPHAN CLEANUP
# ─────────────────────────────────────────────

def remove_orphaned_relationships(nodes: list, kept_ids: set) -> list:
    """Remove relationship references pointing to discarded nodes."""
    for node in nodes:
        for rel_key in list(node["relationships"].keys()):
            rel = node["relationships"][rel_key]
            if isinstance(rel, list):
                node["relationships"][rel_key] = [
                    r for r in rel if r["node_id"] in kept_ids
                ]
                if not node["relationships"][rel_key]:
                    del node["relationships"][rel_key]
            elif isinstance(rel, dict):
                if rel.get("node_id") not in kept_ids:
                    del node["relationships"][rel_key]
    return nodes


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def normalise_pipeline(nodes: list, cfg: NormaliserConfig,
                       verbose: bool = True) -> tuple:
    """
    Full normalisation pipeline.

    Args:
        nodes:   raw LlamaIndex TextNode dicts (from JSON export)
        cfg:     NormaliserConfig loaded from a YAML file
        verbose: print detailed logs

    Returns:
        (clean_nodes, discard_log)
        - clean_nodes: list of cleaned node dicts ready for embedding
        - discard_log: list of dicts describing every discarded node
    """
    nodes = deepcopy(nodes)
    print(f"[normalise] Starting with {len(nodes)} nodes")

    # ── Phase 0: structural header strip (pre-pass before scoring) ────
    print("[normalise] Phase 0: stripping structural header blocks...")
    for node in nodes:
        node["text"] = strip_structural_headers(node.get_content(), cfg)
    print(f"[normalise]   Header blocks stripped from all nodes")

    # ── Phase 1: dynamic boilerplate scoring on cleaned texts ─────────
    print("[normalise] Phase 1: scoring remaining repeated lines...")
    boilerplate_lines = build_boilerplate_registry(nodes, cfg)
    print(f"[normalise]   Found {len(boilerplate_lines)} additional boilerplate lines")
    if verbose and boilerplate_lines:
        for line in sorted(boilerplate_lines):
            print(f"             └─ '{line}'")

    # ── Phase 2: apply cleaning and filter nodes ──────────────────────
    print("[normalise] Phase 2: cleaning and filtering nodes...")
    clean_nodes = []
    discard_log = []

    for node in nodes:
        cleaned_text = clean_node(node, boilerplate_lines, cfg)
        discard, reason = should_discard(cleaned_text, cfg)

        if discard:
            discard_log.append({
                "id": node["id_"],
                "page": node["metadata"].get("page_label"),
                "reason": reason,
                "original_preview": node["text"][:120]
            })
        else:
            node["text"] = cleaned_text
            clean_nodes.append(node)

    print(f"[normalise]   Kept: {len(clean_nodes)}  |  Discarded: {len(discard_log)}")
    if verbose:
        reasons = defaultdict(int)
        for d in discard_log:
            reasons[d["reason"]] += 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"             └─ {reason}: {count}")

    # ── Phase 3: fix dangling relationship refs ───────────────────────
    print("[normalise] Phase 3: removing orphaned relationships...")
    kept_ids = {n["id_"] for n in clean_nodes}
    clean_nodes = remove_orphaned_relationships(clean_nodes, kept_ids)

    print(f"[normalise] ✓ Done. {len(clean_nodes)} clean nodes ready for embedding.\n")
    return clean_nodes, discard_log


# ─────────────────────────────────────────────
# IDEMPOTENCY GUARD
# ─────────────────────────────────────────────

def is_already_normalised(nodes: list, cfg: NormaliserConfig) -> bool:
    """
    Returns True if this data has already been normalised.
    Prevents accidental double-processing.
    """
    signal = cfg.idempotency_signal
    if not signal:
        return False
    return all(signal not in n.get_content() for n in nodes)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Normalise LlamaIndex nodes using a YAML config"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the document-specific YAML config file"
    )
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log",    default="discard_log.json")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    with open(args.input) as f:
        data = json.load(f)

    nodes = data["nodes"]

    if is_already_normalised(nodes, cfg):
        print("[normalise] Already normalised — skipping.")
        return

    clean_nodes, discard_log = normalise_pipeline(
        nodes, cfg, verbose=not args.quiet
    )

    out = {"nodes": clean_nodes, "source_id": data.get("source_id", "1")}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[normalise] Clean nodes → {args.output}")

    with open(args.log, "w") as f:
        json.dump(discard_log, f, indent=2)
    print(f"[normalise] Discard log → {args.log}")


if __name__ == "__main__":
    main()
