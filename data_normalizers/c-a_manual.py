"""
normalise_nodes.py
==================
LlamaIndex node normalisation pipeline.

What this solves:
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
  python normalise_nodes.py --input data.json --output clean_nodes.json

  Or import directly:
    from normalise_nodes import normalise_pipeline
    clean_nodes, discard_log = normalise_pipeline(nodes)
"""

import json
import re
import math
import argparse
from collections import defaultdict
from copy import deepcopy


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

class Config:

    # ── Header block regex (document-specific) ────────────────────────
    # Strip the repeating header block that appears on every page.
    # This catches it as a unit, including short orphaned lines like
    # "Reviewed By" and "Date Issued" that are too short for frequency scoring.
    #
    # Pattern matches everything from the title line to the department/page ref.
    HEADER_BLOCK_PATTERN = re.compile(
        r'Complaints\s*&\s*Adjudication\s+Departmental\s+Manual'
        r'.*?'
        r'(?:Risk\s*&\s*Quality\s*Mgt\.?\s*Dept\.?'
        r'|Risk\s*&\s*Quality\s*Management\s*Department)'
        r'[^\n]*\n?',
        re.DOTALL | re.IGNORECASE
    )

    # The second half of the header (policy/issue/date block)
    POLICY_BLOCK_PATTERN = re.compile(
        r'Policy\s+No\.\s+SSP/CAD-\d+\s*\n'
        r'(?:Issue\s+[\d.]+\s*\n)?'
        r'(?:[^\n]*\n){0,3}'
        r'(?:Risk\s*&\s*Quality\s*Mgt\.?\s*Dept\.?[^\n]*\n?)',
        re.IGNORECASE
    )

    # Orphaned single lines left behind after header block removal
    ORPHAN_LINE_PATTERN = re.compile(
        r'^\s*(?:'
        r'Reviewed By'
        r'|Date Reviewed'
        r'|Date Issued'
        r'|Issue\s+[\d.]+'
        r'|\d+\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}'
        r')\s*$',
        re.MULTILINE | re.IGNORECASE
    )

    # ── Dynamic frequency scoring ─────────────────────────────────────
    # A line must appear at least this many times to be a candidate.
    MIN_FREQUENCY = 10

    # Composite score threshold. Lines above this are boilerplate.
    # Tuned on this document:
    #   Pure boilerplate ("Schedule Officer" col header): 1.2
    #   Borderline table header ("ACTIVITY RESPONSIBILITY REMARK"): 0.14
    #   Legitimate repeated content (SSNIT definition): 0.004
    BOILERPLATE_SCORE_THRESHOLD = 0.15

    # Only score lines at least this long (very short lines are too generic to flag)
    MIN_LINE_LENGTH_FOR_SCORING = 15

    # Pairwise context similarity sample size (for speed)
    CONTEXT_SAMPLE_SIZE = 6

    # ── Node-level filters ────────────────────────────────────────────
    # Discard a node if its cleaned text is shorter than this
    MIN_CLEAN_TEXT_LENGTH = 80

    # Discard if more than this fraction of text is dot-leader characters
    MAX_DOT_FRACTION = 0.12
    DOT_LEADER_MIN_RUN = 4

    # ── Whitelist ─────────────────────────────────────────────────────
    # Lines containing these substrings are never removed by frequency scoring.
    # The regex header strip above is unaffected by this whitelist.
    WHITELIST_SUBSTRINGS = [
        "SSNIT",
        "Act 766",
        "National Pensions Act",
        "Tier 1",
        "Tier 2",
        "PNDCL 247",
        "SLTF",
        "ATPP",
    ]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def tokenize(text: str) -> set:
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def avg_pairwise_similarity(texts: list) -> float:
    """
    Average Jaccard similarity between sampled node texts.
    High → contexts are similar → boilerplate.
    Low  → contexts are diverse → legitimate repetition.
    """
    sample = texts[:Config.CONTEXT_SAMPLE_SIZE]
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


def is_whitelisted(line: str) -> bool:
    lower = line.lower()
    return any(w.lower() in lower for w in Config.WHITELIST_SUBSTRINGS)


def is_dot_leader_node(text: str) -> bool:
    runs = re.findall(r'\.{' + str(Config.DOT_LEADER_MIN_RUN) + r',}', text)
    dot_chars = sum(len(r) for r in runs)
    return (dot_chars / max(len(text), 1)) > Config.MAX_DOT_FRACTION


# ─────────────────────────────────────────────
# PHASE 0: STRUCTURAL HEADER STRIP
# ─────────────────────────────────────────────

def strip_structural_headers(text: str) -> str:
    """
    Remove the repeating page header/footer block as a unit using regex.
    This is the first pass — faster and more complete than line-by-line scoring
    for known structural patterns.
    """
    text = Config.HEADER_BLOCK_PATTERN.sub('', text)
    text = Config.POLICY_BLOCK_PATTERN.sub('', text)
    text = Config.ORPHAN_LINE_PATTERN.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────
# PHASE 1: DYNAMIC BOILERPLATE SCORING
# ─────────────────────────────────────────────

def build_boilerplate_registry(nodes: list) -> set:
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
            if len(line) < Config.MIN_LINE_LENGTH_FOR_SCORING:
                continue
            if line not in seen:
                line_to_node_texts[line].append(node["text"])
                seen.add(line)

    boilerplate: set = set()

    for line, node_texts in line_to_node_texts.items():
        freq = len(node_texts)
        if freq < Config.MIN_FREQUENCY:
            continue
        if is_whitelisted(line):
            continue

        ctx_sim = avg_pairwise_similarity(node_texts)
        score = boilerplate_score(freq, len(line), ctx_sim, total_nodes)

        if score >= Config.BOILERPLATE_SCORE_THRESHOLD:
            boilerplate.add(line)

    return boilerplate


# ─────────────────────────────────────────────
# PHASE 2: CLEAN INDIVIDUAL NODES
# ─────────────────────────────────────────────

def clean_node(node: dict, boilerplate_lines: set) -> str:
    """
    Apply both cleaning passes to a single node.
    Returns the cleaned text string.
    """
    text = node["text"]

    # Pass A: regex structural strip
    text = strip_structural_headers(text)

    # Pass B: remove any remaining boilerplate lines found by scoring
    cleaned = []
    for raw_line in text.split("\n"):
        if raw_line.strip() not in boilerplate_lines:
            cleaned.append(raw_line)
    text = "\n".join(cleaned)

    # Final whitespace cleanup
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def should_discard(cleaned_text: str) -> tuple:
    """Returns (discard: bool, reason: str)"""
    if len(cleaned_text) < Config.MIN_CLEAN_TEXT_LENGTH:
        return True, f"too_short ({len(cleaned_text)} chars)"
    if is_dot_leader_node(cleaned_text):
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

def normalise_pipeline(nodes: list, verbose: bool = True) -> tuple:
    """
    Full normalisation pipeline.

    Args:
        nodes: raw LlamaIndex TextNode dicts (from JSON export)
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
        node["text"] = strip_structural_headers(node["text"])
    print(f"[normalise]   Header blocks stripped from all nodes")

    # ── Phase 1: dynamic boilerplate scoring on cleaned texts ─────────
    print("[normalise] Phase 1: scoring remaining repeated lines...")
    boilerplate_lines = build_boilerplate_registry(nodes)
    print(f"[normalise]   Found {len(boilerplate_lines)} additional boilerplate lines")
    if verbose and boilerplate_lines:
        for line in sorted(boilerplate_lines):
            print(f"             └─ '{line}'")

    # ── Phase 2: apply cleaning and filter nodes ──────────────────────
    print("[normalise] Phase 2: cleaning and filtering nodes...")
    clean_nodes = []
    discard_log = []

    for node in nodes:
        cleaned_text = clean_node(node, boilerplate_lines)
        discard, reason = should_discard(cleaned_text)

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

def is_already_normalised(nodes: list) -> bool:
    """
    Returns True if this data has already been normalised.
    Prevents accidental double-processing.
    """
    signal = "Document No. SSQM 28"
    return all(signal not in n.get("text", "") for n in nodes)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Normalise LlamaIndex nodes")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log",    default="discard_log.json")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    nodes = data["nodes"]

    if is_already_normalised(nodes):
        print("[normalise] Already normalised — skipping.")
        return

    clean_nodes, discard_log = normalise_pipeline(nodes, verbose=not args.quiet)

    out = {"nodes": clean_nodes, "source_id": data.get("source_id", "1")}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[normalise] Clean nodes → {args.output}")

    with open(args.log, "w") as f:
        json.dump(discard_log, f, indent=2)
    print(f"[normalise] Discard log → {args.log}")


if __name__ == "__main__":
    main()