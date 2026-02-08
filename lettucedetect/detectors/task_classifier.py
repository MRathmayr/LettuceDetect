"""Heuristic task-type classifier for cascade routing.

Classifies RAG inputs as 'qa', 'summarization', 'data2txt', or 'unknown'
to route them through the appropriate cascade stages. The reading probe
only helps on QA inputs (AUROC +0.093); it hurts on all other types.
"""

from __future__ import annotations

import re

# Summarization: imperative instructions to condense/summarize.
# Covers RAGTruth ("Summarize the following news within N words:"),
# HaluEval (injected "Summarize the following document."),
# and production patterns ("TL;DR", "key takeaways", etc.)
_SUMMARIZATION_RE = re.compile(
    r"(?:^|\b)("
    r"summarize|summarise|sum up|summing up"
    r"|write a summary|provide a summary|give a summary"
    r"|tl;?\s*dr"
    r"|condense|distill|distil"
    r"|brief(?:ly)?\s+(?:describe|explain|outline|recap)"
    r"|key takeaways|key points|main points|main ideas"
    r"|recap|synopsis|gist"
    r"|shorten this|make (?:this |it )?(?:shorter|more concise)"
    r")\b",
    re.IGNORECASE,
)

# Data2txt: instructions to convert structured data to text.
# Covers RAGTruth ("Write an objective overview...structured data in JSON")
# and production patterns ("convert data to text", "describe this table", etc.)
_DATA2TXT_QUESTION_RE = re.compile(
    r"(?:^|\b)("
    r"write an? (?:objective )?overview"
    r"|(?:convert|transform|turn)\b.{0,20}\b(?:to |into )(?:text|prose|narrative|natural language)"
    r"|based on the (?:following |provided |above )?(?:data|table|json|structured)"
    r"|describe (?:the |this )?(?:data|table|record|entries)"
    r"|generate (?:a )?(?:text|description|report|narrative) (?:from|based on)"
    r"|data[- ]to[- ]text"
    r")\b",
    re.IGNORECASE,
)

# Dialogue: multi-turn conversation format (HaluEval dialogue uses [Human]:...[Assistant]:...)
_DIALOGUE_RE = re.compile(
    r"\[(?:Human|User|Assistant|Bot|System)\]\s*:", re.IGNORECASE
)


def _looks_structured(text: str) -> bool:
    """Detect structured/tabular context (tables, JSON, key-value pairs, CSV)."""
    if not text or len(text) < 20:
        return False
    lines = text.strip().split("\n")
    # JSON-like: starts with { or [
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        return True
    if len(lines) < 3:
        return False
    # Pipe-delimited table rows (e.g., "| col1 | col2 |")
    pipe_rows = sum(1 for line in lines if line.count("|") >= 2)
    if pipe_rows >= 3:
        return True
    # Tab-separated rows
    tab_rows = sum(1 for line in lines if line.count("\t") >= 2)
    if tab_rows >= 3:
        return True
    # CSV-like: consistent comma-separated fields across rows
    if len(lines) >= 3:
        comma_counts = [line.count(",") for line in lines[:10]]
        if all(c >= 2 for c in comma_counts) and max(comma_counts) - min(comma_counts) <= 1:
            return True
    # Key-value pairs: "key: value" or "key = value" pattern
    kv_rows = sum(1 for line in lines if re.match(r"^\s*\w[\w\s]*[:=]", line))
    if kv_rows >= 3 and kv_rows / len(lines) > 0.5:
        return True
    return False


class TaskTypeClassifier:
    """Classify RAG input into task type for routing.

    Priority order:
    1. Summarization keywords in question (imperative instructions)
    2. Data2txt keywords in question (conversion instructions)
    3. Dialogue pattern detection
    4. Structured context detection (JSON, tables, CSV)
    5. Default: QA if question present, unknown otherwise
    """

    def classify(self, question: str | None, context: list[str] | None) -> str:
        """Classify input as 'qa', 'summarization', 'data2txt', or 'unknown'."""
        if question:
            if _SUMMARIZATION_RE.search(question):
                return "summarization"
            if _DATA2TXT_QUESTION_RE.search(question):
                return "data2txt"
            if _DIALOGUE_RE.search(question):
                return "unknown"

        if context and context[0] and _looks_structured(context[0]):
            return "data2txt"

        if question:
            return "qa"

        return "unknown"
