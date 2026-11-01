"""
retriever.py - the "R" in RAG for the Mood Check-In Agent.

The knowledge base is a CSV of labeled example posts (data/labeled_examples.csv).
Given a new post, the retriever finds the k most similar labeled posts using
TF-IDF vectors and cosine similarity.

Those retrieved examples are then used two ways by the agent:
  1. As few-shot context in the LLM prompt (retrieval-augmented generation).
  2. As a similarity-weighted "vote" when no LLM is available (offline mode).
"""

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_KB_PATH = Path(__file__).parent / "data" / "labeled_examples.csv"
ALLOWED_LABELS = {"positive", "negative", "neutral", "mixed"}

# Negation words carry meaning for mood, so we do NOT treat them as stop words.
_KEEP = {"not", "no", "never", "nothing", "nobody", "cannot", "but"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - _KEEP

# Same idea as MoodAnalyzer.preprocess: keep words AND emojis as tokens.
# (sklearn's default tokenizer silently drops emojis.)
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[\U0001F300-\U0001FAFF☀-➿]")


def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


@dataclass
class Example:
    text: str
    label: str
    similarity: float = 0.0


def load_examples(path: Path = DEFAULT_KB_PATH) -> List[Example]:
    """Load and validate the labeled example bank."""
    examples: List[Example] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row_num, row in enumerate(csv.DictReader(f), start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not text or label not in ALLOWED_LABELS:
                # Guardrail on the data itself: skip bad rows instead of crashing.
                logger.warning("Skipping bad row %d in %s: %r", row_num, path.name, row)
                continue
            examples.append(Example(text=text, label=label))
    if not examples:
        raise ValueError(f"No valid examples found in {path}")
    logger.info("Loaded %d examples from %s", len(examples), path.name)
    return examples


class Retriever:
    def __init__(self, examples: Optional[List[Example]] = None) -> None:
        self.examples = examples if examples is not None else load_examples()
        # ngram_range=(1, 2) lets "not happy" become its own feature.
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform([e.text for e in self.examples])

    def retrieve(self, text: str, k: int = 4) -> List[Example]:
        """Return the k most similar examples (similarity > 0), best first."""
        query = self.vectorizer.transform([text])
        scores = cosine_similarity(query, self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [
            Example(self.examples[i].text, self.examples[i].label, round(float(scores[i]), 3))
            for i in ranked[:k]
            if scores[i] > 0
        ]
        return results

    @staticmethod
    def vote(examples: List[Example]) -> Dict[str, float]:
        """Similarity-weighted vote share per label, e.g. {'negative': 0.7, 'mixed': 0.3}."""
        totals: Dict[str, float] = defaultdict(float)
        for ex in examples:
            totals[ex.label] += ex.similarity
        grand = sum(totals.values())
        if grand == 0:
            return {}
        return {label: round(v / grand, 3) for label, v in totals.items()}
