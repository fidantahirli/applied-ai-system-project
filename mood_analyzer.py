# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    # Set to True to print the tokens and score for every text analyzed.
    debug = False

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

        # Counts from the most recent score_text call (used for "mixed").
        self.last_pos = 0
        self.last_neg = 0
        # (token, +1/-1) pairs from the most recent score_text call.
        self.last_hits: List[Tuple[str, int]] = []

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        - Strips leading and trailing whitespace
        - Converts everything to lowercase
        - Turns ":)" and ":(" into emoji tokens
        - Normalizes repeated characters ("soooo" -> "soo")
        - Removes punctuation and splits into word / emoji tokens
        """
        cleaned = text.strip().lower()
        cleaned = cleaned.replace(":)", " 🙂 ").replace(":(", " 😞 ")
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)  # "soooo" -> "soo"
        tokens = re.findall(
            r"[a-z]+(?:'[a-z]+)?|[\U0001F300-\U0001FAFF\u2600-\u27BF]", cleaned
        )

        if self.debug:
            print(f"  tokens: {tokens}")

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        Enhancements implemented:
          - Negation handling: if "not", "no", "never", "cannot", "without",
            or any word ending in "n't" appears in the 2 tokens before a
            sentiment word, that word's sign is flipped
            ("not happy" -> negative, "not bad" -> positive).
          - Emoji signals: some emojis count as positive or negative.
        """
        tokens = self.preprocess(text)

        negators = {"not", "no", "never", "cannot", "without"}
        pos_emoji = {"😀", "😊", "😄", "😍", "🥰", "🎉", "👍", "😂"}
        neg_emoji = {"😞", "😢", "😭", "😡", "😠", "💔", "👎", "😩"}

        score = 0
        self.last_pos = 0
        self.last_neg = 0
        self.last_hits = []

        for i, token in enumerate(tokens):
            if token in self.positive_words or token in pos_emoji:
                polarity = 1
            elif token in self.negative_words or token in neg_emoji:
                polarity = -1
            else:
                continue

            # Negation: a negator in the previous 2 tokens flips the sign.
            window = tokens[max(0, i - 2):i]
            if any(w in negators or w.endswith("n't") for w in window):
                polarity = -polarity

            score += polarity
            negated = (token in self.positive_words or token in pos_emoji) != (polarity > 0)
            self.last_hits.append((f"not {token}" if negated else token, polarity))
            if polarity > 0:
                self.last_pos += 1
            else:
                self.last_neg += 1

        if self.debug:
            print(f"  score: {score} (pos={self.last_pos}, neg={self.last_neg})")

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The mapping is:
          - both positive and negative signals present -> "mixed"
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"
        """
        score = self.score_text(text)

        if self.last_pos > 0 and self.last_neg > 0:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Structured output (used by the agent in agent.py)
    # ---------------------------------------------------------------------

    def analyze(self, text: str) -> Dict:
        """Label + the evidence behind it, so other components can reason about it."""
        label = self.predict_label(text)
        return {
            "label": label,
            "score": sum(p for _, p in self.last_hits),
            "positive_hits": [t for t, p in self.last_hits if p > 0],
            "negative_hits": [t for t, p in self.last_hits if p < 0],
        }

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """Explain the label using the same logic as score_text (incl. negation)."""
        result = self.analyze(text)
        return (
            f"Score = {result['score']} "
            f"(positive: {result['positive_hits']}, negative: {result['negative_hits']})"
        )
