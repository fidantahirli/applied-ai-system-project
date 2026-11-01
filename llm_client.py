"""
llm_client.py - optional Gemini connection + the RAG prompt.

The agent only needs "something that takes a prompt string and returns a
string". That keeps the LLM swappable:
  - GeminiLLM  : real calls (needs GEMINI_API_KEY and the google-genai package)
  - a fake      : used in tests so they run offline and give the same result every time
  - None        : offline mode; the agent falls back to rules + retrieval only
"""

import json
import logging
import os
import re
from typing import Callable, List, Optional

from retriever import Example

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], str]
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        from google import genai  # imported here so offline mode never needs the package

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.0},
        )
        return response.text or ""


def get_llm(offline: bool = False) -> Optional[LLMFn]:
    """Return a Gemini client if possible, else None (offline mode)."""
    if offline:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("No GEMINI_API_KEY set; running offline (rules + retrieval).")
        return None
    try:
        return GeminiLLM(api_key)
    except Exception as exc:  # missing package, bad key format, etc.
        logger.warning("Could not start Gemini (%s); running offline.", exc)
        return None


def build_prompt(
    text: str,
    examples: List[Example],
    rule_label: str,
    rule_hits: List[str],
    feedback: Optional[str] = None,
) -> str:
    """The retrieval-augmented prompt: retrieved examples become few-shot context."""
    if examples:
        shots = "\n".join(f'- "{e.text}" -> {e.label}' for e in examples)
    else:
        shots = "- (no similar examples found)"
    prompt = f"""You classify the mood of short social media posts.
Allowed labels: positive, negative, neutral, mixed.
- Sarcasm is labeled by what the writer MEANS (e.g. "Oh great, another Monday" -> negative).
- Use "mixed" only when the post clearly contains both good and bad feelings.

Labeled posts similar to the new one (retrieved from our dataset):
{shots}

A keyword rule system guessed: {rule_label} (sentiment words found: {rule_hits or 'none'}).
It cannot understand sarcasm or slang, so treat it only as a hint.

New post: "{text}"

Reply with JSON only:
{{"label": "<one allowed label>", "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}"""
    if feedback:
        prompt += f"\n\nYour previous answer was rejected: {feedback} Try again."
    return prompt


def parse_response(raw: str) -> dict:
    """Pull the JSON object out of an LLM reply. Raises ValueError if impossible."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)  # tolerates ```json fences
    if not match:
        raise ValueError(f"No JSON object in LLM reply: {raw[:120]!r}")
    data = json.loads(match.group(0))
    return {
        "label": str(data.get("label", "")).strip().lower(),
        "confidence": data.get("confidence"),
        "reason": str(data.get("reason", "")).strip(),
    }
