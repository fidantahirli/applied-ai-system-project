"""
guardrails.py - safety checks that run BEFORE and AFTER the AI makes a decision.

Input guardrails  : reject empty / too-long / non-text input, and catch posts
                    that may signal someone is in distress. Those go straight to
                    a human instead of getting an automatic mood label.
Output guardrails : make sure whatever the AI returns is a valid label with a
                    sensible confidence, otherwise flag it for human review.
"""

import re
from dataclasses import dataclass
from typing import Optional

ALLOWED_LABELS = {"positive", "negative", "neutral", "mixed"}
MAX_CHARS = 500
REVIEW_THRESHOLD = 0.55  # below this confidence, a human should look at it

# A small, deliberately conservative list. A keyword list can never catch every
# way a person might express distress -- that limitation is documented in the
# model card. The point is: when in doubt, route to a human, never auto-label.
DISTRESS_PATTERNS = [
    r"\bkill (my ?self|me)\b",
    r"\bwant(ed)? to die\b",
    r"\bend (it all|my life)\b",
    r"\bhurt(ing)? my ?self\b",
    r"\bno reason to live\b",
    r"\bsuicid",
    r"\bself[- ]harm",
]


@dataclass
class InputCheck:
    ok: bool
    text: str = ""
    reason: Optional[str] = None
    escalate: bool = False  # True = send to a human immediately


def check_input(raw) -> InputCheck:
    if not isinstance(raw, str):
        return InputCheck(ok=False, reason="Input must be text.")
    text = raw.strip()
    if not text:
        return InputCheck(ok=False, reason="Input is empty.")
    if len(text) > MAX_CHARS:
        return InputCheck(ok=False, reason=f"Input is longer than {MAX_CHARS} characters.")
    if not re.search(r"[A-Za-z\U0001F300-\U0001FAFF☀-➿]", text):
        return InputCheck(ok=False, reason="Input has no words or emojis to analyze.")
    lowered = text.lower()
    for pattern in DISTRESS_PATTERNS:
        if re.search(pattern, lowered):
            return InputCheck(
                ok=True,
                text=text,
                escalate=True,
                reason="Possible distress language detected; routed to a human reviewer.",
            )
    return InputCheck(ok=True, text=text)


def check_output(label, confidence) -> Optional[str]:
    """Return a problem description, or None if the output is valid."""
    if label not in ALLOWED_LABELS:
        return f"Label {label!r} is not one of {sorted(ALLOWED_LABELS)}."
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        return f"Confidence {confidence!r} is not a number between 0 and 1."
    return None
