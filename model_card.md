# Model Card: Mood Check-In Agent

Extends the Module 3 Mood Machine. The original Module 3 model card is in this repo's git history.

## 1. Overview

- **Task:** label short social-media-style posts as `positive`, `negative`, `neutral`, or `mixed`, with a confidence score and a reason. Uncertain or distress-related posts go to a human.
- **Components:** the rule-based `MoodAnalyzer` (Module 3), a TF-IDF retriever over 61 labeled posts, an optional Gemini call (prompted with the retrieved posts), a verifier, and a confidence router.
- **Intended use:** a learning project and demo of a triage-style assistant. **Not** for mental-health screening or any decision about a real person.

## 2. Data

| File | Size | Labels | Role |
|---|---|---|---|
| `data/labeled_examples.csv` | 61 posts | 21 neg / 20 pos / 11 neutral / 9 mixed | Knowledge base for retrieval |
| `data/test_set.csv` | 24 posts | 9 neg / 7 pos / 4 neutral / 4 mixed | Held-out evaluation only |

- All labels come from one person (me), so they reflect my reading of slang and tone.
- Sarcastic posts are labeled by **intended meaning** (`negative`).
- `evaluate.py` checks that no test post appears in the knowledge base.

## 3. Evaluation results

| System | Accuracy (24 held-out) | Accuracy when auto-accepted | Sent to human |
|---|---|---|---|
| Rules only (Module 3) | 0.54 | 0.54 | 0 |
| Agent offline (rules + RAG) | 0.79 | 0.86 | 3 |
| Agent + Gemini | _TODO: run with API key_ | | |

Guardrail checks: 6/6 passed. Unit tests: 15/15 passed.

| Test input | Criteria | Result |
|---|---|---|
| `""`, `"    "`, `"!!!???"` | Rejected with a reason | Pass |
| 600-character post | Rejected | Pass |
| "I want to die" | Sent to human, no label, LLM not called | Pass |
| "this burger slaps fr" | positive (slang) | Pass (0.80) |
| "Oh wonderful, it's raining on my day off" | negative (sarcasm) | Pass (0.55, borderline) |
| "I just love when my laptop crashes mid essay" | negative (sarcasm) | **Fail**: positive offline |
| "The weather is cloudy" | neutral | **Fail**: negative at 0.80 confidence |
| "Couldn't be prouder of my team" | positive | Partial: neutral, but sent to human |

## 4. Limitations and biases

- **Retrieval matches words, not feelings.** "The weather is cloudy" retrieved "I can't stand this weather" and was labeled `negative` with *high* confidence. A confident wrong answer is the worst kind of failure here.
- **Small, single-labeler data.** 61 examples cannot cover how everyone writes. Slang I know is covered ("slaps", "ate", "mid", 💀); slang I don't know falls back to the word lists. Those lists are in standard English, so people who write in dialects or slang I didn't include get worse results.
- **Confidence is hand-designed, not calibrated.** The numbers (0.55 threshold, 0.6 for rules-only, etc.) are rules I chose, not probabilities learned from data.
- **The distress guardrail is a keyword list.** It catches only phrasing on its list and will miss indirect or misspelled expressions of distress.
- **Some tuning used the test set.** I found the "I am so happy" → negative bug while looking at test results, so the 0.79 is somewhat optimistic.

## 5. Misuse and prevention

- **Possible misuse:** monitoring people's private messages, or scoring students or employees by "mood".
- **Prevention in the design:** no labels on distress posts, low-confidence posts go to a human, every decision is logged with a reason so it can be audited, and the intended-use statement above rules out decisions about real people.
- **What I would add:** a consent notice before analyzing anyone's text, and no storage of raw text in logs for a real deployment. Right now `traces.jsonl` stores the post text.

## 6. What surprised me during reliability testing

_TODO (your words). Candidates from the real results:_ adding retrieval made one *easy* case worse ("The weather is cloudy"); the first offline design let a shared word ("happy" in "I am **not** happy") flip a clearly positive post; and confidence was highest on one of the wrong answers.

## 7. Collaboration with AI

_TODO (required, in your words). Describe how you used an AI assistant on this project, then answer:_

- **One suggestion I accepted, and why:** _..._
- **One suggestion I rejected or changed, and why:** _..._

_Hint: these must be things that really happened in your own build. Possible types: a design that was over-engineered, a confidence formula you simplified, a fallback rule that caused a bug, a feature you cut for scope._
