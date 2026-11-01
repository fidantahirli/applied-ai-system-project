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
| Agent + Gemini | Not evaluated (offline only) | | |

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

The biggest surprise was that adding retrieval made one *easy* case worse. "The weather is cloudy" was correct with the old rules, but the agent labeled it `negative` with 0.80 confidence, because the word "weather" matched "I can't stand this weather" in the knowledge base. That was one of the most confident answers in the whole test, and it was wrong. Higher confidence did not mean more correct.

I was also surprised by how easily one shared word could flip a result. In the first offline design, "I am so happy right now" came back `negative` because its closest match was "I am **not** happy about this." The fix was to let retrieval overrule the keyword rules only when the rules found nothing, or when at least 70% of the similar posts agreed. Overall, accuracy went up from 0.54 to 0.79, but the failures taught me more than the score did: retrieval matches words, not meaning.

## 7. Collaboration with AI

I used Claude as a coding partner throughout this project. It helped me plan the architecture (rules → retrieval → planner → verifier → human review), write the new modules, design the tests and evaluation script, and draft the documentation. I tested every step myself before committing it: I ran each file on example posts, checked that `main.py` still gave the original 0.76, and ran `pytest` and `evaluate.py`.

- **One suggestion I accepted, and why:** making the system work fully offline, and testing it with a fake LLM instead of real Gemini calls. I accepted this because it makes the project reproducible. Anyone can clone and run it without an API key, and the tests give the same result every time instead of depending on the internet or API limits.

- **One suggestion I rejected or changed, and why:** Claude first gave me the whole finished project as one zip to drop into my repo. I didn't accept that as written. I added it one piece at a time instead (data, retriever, analyzer, guardrails, LLM client, agent, tests, evaluation, docs), running and checking each part before committing. I changed the approach because the assignment requires meaningful commit history, and because building it step by step meant I understood and could explain each component. I also didn't take its shortcut for setting up the repo and followed the official Step 0 exactly (clone the original repo, then point it at a new remote).

- **A design suggestion that had to be revised:** Claude's first offline fallback let the retrieval vote overrule the keyword rules whenever they disagreed. Testing showed that this flipped "I am so happy right now" to `negative`. We changed the rule so retrieval needs at least 70% agreement among the similar posts before it can overrule clear keywords, and added a regression test so the bug can't come back.
