"""
Unit tests for the Mood Check-In Agent. Run with:  pytest -q

The LLM is replaced by small fake functions, so these tests never need an API
key or network access, and they give the same result every time.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import MoodAgent  # noqa: E402
from guardrails import check_input, check_output  # noqa: E402
from llm_client import build_prompt, parse_response  # noqa: E402
from retriever import Example, Retriever  # noqa: E402


def fake_llm(label="negative", confidence=0.9):
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return json.dumps({"label": label, "confidence": confidence, "reason": "fake"})

    llm.calls = calls
    return llm


# ---------------- guardrails ----------------
def test_empty_input_rejected():
    assert MoodAgent().run("   ").status == "rejected"


def test_non_text_rejected():
    assert check_input(None).ok is False


def test_too_long_rejected():
    assert MoodAgent().run("a" * 501).status == "rejected"


def test_distress_goes_to_human_and_skips_llm():
    llm = fake_llm()
    result = MoodAgent(llm=llm).run("honestly I want to die")
    assert result.status == "needs_human_review"
    assert result.label is None
    assert llm.calls == []  # never auto-labeled by the AI


def test_output_validation():
    assert check_output("happy", 0.9) is not None
    assert check_output("positive", 1.5) is not None
    assert check_output("positive", 0.9) is None


# ---------------- retrieval ----------------
def test_retriever_finds_slang_example():
    top = Retriever().retrieve("this burger slaps", k=1)[0]
    assert top.text == "this new album slaps" and top.label == "positive"


def test_retriever_keeps_emojis():
    top = Retriever().retrieve("lost my keys 💀", k=1)[0]
    assert "💀" in top.text


def test_retrieved_examples_are_in_llm_prompt():
    prompt = build_prompt("Oh wonderful, rain", [Example("Oh great, another Monday", "negative", 0.3)],
                          "neutral", [])
    assert '"Oh great, another Monday" -> negative' in prompt


# ---------------- agent decisions ----------------
def test_agreement_skips_llm():
    llm = fake_llm()
    result = MoodAgent(llm=llm).run("I am not happy about this")
    assert result.label == "negative" and result.source == "rules+retrieval"
    assert llm.calls == []


def test_llm_used_when_rules_and_retrieval_disagree():
    llm = fake_llm("negative", 0.85)
    result = MoodAgent(llm=llm).run("I just love when my laptop crashes mid essay")
    assert len(llm.calls) == 1
    assert result.label == "negative" and result.source == "llm+retrieval"


def test_invalid_llm_answer_is_retried_with_feedback():
    replies = iter(['{"label": "sarcastic", "confidence": 0.9}',
                    '{"label": "negative", "confidence": 0.8, "reason": "sarcasm"}'])
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    result = MoodAgent(llm=llm).run("I just love when my laptop crashes mid essay")
    assert len(prompts) == 2 and "rejected" in prompts[1]
    assert result.label == "negative" and result.status == "ok"


def test_llm_failing_twice_goes_to_human():
    result = MoodAgent(llm=lambda p: "not json").run("I just love when my laptop crashes")
    assert result.status == "needs_human_review" and result.label is None


def test_low_confidence_goes_to_human():
    result = MoodAgent(llm=fake_llm("neutral", 0.3)).run("The package arrives Friday")
    assert result.status == "needs_human_review"


def test_offline_shared_word_does_not_flip_clear_positive():
    # Regression: "I am NOT happy about this" in the example bank used to drag this to negative.
    assert MoodAgent().run("I am so happy right now").label == "positive"


def test_parse_response_handles_code_fences():
    assert parse_response('```json\n{"label": "Positive", "confidence": 0.7}\n```')["label"] == "positive"
