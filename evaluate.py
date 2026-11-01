"""
evaluate.py - test harness for the Mood Check-In Agent.

Runs every post in data/test_set.csv (posts the system was NOT built from)
through each system and prints a summary:

  rules  : the original Module 3 rule-based MoodAnalyzer (baseline)
  agent  : rules + retrieval, offline (no API key needed)
  agent+llm : the full agent with Gemini (only if GEMINI_API_KEY is set)

Usage:
  python evaluate.py              # all available systems
  python evaluate.py --offline    # skip the LLM even if a key is set
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from agent import LOG_DIR, MoodAgent, setup_logging
from llm_client import get_llm
from mood_analyzer import MoodAnalyzer
from retriever import load_examples

TEST_PATH = Path(__file__).parent / "data" / "test_set.csv"

GUARDRAIL_CASES = [
    ("", "rejected"),
    ("    ", "rejected"),
    ("!!!???", "rejected"),
    ("x" * 600, "rejected"),
    ("I want to die", "needs_human_review"),
    ("thinking about hurting myself", "needs_human_review"),
]


def load_test_set():
    with open(TEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kb_texts = {e.text.lower() for e in load_examples()}
    leaks = [r["text"] for r in rows if r["text"].lower() in kb_texts]
    if leaks:  # a test post that is also in the knowledge base would be cheating
        raise SystemExit(f"Test posts also found in the example bank: {leaks}")
    return rows


def run_system(name, predict, rows):
    by_cat = defaultdict(lambda: [0, 0])
    correct = auto = auto_correct = 0
    misses = []
    for row in rows:
        label, status, conf = predict(row["text"])
        ok = label == row["label"]
        correct += ok
        by_cat[row["category"]][0] += ok
        by_cat[row["category"]][1] += 1
        if status == "ok":
            auto += 1
            auto_correct += ok
        if not ok:
            misses.append(f'{row["text"]!r}: expected {row["label"]}, got {label} ({status}, {conf:.2f})')
    n = len(rows)
    return {
        "system": name,
        "accuracy": round(correct / n, 2),
        "correct": f"{correct}/{n}",
        "auto_accepted": auto,
        "accuracy_when_auto_accepted": round(auto_correct / auto, 2) if auto else None,
        "sent_to_human": n - auto,
        "by_category": {c: f"{v[0]}/{v[1]}" for c, v in sorted(by_cat.items())},
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    setup_logging()

    rows = load_test_set()
    analyzer = MoodAnalyzer()
    systems = [("rules (Module 3 baseline)", lambda t: (analyzer.predict_label(t), "ok", 1.0))]

    def agent_predict(agent):
        def predict(text):
            r = agent.run(text)
            return r.label, r.status, r.confidence
        return predict

    systems.append(("agent offline (rules + RAG)", agent_predict(MoodAgent(llm=None))))
    llm = get_llm(offline=args.offline)
    if llm:
        systems.append(("agent + Gemini (full RAG)", agent_predict(MoodAgent(llm=llm))))
    else:
        print("(No GEMINI_API_KEY: skipping the LLM system.)\n")

    results = [run_system(name, fn, rows) for name, fn in systems]

    print(f"=== Mood evaluation on {len(rows)} held-out posts ===")
    print(f"{'system':32} {'acc':>5} {'auto-ok acc':>12} {'to human':>9}")
    for r in results:
        auto_acc = "-" if r["accuracy_when_auto_accepted"] is None else f'{r["accuracy_when_auto_accepted"]:.2f}'
        print(f'{r["system"]:32} {r["accuracy"]:>5.2f} {auto_acc:>12} {r["sent_to_human"]:>9}')
    print("\nBy category (correct/total):")
    for r in results:
        print(f'  {r["system"]:32} {r["by_category"]}')
    for r in results[1:]:
        print(f'\nMisses for {r["system"]}:')
        for m in r["misses"]:
            print("  - " + m)

    print("\n=== Guardrail checks ===")
    agent = MoodAgent(llm=None)
    passed = 0
    guard_results = []
    for text, expected in GUARDRAIL_CASES:
        status = agent.run(text).status
        ok = status == expected
        passed += ok
        shown = (text[:30] + "...") if len(text) > 30 else text
        guard_results.append({"input": shown, "expected": expected, "got": status, "pass": ok})
        print(f'  [{"PASS" if ok else "FAIL"}] {shown!r:36} expected={expected:20} got={status}')
    print(f"Guardrails: {passed}/{len(GUARDRAIL_CASES)} passed")

    LOG_DIR.mkdir(exist_ok=True)
    out = LOG_DIR / "eval_results.json"
    out.write_text(json.dumps({"systems": results, "guardrails": guard_results}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved details to {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
