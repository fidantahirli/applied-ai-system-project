"""
agent.py - the Mood Check-In Agent (orchestration).

For every post the agent runs the same loop and records each step in a trace:

  1. GUARDRAIL  check the input (empty? too long? distress language?)
  2. RULES      run the Module 3 rule-based MoodAnalyzer
  3. RETRIEVE   find similar labeled posts in data/labeled_examples.csv
  4. DECIDE     rules and retrieval agree?  -> accept
                they disagree / no evidence? -> ask the LLM with the retrieved
                                               examples (RAG), or fall back
                                               to the retrieval vote offline
  5. VERIFY     validate the answer; if the LLM gives an invalid answer,
                send it feedback and retry once
  6. ROUTE      low confidence -> "needs_human_review" (human-in-the-loop)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from guardrails import REVIEW_THRESHOLD, check_input, check_output
from llm_client import LLMFn, build_prompt, parse_response
from mood_analyzer import MoodAnalyzer
from retriever import Retriever

logger = logging.getLogger("mood_agent")

LOG_DIR = Path(__file__).parent / "logs"
MIN_SIMILARITY = 0.25  # retrieved examples weaker than this are not trusted on their own
STRONG_VOTE = 0.7      # share of similar posts needed to overrule the keyword rules


@dataclass
class AgentResult:
    text: str
    label: Optional[str]
    confidence: float
    status: str  # "ok" | "needs_human_review" | "rejected"
    source: str  # which path produced the label
    reason: str
    trace: List[Dict] = field(default_factory=list)


class MoodAgent:
    def __init__(
        self,
        llm: Optional[LLMFn] = None,
        retriever: Optional[Retriever] = None,
        k: int = 4,
        save_traces: bool = False,
    ) -> None:
        self.llm = llm
        self.analyzer = MoodAnalyzer()
        self.retriever = retriever or Retriever()
        self.k = k
        self.save_traces = save_traces

    # ------------------------------------------------------------------
    def run(self, raw_text) -> AgentResult:
        trace: List[Dict] = []

        def step(name: str, **details) -> None:
            trace.append({"step": name, **details})
            logger.debug("%s %s", name, details)

        # 1. GUARDRAIL --------------------------------------------------
        check = check_input(raw_text)
        step("guardrail", ok=check.ok, escalate=check.escalate, reason=check.reason)
        if not check.ok:
            return self._finish(AgentResult(str(raw_text), None, 0.0, "rejected",
                                            "guardrail", check.reason, trace))
        text = check.text
        if check.escalate:
            return self._finish(AgentResult(text, None, 0.0, "needs_human_review",
                                            "guardrail", check.reason, trace))

        # 2. RULES -------------------------------------------------------
        rules = self.analyzer.analyze(text)
        has_signal = bool(rules["positive_hits"] or rules["negative_hits"])
        step("rules", **rules)

        # 3. RETRIEVE ----------------------------------------------------
        examples = self.retriever.retrieve(text, k=self.k)
        votes = Retriever.vote(examples)
        top_sim = examples[0].similarity if examples else 0.0
        rag_label = max(votes, key=votes.get) if votes else None
        rag_share = votes.get(rag_label, 0.0) if rag_label else 0.0
        step("retrieve", top_similarity=top_sim, votes=votes,
             examples=[f'{e.text} -> {e.label} ({e.similarity})' for e in examples])

        # 4. DECIDE ------------------------------------------------------
        if has_signal and rag_label == rules["label"] and top_sim >= MIN_SIMILARITY:
            plan = "accept: rules and retrieved examples agree"
        elif self.llm is not None:
            plan = "ask LLM with retrieved examples (RAG)"
        else:
            plan = "offline fallback: retrieval vote, else rules"
        step("plan", decision=plan)

        if plan.startswith("accept"):
            label = rules["label"]
            confidence = round(0.7 + 0.25 * rag_share, 2)
            source, reason = "rules+retrieval", (
                f"Keywords {rules['positive_hits'] + rules['negative_hits']} and "
                f"{len(examples)} similar labeled posts both point to '{label}'.")
        elif plan.startswith("ask"):
            label, confidence, reason = self._ask_llm(text, examples, rules, step)
            source = "llm+retrieval"
        else:
            label, confidence, source, reason = self._offline(rules, has_signal,
                                                              rag_label, rag_share, top_sim)

        # 5. VERIFY ------------------------------------------------------
        problem = check_output(label, confidence)
        step("verify", label=label, confidence=confidence, problem=problem)
        if problem:
            return self._finish(AgentResult(text, None, 0.0, "needs_human_review",
                                            source, f"Output failed validation: {problem}", trace))

        # 6. ROUTE -------------------------------------------------------
        status = "ok" if confidence >= REVIEW_THRESHOLD else "needs_human_review"
        step("route", status=status, threshold=REVIEW_THRESHOLD)
        return self._finish(AgentResult(text, label, confidence, status, source, reason, trace))

    # ------------------------------------------------------------------
    def _ask_llm(self, text, examples, rules, step):
        """Call the LLM; if its answer is invalid, give feedback and retry once."""
        feedback = None
        for attempt in (1, 2):
            prompt = build_prompt(text, examples, rules["label"],
                                  rules["positive_hits"] + rules["negative_hits"], feedback)
            try:
                reply = parse_response(self.llm(prompt))
            except Exception as exc:  # network error, bad JSON, ...
                feedback = f"Reply could not be used ({exc})."
                step("llm", attempt=attempt, error=str(exc))
                continue
            step("llm", attempt=attempt, **reply)
            feedback = check_output(reply["label"], reply["confidence"])
            if feedback is None:
                return reply["label"], round(float(reply["confidence"]), 2), reply["reason"]
        # Two failed attempts: return an invalid label so VERIFY routes to a human.
        return None, 0.0, f"LLM failed twice: {feedback}"

    @staticmethod
    def _offline(rules, has_signal, rag_label, rag_share, top_sim):
        hits = rules["positive_hits"] + rules["negative_hits"]
        close_match = rag_label is not None and top_sim >= MIN_SIMILARITY
        # Retrieval may override the rules only if the rules found nothing, or the
        # similar posts strongly agree (e.g. sarcasm). Otherwise a shared word like
        # "happy" in "I am NOT happy" would flip "I am so happy" to negative.
        if close_match and (not has_signal or rag_share >= STRONG_VOTE):
            conf = round(0.3 + 0.5 * rag_share, 2)
            return (rag_label, conf, "retrieval vote",
                    f"{int(rag_share * 100)}% of the similar labeled posts say '{rag_label}' "
                    f"(rules said '{rules['label']}').")
        if has_signal:
            note = "similar posts were split" if close_match else "no close examples"
            return (rules["label"], 0.6, "rules only", f"Keywords {hits}; {note}.")
        return ("neutral", 0.4, "rules only",
                "No sentiment words and no similar examples; defaulting to neutral.")

    # ------------------------------------------------------------------
    def _finish(self, result: AgentResult) -> AgentResult:
        logger.info("label=%s conf=%.2f status=%s source=%s text=%r",
                    result.label, result.confidence, result.status, result.source, result.text)
        if self.save_traces:
            LOG_DIR.mkdir(exist_ok=True)
            record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      **asdict(result)}
            with open(LOG_DIR / "traces.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result


def setup_logging(verbose: bool = False) -> None:
    """Log to logs/mood_agent.log (always) and the console (warnings, or all with -v)."""
    LOG_DIR.mkdir(exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "mood_agent.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.handlers = [file_handler, console]
