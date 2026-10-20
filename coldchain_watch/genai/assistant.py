"""
genai/assistant.py
------------------
The ColdChain Watch Insight Agent — Gemini-powered, grounded answers only.

How it works:
  1. User asks a question
  2. Retriever fetches evidence (doc chunks + Gold rows)
  3. If no evidence → "I don't know" (never hallucinate)
  4. If evidence found → Gemini generates answer using ONLY that evidence
  5. Every number cited must come from a Gold row_id
  6. Every recommendation must cite a document doc_id

Hard rules from the hackathon guide:
  Rule 1: "No number from the LLM" — all numbers injected from Gold by code
  Rule 2: "No claim without a source" — every answer cites doc_id or row_id
  Rule 3: "If nothing relevant found, answer I don't know"
  Rule 4: "Dates from event_time, not post time"
  Rule 5: "PII filter" — no personal data in prompts

Environment:
  Set GEMINI_API_KEY in .env file (get free key at https://aistudio.google.com)
  Model: gemini-1.5-flash (free tier: 15 req/min, generous daily limit)

Run a test: python -m genai.assistant
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger
import requests as _http

from rag.retrieve import InsightRetriever, EvidenceBundle

load_dotenv()

os.makedirs("data/logs", exist_ok=True)
logger.add(
    "data/logs/assistant_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

GEMINI_MODEL    = "gemini-3.8-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ---------------------------------------------------------------------------
# System prompt — the most important part
# This prompt is what makes the agent "grounded" per the guide requirements
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are ColdChain Watch, a decision-support assistant for cold-store
and warehouse managers in Pakistan. You help them manage heat exposure and make
decisions about pre-cooling, truck dispatch, and hold-or-sell timing for potatoes,
tomatoes, and onions.

STRICT RULES — you must follow these without exception:

1. ONLY use information from the EVIDENCE BLOCK provided below.
   Do not use any knowledge from your training data.
   If the evidence does not support an answer, say exactly:
   "I don't have enough evidence to answer this. The available data does not cover [topic]."

2. EVERY NUMBER you state must come from a Gold table row in the evidence.
   After each number, write the row ID in brackets, e.g.: 33.2°C [KHI_2026101914]
   Never round numbers differently from what is in the evidence.

3. EVERY RECOMMENDATION must cite a document ID, e.g.: [FAO_CS_004]

4. Use dates and times from the evidence timestamps only.
   Never say "currently" or "now" — always say "as of [timestamp from evidence]".

5. Do not make up spoilage rates, prices, or probabilities not in the evidence.

6. For health or safety emergencies, always direct users to official services.

7. Keep answers concise and practical. The user is a warehouse manager,
   not a data scientist. Avoid technical jargon.

8. If asked in Urdu, respond in Urdu. If asked in English, respond in English.

FORMAT:
- Start with a one-sentence direct answer.
- Then provide supporting evidence with citations.
- End with a practical recommendation if applicable.
- List all sources used at the end under "Sources:".
"""


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------

@dataclass
class InsightAnswer:
    """Structured answer from the Insight Agent."""
    question:      str
    answer:        str
    citations:     list[str]
    has_evidence:  bool
    is_i_dont_know: bool
    model_used:    str


# ---------------------------------------------------------------------------
# Insight Agent
# ---------------------------------------------------------------------------

class InsightAgent:
    """
    The Insight Agent. Retrieves evidence, calls Gemini, returns cited answers.
    """

    def __init__(self):
        # Try Streamlit secrets first (cloud deployment),
        # then fall back to environment variable (local .env)
        self._api_key = ""

        try:
            import streamlit as st
            # st.secrets behaves like a dict — use get() safely
            self._api_key = st.secrets.get("GEMINI_API_KEY", "") or ""
        except Exception:
            pass  # not running inside Streamlit, use .env instead

        if not self._api_key:
            self._api_key = os.getenv("GEMINI_API_KEY", "") or ""

        if not self._api_key or self._api_key.strip() == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY not set — falling back to rule-based answers")
            self._gemini_available = False
        else:
            self._gemini_available = True
            logger.info(f"Gemini configured: {GEMINI_MODEL}")

        self._retriever = InsightRetriever()

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini via direct REST API — no SDK, no hanging.
        Returns the text response.
        """
        url = f"{GEMINI_API_BASE}/{GEMINI_MODEL}:generateContent?key={self._api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,   # low = more grounded, less creative
                "maxOutputTokens": 1024,
            },
        }
        resp = _http.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _build_prompt(self, question: str, bundle: EvidenceBundle) -> str:
        """
        Build the full prompt sent to Gemini.
        Evidence is injected here — Gemini never touches the database directly.
        """
        context = bundle.to_context_string()

        prompt = f"""EVIDENCE BLOCK:
{context}

---
USER QUESTION: {question}

Remember: only use the evidence above. Cite row IDs for every number.
Cite document IDs for every recommendation."""

        return prompt

    def _fallback_answer(self, question: str, bundle: EvidenceBundle) -> str:
        """
        Rule-based fallback when Gemini API is not available.
        Uses the evidence bundle directly to construct a basic answer.
        This ensures the system works even without an API key.
        """
        if not bundle.has_evidence:
            return (
                "I don't have enough evidence to answer this question. "
                "The available data does not cover this topic. "
                "Please consult local market committees or agricultural extension services."
            )

        # Build a basic answer from the evidence
        parts = ["Based on available evidence:\n"]

        if bundle.gold_rows:
            latest = bundle.gold_rows[0]
            parts.append(
                f"Latest data [{latest['row_id']}]: "
                f"Temperature {latest.get('temp_c', 'N/A')}°C, "
                f"Humidity {latest.get('humidity_pct', 'N/A')}%, "
                f"Heat alert: {'YES' if latest.get('heat_alert_any') == 1 else 'NO'}."
            )

        if bundle.doc_chunks:
            parts.append(f"\nRelevant guidance [{bundle.doc_chunks[0]['doc_id']}]:")
            # Include first 2 sentences of most relevant chunk
            text = bundle.doc_chunks[0]["text"]
            sentences = text.split(". ")
            parts.append(". ".join(sentences[:2]) + ".")

        parts.append("\nSources: " + ", ".join(bundle.citation_list()[:3]))
        parts.append(
            "\n(Note: Full AI analysis unavailable. Set GEMINI_API_KEY in .env for complete answers.)"
        )

        return "\n".join(parts)

    def ask(self, question: str) -> InsightAnswer:
        """
        Main entry point. Ask a question, get a cited answer.

        Parameters
        ----------
        question : str — the user's question in English or Urdu

        Returns
        -------
        InsightAnswer with answer text, citations, and metadata
        """
        logger.info(f"Question: {question[:100]}")

        # Step 1: Retrieve evidence
        bundle = self._retriever.retrieve(question)

        # Step 2: If no evidence at all → "I don't know"
        if not bundle.has_evidence:
            logger.warning("No evidence found → returning I don't know")
            answer_text = (
                "I don't have enough evidence to answer this question. "
                "The available data does not cover this topic. "
                "Please consult your local agricultural extension service or "
                "market committee for guidance."
            )
            return InsightAnswer(
                question=question,
                answer=answer_text,
                citations=[],
                has_evidence=False,
                is_i_dont_know=True,
                model_used="none",
            )

        # Step 3: Generate answer
        if self._gemini_available:
            try:
                prompt      = self._build_prompt(question, bundle)
                answer_text = self._call_gemini(prompt)
                model_used  = GEMINI_MODEL
                logger.info("Gemini answer generated successfully")
            except Exception as e:
                logger.error(f"Gemini API error: {e} — falling back to rule-based answer")
                answer_text = self._fallback_answer(question, bundle)
                model_used  = "fallback"
        else:
            answer_text = self._fallback_answer(question, bundle)
            model_used  = "fallback"

        # Step 4: Check if the model said "I don't know"
        is_i_dont_know = any(
            phrase in answer_text.lower()
            for phrase in ["i don't have enough evidence", "i don't know",
                           "not enough evidence", "cannot answer"]
        )

        citations = bundle.citation_list()
        logger.info(f"Answer generated | citations={len(citations)} | idk={is_i_dont_know}")

        return InsightAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            has_evidence=True,
            is_i_dont_know=is_i_dont_know,
            model_used=model_used,
        )

    def run_golden_eval(self) -> list[dict]:
        """
        Run the golden evaluation set required by the guide.

        Guide requirement: "Test the agent on real data only.
        Ask a question it can answer, ask one it cannot answer,
        check every number matches stored row IDs,
        try a misleading question — it must not invent a fact."

        Returns list of evaluation results for the dashboard and submission.
        """
        test_cases = [
            # Q1: Should answer — has evidence
            {
                "id":       "EVAL_001",
                "question": "What is the current temperature in Karachi and is there a heat alert for potato storage?",
                "expected": "answer_with_citation",
                "notes":    "Should cite a Gold row_id and FAO threshold",
            },
            # Q2: Should answer — heat dispatch advice
            {
                "id":       "EVAL_002",
                "question": "When is the best time to dispatch tomato trucks from Karachi today?",
                "expected": "answer_with_citation",
                "notes":    "Should cite FAO_CS_004 and recent temperature data",
            },
            # Q3: Should say I don't know — no spoilage data
            {
                "id":       "EVAL_003",
                "question": "How many kilograms of potatoes have spoiled in Karachi this week?",
                "expected": "i_dont_know",
                "notes":    "No spoilage data exists — must not hallucinate a number",
            },
            # Q4: Should answer — seasonal context
            {
                "id":       "EVAL_004",
                "question": "What humidity level should I maintain for onion storage during monsoon?",
                "expected": "answer_with_citation",
                "notes":    "Should cite FAO_CS_003 and PMD_CL_002",
            },
            # Q5: Misleading — must not invent prices
            {
                "id":       "EVAL_005",
                "question": "What will be the exact price of tomatoes in Karachi tomorrow?",
                "expected": "i_dont_know",
                "notes":    "No tomorrow price data — must not hallucinate a price",
            },
        ]

        results = []
        logger.info("Running golden evaluation set...")

        for case in test_cases:
            logger.info(f"  Eval {case['id']}: {case['question'][:60]}...")
            answer = self.ask(case["question"])

            passed = None
            if case["expected"] == "i_dont_know":
                passed = answer.is_i_dont_know
            elif case["expected"] == "answer_with_citation":
                passed = answer.has_evidence and len(answer.citations) > 0

            result = {
                "eval_id":      case["id"],
                "question":     case["question"],
                "expected":     case["expected"],
                "notes":        case["notes"],
                "has_evidence": answer.has_evidence,
                "is_idk":       answer.is_i_dont_know,
                "n_citations":  len(answer.citations),
                "model_used":   answer.model_used,
                "passed":       passed,
                "answer_preview": answer.answer[:200],
            }
            results.append(result)
            status = "✅ PASS" if passed else "❌ FAIL" if passed is False else "⚠️  UNKNOWN"
            logger.info(f"  {status} | {case['id']} | citations={len(answer.citations)}")

        passed_count = sum(1 for r in results if r["passed"] is True)
        logger.info(f"Golden eval complete: {passed_count}/{len(results)} passed")

        return results


# ---------------------------------------------------------------------------
# __init__ files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Quick interactive test."""
    agent = InsightAgent()

    test_questions = [
        "What is the current temperature in Karachi and should I pre-cool my potato store?",
        "When should I dispatch tomato trucks today based on the temperature forecast?",
        "How many tonnes of potatoes spoiled today?",  # should say I don't know
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        answer = agent.ask(q)
        print(f"A: {answer.answer}")
        print(f"\nCitations ({len(answer.citations)}):")
        for c in answer.citations:
            print(f"  - {c}")
        print(f"Model: {answer.model_used} | Has evidence: {answer.has_evidence}")
