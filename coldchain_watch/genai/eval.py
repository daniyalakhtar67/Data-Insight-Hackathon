"""
genai/eval.py
-------------
Run the golden evaluation set on the Insight Agent and save results.

Guide requirement: "Test the agent on real data only.
  - Ask a question your documents CAN answer → it cites the right section
  - Ask a question your documents CANNOT answer → it says I don't know
  - Check every number matches stored row IDs
  - Try a misleading question → it must not invent a fact"

Results saved to model/artifacts/golden_eval.json
Dashboard reads this file to display eval results.

Run: python -m genai.eval
"""

import json
import os
import sys

sys.path.insert(0, ".")

from loguru import logger
from genai.assistant import InsightAgent

os.makedirs("model/artifacts", exist_ok=True)
OUTPUT_PATH = "model/artifacts/golden_eval.json"


def run_eval():
    print("=" * 60)
    print("ColdChain Watch — Golden Evaluation Set")
    print("=" * 60)

    agent = InsightAgent()
    results = agent.run_golden_eval()

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\nResults saved: {OUTPUT_PATH}")
    print()
    passed  = sum(1 for r in results if r["passed"] is True)
    failed  = sum(1 for r in results if r["passed"] is False)
    unknown = sum(1 for r in results if r["passed"] is None)

    print(f"Total:   {len(results)}")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Unknown: {unknown}")
    print()

    for r in results:
        status = "✅ PASS" if r["passed"] else ("❌ FAIL" if r["passed"] is False else "⚠️  N/A")
        print(f"  {status} [{r['eval_id']}] {r['question'][:70]}")
        print(f"         expected={r['expected']} | citations={r['n_citations']} | model={r['model_used']}")
        print(f"         answer: {r['answer_preview'][:100]}...")
        print()

    return results


if __name__ == "__main__":
    run_eval()
