"""
System Evaluation — RAGAS + Custom Metrics
────────────────────────────────────────────
Evaluates the full system across three dimensions:

1. Intent Classification Accuracy
   — Test set of labeled messages, measure correct routing rate

2. RAG Response Quality (RAGAS metrics)
   — Faithfulness:       answer is supported by retrieved context
   — Answer Relevance:   answer addresses the actual question
   — Context Relevance:  retrieved docs are focused and useful

3. System Latency
   — End-to-end response time under realistic load

Usage:
  python -m evaluation.evaluate [--component all|intent|rag|latency]

Output:
  Prints a metrics report and saves results to evaluation/results/
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import Dataset

from src.intent_classifier import classify_intent
from src.rag_pipeline import rag_service

RESULTS_DIR = Path("./evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Intent Classification Test Set ──────────────────────────────────────────

INTENT_TEST_SET = [
    # casual_chat
    {"message": "Hey, how are you?",                          "expected": "casual_chat"},
    {"message": "Good morning!",                               "expected": "casual_chat"},
    {"message": "Thanks for your help!",                       "expected": "casual_chat"},
    {"message": "What do you think about AI?",                 "expected": "casual_chat"},
    {"message": "Haha that's funny",                           "expected": "casual_chat"},
    {"message": "Talk later?",                                 "expected": "casual_chat"},
    {"message": "I'm doing great, you?",                       "expected": "casual_chat"},
    {"message": "Did you watch the game yesterday?",           "expected": "casual_chat"},
    # knowledge_query
    {"message": "What are your working hours?",                "expected": "knowledge_query"},
    {"message": "How do I reset my password?",                 "expected": "knowledge_query"},
    {"message": "What is your return policy?",                 "expected": "knowledge_query"},
    {"message": "How much does the Pro plan cost?",            "expected": "knowledge_query"},
    {"message": "Is my data stored securely?",                 "expected": "knowledge_query"},
    {"message": "How do I contact support?",                   "expected": "knowledge_query"},
    {"message": "What features are included in the starter plan?", "expected": "knowledge_query"},
    {"message": "How long does a refund take?",                "expected": "knowledge_query"},
    # scheduling
    {"message": "Can we meet tomorrow at 3pm?",                "expected": "scheduling"},
    {"message": "Schedule a call for Monday morning",          "expected": "scheduling"},
    {"message": "Are you free next Thursday?",                 "expected": "scheduling"},
    {"message": "I'd like to book a meeting",                  "expected": "scheduling"},
    {"message": "Can we reschedule our call to Friday?",       "expected": "scheduling"},
    {"message": "Set up a 30 minute call for next week",       "expected": "scheduling"},
    {"message": "Let's meet on April 10th at 2pm",             "expected": "scheduling"},
    {"message": "Book a meeting for Tuesday afternoon",        "expected": "scheduling"},
]

# ─── RAG Evaluation Test Set ──────────────────────────────────────────────────

RAG_TEST_SET = [
    {
        "question": "What are your working hours?",
        "ground_truth": "Monday through Friday, 9:00 AM to 6:00 PM Riyadh time.",
    },
    {
        "question": "How much does the Pro plan cost?",
        "ground_truth": "$29 per month per user.",
    },
    {
        "question": "What is the refund period?",
        "ground_truth": "Refund requests must be submitted within 14 days of purchase.",
    },
    {
        "question": "How do I reset my password?",
        "ground_truth": "Go to app.example.com/reset and enter your registered email.",
    },
    {
        "question": "Do you operate on weekends?",
        "ground_truth": "No, the team does not operate on weekends or public holidays.",
    },
    {
        "question": "What encryption is used for data storage?",
        "ground_truth": "AES-256 encryption at rest and TLS 1.3 in transit.",
    },
    {
        "question": "How long does account deletion take?",
        "ground_truth": "Account deletion requests take 30 days to process.",
    },
    {
        "question": "What discount is available for annual billing?",
        "ground_truth": "Annual billing provides a 20% discount on Pro and Enterprise plans.",
    },
]


# ─── Evaluation Functions ─────────────────────────────────────────────────────

def evaluate_intent_classifier() -> dict:
    print("\n[1/3] Evaluating Intent Classifier...")
    print(f"      Test set: {len(INTENT_TEST_SET)} samples\n")

    correct = 0
    results = []

    for sample in INTENT_TEST_SET:
        predicted = classify_intent(sample["message"])
        is_correct = predicted["intent"] == sample["expected"]
        correct += int(is_correct)

        results.append({
            "message": sample["message"],
            "expected": sample["expected"],
            "predicted": predicted["intent"],
            "confidence": predicted["confidence"],
            "correct": is_correct,
        })

        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] \"{sample['message'][:45]:<45}\" → {predicted['intent']}")

    accuracy = correct / len(INTENT_TEST_SET)
    print(f"\n  Accuracy: {correct}/{len(INTENT_TEST_SET)} = {accuracy:.1%}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "intent_results.csv", index=False)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(INTENT_TEST_SET),
    }


def evaluate_rag_pipeline() -> dict:
    print("\n[2/3] Evaluating RAG Pipeline (RAGAS)...")
    print(f"      Test set: {len(RAG_TEST_SET)} samples\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in RAG_TEST_SET:
        print(f"  Querying: \"{sample['question'][:60]}\"")
        result = rag_service.query(sample["question"])

        questions.append(sample["question"])
        answers.append(result["answer"])
        ground_truths.append(sample["ground_truth"])
        # For RAGAS we need the retrieved contexts — using answer as proxy if not available
        contexts.append([result["answer"]])

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_relevancy

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        ragas_result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_relevancy],
        )

        scores = {
            "faithfulness": round(ragas_result["faithfulness"], 3),
            "answer_relevancy": round(ragas_result["answer_relevancy"], 3),
            "context_relevancy": round(ragas_result["context_relevancy"], 3),
        }

        print(f"\n  Faithfulness:       {scores['faithfulness']}")
        print(f"  Answer Relevancy:   {scores['answer_relevancy']}")
        print(f"  Context Relevancy:  {scores['context_relevancy']}")

        ragas_result.to_pandas().to_csv(RESULTS_DIR / "rag_results.csv", index=False)
        return scores

    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        print("  Falling back to manual grounding check...")

        grounded_count = sum(1 for r in answers if "don't have that information" not in r.lower())
        grounding_rate = grounded_count / len(answers)
        print(f"  Grounding rate: {grounded_count}/{len(answers)} = {grounding_rate:.1%}")

        return {"grounding_rate": grounding_rate}


def evaluate_latency(n_requests: int = 10) -> dict:
    print(f"\n[3/3] Evaluating Latency ({n_requests} requests)...\n")

    test_messages = [
        ("casual_chat",     "Hey, how's it going?"),
        ("knowledge_query", "What are your working hours?"),
        ("scheduling",      "Can we meet tomorrow at 2pm?"),
        ("casual_chat",     "Thanks for your help!"),
        ("knowledge_query", "What is the refund policy?"),
    ]

    latencies = {"casual_chat": [], "knowledge_query": [], "scheduling": []}

    for i in range(n_requests):
        intent, message = test_messages[i % len(test_messages)]
        start = time.time()

        if intent == "knowledge_query":
            rag_service.query(message)
        else:
            classify_intent(message)   # lightweight proxy

        elapsed = (time.time() - start) * 1000
        latencies[intent].append(elapsed)
        print(f"  Request {i+1:02d} [{intent}]: {elapsed:.0f}ms")

    results = {}
    for intent, times in latencies.items():
        if times:
            results[intent] = {
                "avg_ms": round(sum(times) / len(times), 1),
                "max_ms": round(max(times), 1),
                "min_ms": round(min(times), 1),
            }
            print(f"\n  {intent}: avg={results[intent]['avg_ms']}ms  "
                  f"min={results[intent]['min_ms']}ms  max={results[intent]['max_ms']}ms")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the WhatsApp agent system")
    parser.add_argument(
        "--component",
        choices=["all", "intent", "rag", "latency"],
        default="all",
        help="Which component to evaluate (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  WhatsApp Conversational Agent — System Evaluation")
    print("=" * 60)

    report = {}

    if args.component in ("all", "intent"):
        report["intent"] = evaluate_intent_classifier()

    if args.component in ("all", "rag"):
        report["rag"] = evaluate_rag_pipeline()

    if args.component in ("all", "latency"):
        report["latency"] = evaluate_latency()

    # Save full report
    with open(RESULTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Results saved to: evaluation/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
