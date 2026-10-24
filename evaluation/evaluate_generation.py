"""Week 3 generation evaluation. Run from the repository root.

    uv run python evaluation/evaluate_generation.py

Requires GEMINI_API_KEY and a Gemini model configured as JUDGE_MODEL (optional).
Uses the existing app.py pipeline and its config.py settings.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types

from app import run_question
from config import CONFIG

EVAL_IDS = ["Q01", "Q02", "Q03", "Q04", "Q07", "Q10", "Q11", "Q13", "Q05"]
OUTPUT = ROOT / "evaluation" / "generation_results.json"

JUDGE_INSTRUCTIONS = """You are an independent evaluator of a documentation-only FastAPI RAG assistant.
You receive a question, numbered retrieved excerpts, and an assistant answer.
Assess ONLY the supplied excerpts and answer, not your outside knowledge.
Score each dimension as an integer 1 (poor), 2 (partial), or 3 (good):
- context_relevance: whether retrieved excerpts provide useful support for the question.
- groundedness: whether factual claims in the answer are supported by the excerpts and cited correctly.
- answer_relevance: whether the answer addresses the question using available evidence.
For an out-of-scope question, an exact refusal with no evidence is SUCCESSFUL behavior:
score groundedness and answer_relevance 3, and context_relevance 3 if no irrelevant evidence was supplied.
For a question requiring corpus-wide counting or aggregation, do not assume a few excerpts support global claims.
If the assistant refuses an in-scope question, groundedness can be 3 (no unsupported claims), but answer_relevance must be 1.
Do not award groundedness=3 to a substantive answer with unsupported claims or incorrect citations.
Return a JSON object with integer fields context_relevance, groundedness, answer_relevance,
and a concise string rationale explaining strengths and weaknesses.
"""


def judge(client, question, answer, evidence, category):
    excerpts = "\n\n".join(
        f"[{i}] Source: {e['metadata']['source']}\n"
        f"Heading: {e['metadata']['heading']}\n"
        f"Content: {e['text']}"
        for i, e in enumerate(evidence, 1)
    ) or "(No evidence retrieved)"
    payload = (f"CATEGORY: {category}\nQUESTION: {question}\n\n"
               f"RETRIEVED EVIDENCE:\n{excerpts}\n\nASSISTANT ANSWER:\n{answer}")
    response = client.models.generate_content(
        model=CONFIG["judge_model"],
        contents=JUDGE_INSTRUCTIONS + "\n\n" + payload,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    verdict = json.loads(response.text)
    for key in ("context_relevance", "groundedness", "answer_relevance"):
        score = verdict[key]
        if type(score) is not int or score not in (1, 2, 3):
            raise ValueError(f"Invalid {key} score: {score!r}")
    return verdict


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in .env before running evaluation")
    questions = json.loads((ROOT / "evaluation" / "test_questions.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}
    collection = chromadb.PersistentClient(path=CONFIG["chroma_dir"]).get_collection(CONFIG["collection_name"])
    embedding_model = SentenceTransformer(CONFIG["embedding_model"])
    reranker = CrossEncoder(CONFIG["reranker_model"]) if CONFIG["use_reranker"] else None
    judge_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {
        "embedding_model": CONFIG["embedding_model"],
        "retrieval_mode": CONFIG["retrieval_mode"],
        "top_k": CONFIG["top_k"],
        "similarity_threshold": CONFIG["similarity_threshold"],
        "use_reranker": CONFIG["use_reranker"],
        "reranker_model": CONFIG["reranker_model"] if CONFIG["use_reranker"] else None,
        "context_k": CONFIG["context_k"],
        "generator_provider": CONFIG["llm_provider"],
        "generator_model": CONFIG.get(CONFIG["llm_provider"].lower() + "_model"),
        "judge_model": CONFIG["judge_model"],
    }, "judge_scale": "1=poor, 2=partial, 3=good", "results": []}

    for qid in EVAL_IDS:
        item = by_id[qid]
        print(f"\nEvaluating {qid}: {item['question']}", flush=True)
        try:
            start = time.perf_counter()
            result = run_question(item["question"], collection, embedding_model, reranker)
            elapsed = round(time.perf_counter() - start, 3)
            verdict = judge(judge_client, result["question"], result["answer"], result["evidence"], item["category"])
            record = {
                "id": qid, "category": item["category"],
                "question": result["question"], "answer": result["answer"],
                "refused": result["answer"].strip() == CONFIG["refusal_sentence"],
                "latency_seconds_excluding_judge": elapsed,
                "evidence": [{"citation": f"[{i}]", "source": e["metadata"]["source"],
                              "heading": e["metadata"]["heading"],
                              "source_url": e["metadata"].get("source_url"),
                              "text": e["text"], "cosine_similarity": round(e["similarity"], 4)}
                             for i, e in enumerate(result["evidence"], 1)],
                "judge": verdict,
            }
            output["results"].append(record)
            print(f"  scores: context={verdict['context_relevance']}, groundedness={verdict['groundedness']}, answer={verdict['answer_relevance']}")
        except Exception as exc:
            output["results"].append({"id": qid, "error": str(exc)})
            print(f"  ERROR: {exc}")
        finally:
            OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved full exchanges and judge verdicts to {OUTPUT}")
    print("IMPORTANT: Manually audit at least two verdicts before reporting results.")


if __name__ == "__main__":
    main()
