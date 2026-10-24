"""Compare baseline cosine ranking with local cross-encoder reranking.

Run from project root:
    uv add sentence-transformers
    uv run python evaluation/compare_reranking.py

Does not call a generative LLM or modify the existing Chroma collection.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from app import retrieve
from config import CONFIG
from evaluate_retrieval import relevant

RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
EVALUATION_K = 4


def evaluate_ranks(items, gold):
    rank = next((i for i, item in enumerate(items[:EVALUATION_K], 1)
                 if relevant(item['metadata'], gold)), None)
    return rank, int(rank is not None), 1.0 / rank if rank else 0.0


def summarize(rows, method):
    n = len(rows)
    return {
        'method': method,
        'scored_questions': n,
        'hit_rate_at_4': sum(r['hit_at_4'] for r in rows) / n,
        'mrr_at_4': sum(r['reciprocal_rank'] for r in rows) / n,
        'average_latency_seconds': sum(r['latency_seconds'] for r in rows) / n,
        'results': rows,
    }


def main():
    if CONFIG['retrieval_mode'] != 'similarity' or CONFIG['top_k'] != 12:
        raise ValueError('Expected Week 2 baseline: similarity retrieval with top_k=12')
    if CONFIG['similarity_threshold'] != 0.35:
        raise ValueError('Expected Week 2 baseline: similarity_threshold=0.35')

    tests = json.loads((Path(__file__).parent / 'test_questions.json').read_text(encoding='utf-8'))
    tests = [t for t in tests if t['gold_sections']]
    embedding_model = SentenceTransformer(CONFIG['embedding_model'])
    collection = chromadb.PersistentClient(path=str(ROOT / CONFIG['chroma_dir'])).get_collection(CONFIG['collection_name'])
    print(f'Loading reranker: {RERANKER_MODEL}')
    reranker = CrossEncoder(RERANKER_MODEL)

    baseline_rows, reranked_rows = [], []
    for t in tests:
        start = time.perf_counter()
        candidates = retrieve(t['question'], collection, embedding_model)
        baseline_latency = time.perf_counter() - start

        start_rerank = time.perf_counter()
        scores = reranker.predict([(t['question'], c['text']) for c in candidates], show_progress_bar=False) if candidates else []
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        ranked = [candidates[i] for i in order]
        rerank_latency = time.perf_counter() - start_rerank

        for rows, items, latency in (
            (baseline_rows, candidates, baseline_latency),
            (reranked_rows, ranked, baseline_latency + rerank_latency),
        ):
            rank, hit, rr = evaluate_ranks(items, t['gold_sections'])
            rows.append({
                'id': t['id'], 'category': t['category'],
                'first_relevant_rank': rank, 'hit_at_4': hit,
                'reciprocal_rank': rr,
                'candidate_pool_has_gold': any(relevant(c['metadata'], t['gold_sections']) for c in candidates),
                'candidate_count': len(candidates),
                'latency_seconds': round(latency, 4),
                'top_4': [{'source': c['metadata']['source'], 'heading': c['metadata']['heading'],
                           'similarity': round(c['similarity'], 4)} for c in items[:4]],
            })
        print(f"{t['id']} | baseline hit={baseline_rows[-1]['hit_at_4']} rank={baseline_rows[-1]['first_relevant_rank']}"
              f" | rerank hit={reranked_rows[-1]['hit_at_4']} rank={reranked_rows[-1]['first_relevant_rank']}"
              f" | pool_gold={reranked_rows[-1]['candidate_pool_has_gold']} | rerank={rerank_latency:.3f}s")

    base = summarize(baseline_rows, 'similarity_baseline')
    rerank = summarize(reranked_rows, 'cross_encoder_reranking')
    common_config = {
        'embedding_model': CONFIG['embedding_model'], 'top_k': CONFIG['top_k'],
        'similarity_threshold': CONFIG['similarity_threshold'],
        'reranker_model': RERANKER_MODEL, 'evaluation_k': EVALUATION_K,
    }
    output = {'config': common_config, 'baseline': base, 'reranked': rerank,
              'candidate_recall': sum(r['candidate_pool_has_gold'] for r in reranked_rows)/len(reranked_rows)}
    path = Path(__file__).parent / 'reranking_comparison.json'
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('\n                 Baseline       Reranked')
    print(f"Hit Rate@4        {base['hit_rate_at_4']:.3f}          {rerank['hit_rate_at_4']:.3f}")
    print(f"MRR@4             {base['mrr_at_4']:.3f}          {rerank['mrr_at_4']:.3f}")
    print(f"Avg latency (s)   {base['average_latency_seconds']:.3f}          {rerank['average_latency_seconds']:.3f}")
    print(f"Gold in top-12 pool: {output['candidate_recall']:.3f}")
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
