"""Evaluate the existing Week 2 similarity retriever at rank 4.

Run from the project root: uv run python evaluation/evaluate_retrieval.py
No LLM API calls are made.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb
from sentence_transformers import SentenceTransformer
from app import retrieve
from config import CONFIG


def relevant(metadata, gold_sections):
    heading = metadata.get('heading', '').casefold()
    source = metadata.get('source', '')
    return any(source == gold['source'] and gold['heading_contains'].casefold() in heading
               for gold in gold_sections)


def main():
    tests = json.loads((Path(__file__).parent / 'test_questions.json').read_text(encoding='utf-8'))
    scored = [t for t in tests if t['gold_sections']]
    if CONFIG['retrieval_mode'] != 'similarity':
        raise ValueError('Baseline requires retrieval_mode=similarity')
    if CONFIG['top_k'] < 4:
        raise ValueError('Baseline requires top_k >= 4')

    model = SentenceTransformer(CONFIG['embedding_model'])
    db = chromadb.PersistentClient(path=str(ROOT / CONFIG['chroma_dir']))
    collection = db.get_collection(CONFIG['collection_name'])
    print(f"Baseline: {len(scored)} scored questions; top_k={CONFIG['top_k']}; threshold={CONFIG['similarity_threshold']}")
    results = []
    for test in scored:
        start = time.perf_counter()
        evidence = retrieve(test['question'], collection, model)
        latency = time.perf_counter() - start
        first_rank = next((rank for rank, item in enumerate(evidence[:4], 1)
                           if relevant(item['metadata'], test['gold_sections'])), None)
        reciprocal_rank = 1 / first_rank if first_rank else 0.0
        result = {'id':test['id'],'category':test['category'],'first_relevant_rank':first_rank,
                  'hit_at_4':int(first_rank is not None),'reciprocal_rank':reciprocal_rank,
                  'latency_seconds':round(latency, 4),
                  'top_4':[{'source':item['metadata']['source'],
                            'heading':item['metadata']['heading'],
                            'similarity':round(item['similarity'], 4)} for item in evidence[:4]]}
        results.append(result)
        print(f"{test['id']} | hit@4={result['hit_at_4']} | rank={first_rank or '-'} | RR={reciprocal_rank:.3f} | {latency:.3f}s")
    summary = {'method':'similarity_baseline','scored_questions':len(scored),
               'hit_rate_at_4':sum(r['hit_at_4'] for r in results)/len(results),
               'mrr_at_4':sum(r['reciprocal_rank'] for r in results)/len(results),
               'average_latency_seconds':sum(r['latency_seconds'] for r in results)/len(results),
               'config':{'embedding_model':CONFIG['embedding_model'],'top_k':CONFIG['top_k'],
                         'similarity_threshold':CONFIG['similarity_threshold']},'results':results}
    output = Path(__file__).parent / 'baseline_results.json'
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f"\nHit Rate@4: {summary['hit_rate_at_4']:.3f} | MRR@4: {summary['mrr_at_4']:.3f}")
    print(f'Results saved: {output}')

if __name__ == '__main__':
    main()
