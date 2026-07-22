import os
import re
import json
import pickle
import torch
import pandas as pd
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder

QDRANT_PATH = "/home/shreya/Harsha/qdrant_db"
COLLECTION_NAME = "telecom_rag"
INPUT_CHUNKS = "/home/shreya/Harsha/data/telecom_chunks.jsonl"
INPUT_EVAL = "/home/shreya/Harsha/data/evaluation.json"
OUTPUT_FILE = "/home/shreya/Harsha/data/error_analysis_retrival.jsonl" # Fixed filename

TOP_K = 50
CONTEXT_TOKEN_BUDGET = 6000
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def retrieve_dense(query, embed_model, client, collection_name, top_k=30):
    query_vector = embed_model.encode(query, normalize_embeddings=True)
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=top_k
    ).points
    return dense_results


def retrieve_bm25(query, bm25, top_k=30):
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]
    return bm25_top_idx, bm25_scores


def rrf_fusion(dense_results, bm25_top_idx, bm25_scores, df, top_n=100, k=60):
    """Properly combines dense and sparse results using RRF with text-level keys."""
    rrf_scores = {}
    doc_store = {}
    dense_weight = 1.2
    bm25_weight = 0.8
    
    # Process Dense Ranks
    for rank, hit in enumerate(dense_results):
        payload = hit.payload
        key = payload.get("text", "") 
        
        if key not in rrf_scores:
            rrf_scores[key] = 0.0
            doc_store[key] = {"payload": payload, "dense_score": hit.score, "bm25_score": 0.0}
        rrf_scores[key] += dense_weight / (k + (rank + 1))

    # Process BM25 Ranks
    for rank, idx in enumerate(bm25_top_idx):
        row_df = df.iloc[idx]
        payload = row_df.to_dict()
        key = payload.get("text", "") 
        
        if key not in rrf_scores:
            rrf_scores[key] = 0.0
            doc_store[key] = {"payload": payload, "dense_score": 0.0, "bm25_score": bm25_scores[idx]}
        else:
            doc_store[key]["bm25_score"] = bm25_scores[idx]
        rrf_scores[key] += bm25_weight / (k + (rank + 1))

    # Sort candidates based on true fusion score before returning top_n
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_store[key] for key in sorted_keys[:top_n]]


def rerank(query, candidates, reranker):
    """Reranks candidates using Cross-Encoder with a robust negative-safe penalty."""
    pairs = [[query, item["payload"]["text"]] for item in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for item, score in zip(candidates, scores):
        source = item["payload"].get("source", "")
        
        
        if source in ["dictionary", "vocabulary"]:
            if score > 0:
                score *= 0.8
            else:
                score *= 1.2  # Makes negative scores even more negative 
                
        reranked.append((score, item))

    reranked.sort(reverse=True, key=lambda x: x[0])
    return reranked


def build_context(top_chunks, tokenizer, max_tokens=6000):
    context = ""
    citations = []
    current_tokens = 0
    
    for score, item in top_chunks:
        payload = item["payload"]
        chunk_str = (
            f"SPEC: {payload.get('spec', 'N/A')}\n"
            f"SECTION: {payload.get('section', 'N/A')}\n"
            f"TITLE: {payload.get('title', 'N/A')}\n\n"
            f"{payload.get('text', '')}\n"
            f"{'='*50}\n"
        )
        chunk_tokens = len(tokenizer.encode(chunk_str, add_special_tokens=False))
        if current_tokens + chunk_tokens > max_tokens:
            break
        context += chunk_str
        current_tokens += chunk_tokens
        citations.append({
            "spec": payload.get("spec", ""),
            "section": payload.get("section", ""),
            "title": payload.get("title", "")
        })
    return context, citations


def generate_answer(query, context, citations, tokenizer, llm):
    messages = [
        {
            "role": "system",
            "content": "You are an expert in 3GPP and O-RAN standards. Answer the query relying only on facts inside the context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    answer += "\n\nSources:\n"
    for c in citations:
        answer += f"- TS {c['spec']} | Section {c['section']} | {c['title']}\n"
    return answer


def evaluate(eval_df, df, bm25, embed_model, client, reranker, tokenizer, llm):
    recall1, recall5, recall8, recall10 = 0, 0, 0, 0
    mrr = 0.0
    evaluation_results = []
    total_queries = len(eval_df)

    for idx, row in eval_df.iterrows():
        query = row["question"]
        expected = str(row["expected"]).strip()

        print("=" * 100)
        print("QUERY:", query)
        print("=" * 100)

      
        dense_res = retrieve_dense(query, embed_model, client, COLLECTION_NAME, TOP_K)
        bm25_idx, bm25_scores = retrieve_bm25(query, bm25, TOP_K)
     
        candidates = rrf_fusion(dense_res, bm25_idx, bm25_scores, df, top_n=50)
    
        reranked = rerank(query, candidates, reranker)
        
        top1 = reranked[:1]
        top5 = reranked[:5]
        top8 = reranked[:8]
        top10 = reranked[:10]
        
        context, citations = build_context(top8, tokenizer, max_tokens=CONTEXT_TOKEN_BUDGET)
        answer = generate_answer(query, context, citations, tokenizer, llm)
        
        print("predicted answer:", answer)
        print("expected answer in spec:", expected)
        
        specs1 = [str(item["payload"].get("spec", "")).strip() for _, item in top1]
        specs5 = [str(item["payload"].get("spec", "")).strip() for _, item in top5]
        specs8 = [str(item["payload"].get("spec", "")).strip() for _, item in top8]
        specs10 = [str(item["payload"].get("spec", "")).strip() for _, item in top10]
        
        if expected in specs1:
            recall1 += 1
        if expected in specs5:
            recall5 += 1
        if expected in specs8:
            recall8 += 1
        if expected in specs10:
            recall10 += 1
            
        evaluation_results.append({
            "question": query,
            "expected_spec": expected,
            "prediction": answer,
            "retrieved_specs": specs10,
            "citations": citations,
            "context_used": context
        })
    
        rr = 0
        for rank, (score, doc) in enumerate(reranked, start=1):
            payload = doc["payload"]
            if str(payload.get("spec", "")).strip() == expected:
                rr = 1.0 / rank
                break
        mrr += rr
        print(f"RR={rr:.4f}")
    
    print("\n" + "#"*40 + "\nFINAL RETRIEVAL METRICS:\n" + "#"*40)
    print("Recall@1:", round(recall1 / total_queries, 4))
    print("Recall@5:", round(recall5 / total_queries, 4))
    print("Recall@8:", round(recall8 / total_queries, 4))
    print("Recall@10:", round(recall10 / total_queries, 4))
    print("MRR:", round(mrr / total_queries, 4))
    return evaluation_results

if __name__ == "__main__":
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
    client = QdrantClient(path=QDRANT_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()

    df = pd.read_json(INPUT_CHUNKS, lines=True)
    documents = df["text"].tolist()

    if not os.path.exists("bm25.pkl"):
        tokenized = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)
        with open("bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
    else:
        with open("bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)

    eval_data = pd.read_json(INPUT_EVAL)
        
    rows = []
    for _, row in eval_data.iterrows():
        rows.append({
            "question": row["question"],
            "expected": row["spec"]
        })
        
    eval_df = pd.DataFrame(rows)
    results = evaluate(eval_df, df, bm25, embed_model, client, reranker, tokenizer, llm)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} validation run records to {OUTPUT_FILE}")