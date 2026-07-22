
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
INPUT_TELESQNA = "/home/shreya/Harsha/data/TeleQnA.txt"
OUTPUT_FILE = "/home/shreya/Harsha/data/error_analysis_qa.jsonl"

TOP_K = 30
CONTEXT_TOKEN_BUDGET = 6000
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def retrieve_dense(query, embed_model, client, collection_name, top_k=30):
    """Retrieves top_k candidates using dense vector search from Qdrant."""
    query_vector = embed_model.encode(query, normalize_embeddings=True)
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=top_k
    ).points
    return dense_results

def retrieve_bm25(query, bm25, top_k=30):
    """Retrieves top_k matching document indices and scores using BM25."""
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]
    return bm25_top_idx, bm25_scores


def rrf_fusion(dense_results, bm25_top_idx, bm25_scores, df, top_n=50, k=60):
    """Combines dense and sparse results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}
    doc_store = {}
    # for weighted rrf
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

    # process BM25 Ranks
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

    # Sort candidates based on fusion score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_store[key] for key in sorted_keys[:top_n]]


def rerank(query, candidates, reranker, top_n=8):
    """Reranks candidates using Cross-Encoder with custom domain-penalties."""
    pairs = [[query, item["payload"]["text"]] for item in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for item, score in zip(candidates, scores):
        source = item["payload"].get("source", "")
        # Apply strict domain logic context adjustments
        if source in ["dictionary", "vocabulary"]:
            score *= 0.8
        reranked.append((score, item))

    reranked.sort(reverse=True, key=lambda x: x[0])
    return reranked[:top_n]


def build_context(top_chunks, tokenizer, max_tokens=6000):
    """Assembles text context while respecting explicit token budgets."""
    context = ""
    citations=[]
    current_tokens=0
    for score, item in top_chunks:
        payload = item["payload"]
        
        #  Extract explicit citation parameters
        chunk_str = (
            f"SPEC: {payload.get('spec', 'N/A')}\n"
            f"SECTION: {payload.get('section', 'N/A')}\n"
            f"TITLE: {payload.get('title', 'N/A')}\n\n"
            f"{payload.get('text', '')}\n"
            f"{'='*50}\n"
        )
        chunk_tokens=len(tokenizer.encode(chunk_str,add_special_tokens=False))
        #  Context Budget Validation
        if current_tokens+ chunk_tokens > max_tokens:
            break
        context += chunk_str
        current_tokens+=chunk_tokens
        citations.append(
            {
                "spec": payload.get("spec", ""),
                "section": payload.get("section", ""),
                "title": payload.get("title", "")
            }
        )
    return context,citations


def generate_answer(query, options, context, citations, tokenizer, llm):
    """Generates the localized answer utilizing strict prompt constraints."""
    messages = [
    {
        "role": "system",
        "content": """You are an expert in 3GPP and O-RAN standards. 
Your task is to identify the correct multiple-choice option based on the provided context.

Guidelines:
1. Analyze the context to find which option is logically supported by the facts.
2. Output the result strictly in this format: option <number>: <text> (with no extra explanation or conversational intro).
 """
    },
    {
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{query}

Options:
option 1: {options.get("option 1","")}
option 2: {options.get("option 2","")}
option 3: {options.get("option 3","")}
option 4: {options.get("option 4","")}
option 5: {options.get("option 5","")}

Identify the correct option based on the provided content,also you can use your knowledge if it is relevant and your are sure about it. Format your output exactly like: option <number>: <text> in text setion there should be only answer(i.e text comes from correct options)"""
    }
]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=128, do_sample=False)
        
    generated = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    
    # Check if answer contains a valid option pattern, else map sources
    answer += "\n\nSources:\n"
    for c in citations:
        answer += f"- TS {c['spec']} | Section {c['section']} | {c['title']}\n"
    
    return answer


def normalize_answer(text):
    """ Standardizes generated option patterns for strict checking."""
    if not text:
        return ""
    text = text.lower().strip()
    match = re.search(r'option\s*([1-5])', text)
    if match:
        return f"option {match.group(1)}"
    match_digit = re.search(r'\b([1-5])\b', text)
    if match_digit:
        return f"option {match_digit.group(1)}"
    return text


def evaluate(eval_df, df, bm25, embed_model, client, reranker, tokenizer, llm):
    """Orchestrates testing loop execution, metric tracking, and results parsing."""
    correct = 0
    evaluation_results = []

    for idx, row in eval_df.iterrows():
        query = row["question"]
        expected = row["answer"]

        print("=" * 100)
        print("QUERY:", query)
        print("=" * 100)

        # Execution Chain
        dense_res = retrieve_dense(query, embed_model, client, COLLECTION_NAME, TOP_K)
        bm25_idx, bm25_scores = retrieve_bm25(query, bm25, TOP_K)
        candidates = rrf_fusion(dense_res, bm25_idx, bm25_scores, df, top_n=50)
        top8 = rerank(query, candidates, reranker, top_n=5)
        context,citations = build_context(top8, tokenizer, max_tokens=CONTEXT_TOKEN_BUDGET)
        
        options = {f"option {i}": row.get(f"option {i}", "") for i in range(1, 6)}
        answer = generate_answer(query, options, context,citations, tokenizer, llm)
        
        print("predicted answer:", answer)
        print("expected answer:", expected)

        # Metrics Tracking & Normalization Validation
        norm_pred = normalize_answer(answer)
        norm_exp = normalize_answer(expected)
        is_correct = (norm_pred == norm_exp)

        if is_correct:
            correct += 1

        specs8 = [item["payload"].get("spec", "") for _, item in top8]
    
        evaluation_results.append({
            "question": query,
            "ground_truth": expected,
            "prediction": answer,
            "normalized_ground_truth": norm_exp,
            "normalized_prediction": norm_pred,
            "retrieved_correct_chunk": is_correct,
            "retrieved_specs": specs8,
            "context_used": context
        })

    print(f"\nNumber of correct results: {correct}")
    print(f"Accuracy: {correct / len(eval_df):.4f}")
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

    # TeleQnA Dataset Parsing
    with open(INPUT_TELESQNA, "r", encoding="utf-8") as f:
        teleqna_data = json.load(f)
        
    rows = []
    for key, val in teleqna_data.items():
        rows.append({
            "id": key,
            "question": val.get("question", ""),
            **{f"option {i}": val.get(f"option {i}", "") for i in range(1, 6)},
            "answer": val.get("answer", ""),
            "explanation": val.get("explanation", ""),
            "category": val.get("category", "")
        })
        
    teleqna = pd.DataFrame(rows)
    eval_df = teleqna.sample(n=min(100, len(teleqna)), random_state=42)

    # Launch Evaluation Loop
    results = evaluate(eval_df, df, bm25, embed_model, client, reranker, tokenizer, llm)

    #Save Error-Analysis Log File
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} validation run records to {OUTPUT_FILE}")

