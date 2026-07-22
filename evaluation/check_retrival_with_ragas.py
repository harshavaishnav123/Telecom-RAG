
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
from datasets import Dataset
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import faithfulness

QDRANT_PATH = "/home/shreya/Harsha/qdrant_db"
COLLECTION_NAME = "telecom_rag"
INPUT_CHUNKS = "/home/shreya/Harsha/data/telecom_chunks.jsonl"
INPUT_EVAL = "/home/shreya/Harsha/data/evaluation.json"
OUTPUT_FILE = "/home/shreya/Harsha/data/error_analysis_retrival.jsonl"

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
    
    dense_weight = 1.2
    bm25_weight = 0.8
  
    for rank, hit in enumerate(dense_results):
        payload = hit.payload
        key = (payload.get("release_file"), payload.get("section"))
        
        if key not in rrf_scores:
            rrf_scores[key] = 0.0
            doc_store[key] = {"payload": payload, "dense_score": hit.score, "bm25_score": 0.0}
        rrf_scores[key] += dense_weight / (k + (rank + 1))

   
    for rank, idx in enumerate(bm25_top_idx):
        row_df = df.iloc[idx]
        payload = row_df.to_dict()
        key = (payload.get("release_file"), payload.get("section"))
        
        if key not in rrf_scores:
            rrf_scores[key] = 0.0
            doc_store[key] = {"payload": payload, "dense_score": 0.0, "bm25_score": bm25_scores[idx]}
        else:
            doc_store[key]["bm25_score"] = bm25_scores[idx]
        rrf_scores[key] += bm25_weight / (k + (rank + 1))

    
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
        
       
        chunk_str = (
            f"SPEC: {payload.get('spec', 'N/A')}\n"
            f"SECTION: {payload.get('section', 'N/A')}\n"
            f"TITLE: {payload.get('title', 'N/A')}\n\n"
            f"{payload.get('text', '')}\n"
            f"{'='*50}\n"
        )
        chunk_tokens=len(tokenizer.encode(chunk_str,add_special_tokens=False))
      
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


def generate_answer(query,context,citations, tokenizer, llm):
    """Generates the localized answer utilizing strict prompt constraints."""
    messages = [
        {
            "role": "system",
            "content": (
               """ You are an expert in 3GPP and O-RAN standards.
                   Answer ONLY using the supplied context.
                   If the answer is not explicitly supported by the context:
                   Do like:

                   1. Explain briefly and keep only relevant information.
                   2. Cite the relevant specifications."""
            )
        },
        {
           "role": "user",
        "content": f"""Context:{context} Question: {query}"""
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=256, do_sample=False,temperature=0.0,top_p=1.0,repetition_penalty=1.05,pad_token_id=tokenizer.eos_token_id)
        
    generated = outputs[0][inputs.input_ids.shape[1]:]
    answer=tokenizer.decode(generated, skip_special_tokens=True)
    answer += "\n\nSources:\n"
    for c in citations:
        answer += (
            f"- TS {c['spec']} | "
            f"Section {c['section']} | "
            f"{c['title']}\n"
        )
    
    return answer


def evaluate(eval_df, df, bm25, embed_model, client, reranker, tokenizer, llm):
    """Orchestrates testing loop execution, metric tracking, and results parsing."""
    recall1 = 0
    recall5 = 0
    recall8=0
    recall10 = 0
    mrr = 0.0
    evaluation_results = []
    total_queries=len(eval_df)
    for idx, row in eval_df.iterrows():
        query = row["question"]
        expected = row["expected"]

        print("=" * 100)
        print("QUERY:", query)
        print("=" * 100)

        # Execution Chain
        dense_res = retrieve_dense(query, embed_model, client, COLLECTION_NAME, TOP_K)
        bm25_idx, bm25_scores = retrieve_bm25(query, bm25, TOP_K)
        candidates = rrf_fusion(dense_res, bm25_idx, bm25_scores, df, top_n=50)
        reranked=rerank(query,candidates,reranker,top_n=len(candidates))
        top1 = reranked[:1]
        top5 = reranked[:5]
        top8 = reranked[:8]
        top10= reranked[:10]
        context,citations = build_context(top8, tokenizer, max_tokens=CONTEXT_TOKEN_BUDGET)
        
        answer = generate_answer(query, context,citations, tokenizer, llm)
        
        print("predicted answer:", answer)
        print("expected answer in spec:", expected)
        
        specs1 = [item["payload"].get("spec", "") for _, item in top1]
        specs5 = [item["payload"].get("spec", "") for _, item in top5]
        specs8 = [item["payload"].get("spec", "") for _, item in top8]
        specs10 = [item["payload"].get("spec", "") for _, item in top10]
        if expected in specs1:
           recall1 += 1
        if expected in specs5:
           recall5 += 1
        if expected in specs8:
           recall8 += 1
        if expected in specs10:
           recall10 += 1
        # Stashing Evaluation Logs
        evaluation_results.append({
            "question": query,
            "expected_spec": expected,
            "prediction": answer,
            "retrieved_specs": specs10,
            "citations":citations,
            "context_used": context
        })
    
        rr=0
        for rank, (score, doc) in enumerate(reranked, start=1):
            payload=doc["payload"]
            if payload["spec"] == expected:
                rr = 1.0 / rank
                break
        mrr += rr
        print(f"RR={rr:.4f}")
    
    print("Recall@1:",round(recall1 / total_queries, 4))
    print("Recall@5:",round(recall5 / total_queries, 4))
    print("Recall@8:",round(recall8 / total_queries, 4))
    print("Recall@10:",round(recall10 / total_queries, 4))
    print("MRR:",round(mrr / total_queries, 4))
    return evaluation_results
def calculate_ragas(evaluation_results,tokenizer,llm):
    hf_pipeline = pipeline(
        "text-generation", 
        model=llm, 
        tokenizer=tokenizer, 
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False
    )
    
    
    langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    evaluator_llm = LangchainLLMWrapper(langchain_llm)
    
    
    local_faithfulness = faithfulness
    local_faithfulness.llm = evaluator_llm
    
    
    ragas_data = {
        "question": [res["question"] for res in evaluation_results],
        "answer": [res["prediction"] for res in evaluation_results],
        "contexts": [res["contexts_list"] for res in evaluation_results]
    }
    eval_dataset = Dataset.from_dict(ragas_data)
    
 
    config = RunConfig(max_workers=1)
    
    # Execute batch local execution loop
    score_result = evaluate(
        dataset=eval_dataset, 
        metrics=[local_faithfulness],
        run_config=config
    )
    
    return score_result.to_pandas()

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

    # TeleQnA Dataset
    eval_data=pd.read_json(INPUT_EVAL)
        
    rows = []
    for _, row in eval_data.iterrows():
        rows.append({
            "question": row["question"],
             "expected":row["spec"]
        })
        
    eval_df = pd.DataFrame(rows)
 
    results = evaluate(eval_df, df, bm25, embed_model, client, reranker, tokenizer, llm)
    try:
        ragas_df=calculate_ragas(results,tokenizer,llm)
        for idx,row in ragas_df.iterrows():
            results[idx]["ragas_faithfulness"]=row["faithfulness"]
            print(f"Question number {idx}:{row["faithfulness"]}")
    except Exception as e:
        print(f"Failed to calculate local Ragas metrics: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} validation run records to {OUTPUT_FILE}")

