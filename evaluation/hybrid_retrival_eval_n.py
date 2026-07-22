from sentence_transformers import SentenceTransformer,CrossEncoder
from qdrant_client import QdrantClient
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import pandas as pd
import torch

QDRANT_PATH = "/home/shreya/Harsha/qdrant_db"
COLLECTION_NAME = "telecom_rag"
TOP_K = 20
embed_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda"
)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"
)
evaluation_results=[]
client = QdrantClient(
    path=QDRANT_PATH
)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
).cuda()
correct=0

from rank_bm25 import BM25Okapi
INPUT=r"/home/shreya/Harsha/data/telecom_chunks.jsonl"
df=pd.read_json(INPUT,lines=True)
documents = df["text"].fillna("").tolist()

doc_lookup = {}

for _, row in df.iterrows():

    key = (
        row["release_file"],
        row["section"]
    )

    doc_lookup[key] = row.to_dict()

tokenized = [
    doc.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(
    tokenized
)

INPUT=r"/home/shreya/Harsha/data/evaluation.json"
test_queries=pd.read_json(INPUT)
recall1 = 0
recall5 = 0
recall10 = 0
mrr = 0.0

total_queries = len(test_queries)
for _, row in test_queries.iterrows():

    query = row["question"]

    expected = row["spec"]
    print("=" * 100)
    print("QUERY:", query)
    print("=" * 100)
    query_vector = embed_model.encode(
        query,
        normalize_embeddings=True
    )    
    query_tokens = query.lower().split()
    
    bm25_scores = bm25.get_scores(
        query_tokens
    )
    
    bm25_top_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:TOP_K]

    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=TOP_K
    ).points
    
    combined = {}
    for hit in dense_results:

       payload = hit.payload
   
       key = (
           payload.get("release_file"),
           payload.get("section")
       )
   
       combined[key] = {
           "payload": payload,
           "dense_score": hit.score,
           "bm25_score": 0.0
       }
    for idx in bm25_top_idx:

       row_df = df.iloc[idx]
   
       key = (
           row_df["release_file"],
           row_df["section"]
       )
   
       if key in combined:
   
           combined[key]["bm25_score"] = bm25_scores[idx]
   
       else:
   
           combined[key] = {
               "payload": row_df.to_dict(),
               "dense_score": 0.0,
               "bm25_score": bm25_scores[idx]
           }
    max_bm25 = max(
    x["bm25_score"]
    for x in combined.values()
)

    if max_bm25 == 0:
       max_bm25 = 1
    for doc in combined.values():

       doc["bm25_norm"] = (
        doc["bm25_score"] / max_bm25
    )
    for doc in combined.values():

       doc["hybrid_score"] = (
        0.5 * doc["dense_score"]
        +
        0.5 * doc["bm25_norm"]
    )
    docs = sorted(
    combined.values(),
    key=lambda x: x["hybrid_score"],
    reverse=True
)
    docs=docs[:50]
    pairs = []
    
    for doc in docs:
    
        text = doc["payload"]["text"]
    
        pairs.append(
            [query, text]
        )
    scores = reranker.predict(
    pairs
)
    reranked = []
    
    for doc, score in zip(docs, scores):

       source = doc["payload"].get(
    "source",
    ""
)
   
       if source in [
           "dictionary",
           "vocabulary"
       ]:
           score *= 0.8
   
       reranked.append(
           (score, doc)
       )
    reranked.sort(
    reverse=True,
    key=lambda x: x[0]
)   
    top5 = reranked[:5]
    top1 = reranked[:1]
    top10 = reranked[:10]
    context = ""
    for _, doc in top5:
        payload = doc["payload"]
        context += f"""
    SPEC: {payload['spec']}
    SECTION: {payload['section']}
    TITLE: {payload['title']}
    
    {payload['text']}
    =================================================
    """
    # context=context[:500]
    messages = [
    {
        "role": "system",
        "content": "You are a telecom standards expert. Answer only from the provided context."
    },
    {
        "role": "user",
        "content": f"""
    Context:
    
    {context}
    
    Question:
    {query}
    """
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(llm.device)
    
    print("Input tokens:", inputs.input_ids.shape[1])
    print("=" * 80)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False)
    generated = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(
    generated,
    skip_special_tokens=True)
    print(answer)
    specs1 = [
    doc["payload"]["spec"]
    for score,doc in top1
    ]
    
    specs5 = [
    doc["payload"]["spec"]
    for score,doc in top5
    ]
    
    specs10 = [
    doc["payload"]["spec"]
    for score,doc in top10
    ]
    if expected in specs1:
        recall1 += 1
    if expected in specs5:
        recall5 += 1
    if expected in specs10:
        recall10 += 1
    rr = 0

    for rank, (score, doc) in enumerate(reranked, start=1):
        payload=doc["payload"]
        if payload["spec"] == expected:
    
            rr = 1.0 / rank
    
            break
    
    mrr += rr
    print(
    f"RR={rr:.4f}"
    )
    evaluation_results.append(
    {
        "question": query,

        "ground_truth": expected,

        "prediction": answer,

        "retrieved_correct_chunk":
            expected in specs5,

        "retrieved_specs":
            specs5,

        "top1_spec":
            specs1[0]
            if len(specs1) > 0
            else None,

        "rr":
            rr
    }
)
print("\n")
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(
    "Recall@1:",
    round(recall1 / total_queries, 4)
)

print(
    "Recall@5:",
    round(recall5 / total_queries, 4)
)

print(
    "Recall@10:",
    round(recall10 / total_queries, 4)
)

print(
    "MRR:",
    round(mrr / total_queries, 4)
)
import json

OUTPUT_FILE = (
    "/home/shreya/Harsha/data"
    "error_analysis_n.json"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_results,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Saved {len(evaluation_results)} "
    f"records to {OUTPUT_FILE}"
)