from sentence_transformers import SentenceTransformer,CrossEncoder
from qdrant_client import QdrantClient
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import pandas as pd
import torch
import json
# change:
# Dense: 30
# BM25 : 30

# Hybrid Merge
# ↓
# Top 40-50 candidates
# ↓
# BGE-Reranker-v2-M3
# ↓
# Top 8 context chunks

QDRANT_PATH = "/home/shreya/Harsha/qdrant_db"
COLLECTION_NAME = "telecom_rag"
TOP_K = 30
embed_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda"
)
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cuda"
)
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

documents = df["text"].tolist()

tokenized = [
    doc.lower().split()
    for doc in documents
]
evaluation_results=[]
bm25 = BM25Okapi(
    tokenized
)
import pickle

with open(
    "bm25.pkl",
    "wb"
) as f:
    pickle.dump(bm25, f)
with open(
    "bm25.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

INPUT_FILE = "/home/shreya/Harsha/data/TeleQnA.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
rows = []

for key, value in data.items():

    rows.append({

        "id": key,
        "question": value.get("question", ""),
        "option 1": value.get("option 1", ""),
        "option 2": value.get("option 2", ""),
        "option 3": value.get("option 3", ""),
        "option 4": value.get("option 4", ""),
        "option 5": value.get("option 5", ""),
        "answer": value.get("answer", ""),

        "explanation": value.get("explanation", ""),

        "category": value.get("category", "")

    })

teleqna = pd.DataFrame(rows)
print(teleqna.columns)
sample_size = min(100, len(teleqna))

eval_df = teleqna.sample(
    n=sample_size,
    random_state=42
)
for idx,row in eval_df.iterrows():
    query=row["question"]
    expected=row["answer"]
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
    docs = list(combined.values())
    if len(docs) > 50:
       docs = docs[:50]
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

    for doc, score in zip(
        docs,
        scores
    ):
    
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
    top8 = reranked[:8]
    
    context = ""
    for _, doc in top8:
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
        "content": """You are a telecom standards expert.

Use ONLY the provided context.

If the answer is not explicitly present,
respond with:

INSUFFICIENT_CONTEXT

For MCQs:
1. Identify the correct option.
2. Output the option number and answer."""
    },
    {
        "role": "user",
        "content": f"""
Question:

{query}

Options:

option 1: {row["option 1"]}
option 2: {row["option 2"]}
option 3: {row["option 3"]}
option 4: {row["option 4"]}
option 5: {row["option 5"]}

Answer using only the provided context.
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
    print("predicted answer:",answer)

    if expected==answer:
        correct += 1
    print("expected answer:",expected)
    specs8 = [
    doc["payload"]["spec"]
    for score,doc in top8
    ]
    evaluation_results.append(
    {
        "question": query,

        "ground_truth": expected,

        "prediction": answer,

        "retrieved_correct_chunk":
            expected ==answer,

        "retrieved_specs":
            specs8
    }
)
print("Number of correct results:",correct)
print(
    "Accuracy:",
    correct / len(eval_df)
     )
    
import json

OUTPUT_FILE = (
    "/home/shreya/Harsha/data"
    "error_analysis_qa.json"
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
    