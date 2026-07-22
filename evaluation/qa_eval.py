from sentence_transformers import SentenceTransformer,CrossEncoder
from qdrant_client import QdrantClient
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import pandas as pd
import torch
import json


# from this :
#  Query
#  ↓
# Embedding
#  ↓
# Qdrant
#  ↓
# Top-K

# To:
#  Query
#  ↓
# Dense Retrieval (Qdrant)
#  ↓
# Reranker
#  ↓
# Top 5

QDRANT_PATH = "/home/shreya/Harsha/qdrant_db"
COLLECTION_NAME = "telecom_rag"
TOP_K = 20
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


# test_queries=[
#     "During an inter-gNB handover, how are security keys updated and which RRC messages are exchanged between the UE and network?",
#     "How does a Near-RT RIC use E2AP procedures to control an E2 Node?",
#     "What is the difference between SRB1, SRB2 and SRB3?",
#     "How does a UE move from idle state to exchanging user data with the network?",
#     "How can a Near-RT RIC influence handover decisions in a 5G network?",
    
# ]
from rank_bm25 import BM25Okapi
INPUT=r"/home/shreya/Harsha/data/telecom_chunks.jsonl"
df=pd.read_json(INPUT,lines=True)

documents = df["text"].tolist()

tokenized = [
    doc.lower().split()
    for doc in documents
]

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

import re

def get_option_number(text):

    match = re.search(
        r"option\s*(\d+)",
        text.lower()
    )

    if match:
        return match.group(1)

    return None

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
   
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=TOP_K
    ).points
   
    seen = set()
    unique_results = []
    for hit in results:
        payload = hit.payload
        key = (
            payload.get("spec"),
            payload.get("section")
        )
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(hit)
    pairs = []
    for hit in unique_results:
      text = hit.payload["text"]
      pairs.append(
        [query, text]
    )
    scores = reranker.predict(
    pairs
)
    reranked = []

    for hit, score in zip(unique_results, scores):

       source = hit.payload.get(
           "source",
           ""
       )
   
       if source in [
           "dictionary",
           "vocabulary"
       ]:
           score *= 0.8
   
       reranked.append(
           (score, hit)
       )
    reranked.sort(
    reverse=True,
    key=lambda x: x[0]
)   
    top5 = reranked[:5]
    
    context = ""
    for _, hit in top5:
        payload = hit.payload
        context += f"""
    SPEC: {payload['spec']}
    SECTION: {payload['section']}
    TITLE: {payload['title']}
    
    {payload['text'][:500]}
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

    gt_option = get_option_number(
    answer
)

    pred_option = get_option_number(
    expected
)

    if pred_option is not None:
    
        if gt_option == pred_option:
            correct += 1
    
    else:
    
        # fallback text matching
        gt_text = re.sub(
            r"option\s*\d+\s*:",
            "",
            answer.lower()
        ).strip()
    
        if gt_text in expected.lower():
            correct += 1
    print("expected answer:",expected)
#     for rank, (score, hit) in enumerate(top5, start=1):

#       print(
#         f"Rank {rank} | "
#         f"Qdrant={hit.score:.4f} | "
#         f"Rerank={score:.4f}"
#     )
#     query_tokens = query.lower().split()

#     scoresbm = bm25.get_scores(
#     query_tokens
# )
print("Number of correct results:",correct)
print(
    "Accuracy:",
    correct / len(eval_df)
     )

    