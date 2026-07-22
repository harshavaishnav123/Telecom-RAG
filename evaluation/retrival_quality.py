from sentence_transformers import SentenceTransformer,CrossEncoder
from qdrant_client import QdrantClient
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import pandas as pd
import torch
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
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
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

test_queries = [
    (
        "What is SRB1?",
        "38.331"
    ),
    (
        "What is NGAP?",
        "38.413"
    ),
    (
        "What is AMF?",
        "23.501"
    ),
    (
        "What is E2AP?",
        "E2AP"
    ),
    (
    "What is Access and Mobility Management Function?",
    "23.501"
)
]
for query,expected in test_queries:
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
    # for hit, score in zip(unique_results, scores ):
    
    #   reranked.append((score, hit))
    ## it penalizes the dictionary and vocabulary 
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
 
    for rank,(score,hit) in enumerate(top5, start=1):
        payload = hit.payload
        print(f"\nRank #{rank}")
        print(
            "Score:",
            round(score, 4)
        )
        print(
            "Spec:",
            payload.get("spec")
        )
        print(
            "Release:",
            payload.get("release_file")
        )
        print(
            "Section:",
            payload.get("section")
        )
        print(
            "Title:",
            payload.get("title")
        )
        print("-" * 80)
        print(
            payload.get(
                "text",
                ""
            )[:1000]
        )
        print("=" * 100)
        # retrieval code here
        print()
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
    specs = [
        hit.payload["spec"]
        for score,hit in top5]
    if expected in specs:
        correct += 1
    for rank, (score, hit) in enumerate(top5, start=1):

      print(
        f"Rank {rank} | "
        f"Qdrant={hit.score:.4f} | "
        f"Rerank={score:.4f}"
    )
    query_tokens = query.lower().split()

    scoresbm = bm25.get_scores(
    query_tokens
)
   
print(
    "Recall@5:",
    correct / len(test_queries)
     )
    