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

# config
INPUT_TELESQNA = "/home/shreya/Harsha/data/TeleQnA.txt"
OUTPUT_FILE = "/home/shreya/Harsha/data/error_analysis_qa2.jsonl"
QDRANT_PATH = "./qdrant_db"
BM25_DIR = "./bm25_indices"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TOP_K_PER_COLLECTION = 30  
CONTEXT_TOKEN_BUDGET = 8000
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()

# directory initialization for BM25 caches,as we have created separate collections for each domain,to not mix data

os.makedirs(BM25_DIR, exist_ok=True)

COLLECTION_REGISTRY = {
    "data_plane": {
        "chunks_file": "/home/shreya/Harsha/data/data_plane_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "data_plane_bm25.pkl")
    },
    "multi_access_edge": {
        "chunks_file": "/home/shreya/Harsha/data/edge_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "edge_bm25.pkl")
    },
    "l2-l3-protocol-logic": {
        "chunks_file": "/home/shreya/Harsha/data/l_2_3_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "l2_l3_bm25.pkl")
    },
    "physical_mac_layer": {
        "chunks_file": "/home/shreya/Harsha/data/mac_layer_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "mac_layer_bm25.pkl")
    },
    "o_ran_kpi": {
        "chunks_file": "/home/shreya/Harsha/data/o-kpi_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "o_ran_kpi_bm25.pkl")
    },
    "o_ran_management": {
        "chunks_file": "/home/shreya/Harsha/data/o-query_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "o_ran_management_bm25.pkl")
    },
    "ran_architecture": {
        "chunks_file": "/home/shreya/Harsha/data/ran_arch_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "ran_architecture_bm25.pkl")
    }
}

LOADED_RESOURCES = {}
## As based on required collections(dynamic collections based on query) we need to pick the chunk_data for dense and bm25 

def get_collection_resources(collection_name):
    """Loads or instantiates dataframes and BM25 objects on demand (Cached in RAM)."""
    if collection_name in LOADED_RESOURCES:
        return LOADED_RESOURCES[collection_name]
    
    if collection_name not in COLLECTION_REGISTRY:
        return None, None
        
    config = COLLECTION_REGISTRY[collection_name]
    df = pd.read_json(config["chunks_file"], lines=True)
    documents = df["text"].fillna("").astype(str).tolist()
    ## load bm25 if created early else create and use them
    
    if os.path.exists(config["bm25_file"]):
        with open(config["bm25_file"], "rb") as f:
            bm25 = pickle.load(f)
    else:
        tokenized = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)
        with open(config["bm25_file"], "wb") as f:
            pickle.dump(bm25, f)
            
    LOADED_RESOURCES[collection_name] = (df, bm25)
    return df, bm25

# ==========================================================
# MULTI-COLLECTION FUSION LAYER
# ==========================================================
def multi_collection_rrf(all_dense_results, all_bm25_results, k=60):
    #     Reciprocal Rank Fusion (rrf) across multiple independent collections.
    #      all_dense_results: Dict of {collection_name: [qdrant_points]}
    #      all_bm25_results: Dict of {collection_name: (top_indices, scores, dataframe)}
    
    rrf_scores = {}
    doc_store = {}
    ## these are used for penalizing or awarding
    dense_weight = 1.2
    bm25_weight = 0.8

    # get all dense results from all active collections for our query,so that we can use for retrival.
    for col_name, dense_res in all_dense_results.items():
        for rank, hit in enumerate(dense_res):
            payload = hit.payload
            key = payload.get("text", "")
            if not key: continue

             # combine collection + text,to skip duplicates.
            lookup_key = f"{col_name}_{key}"
            
            if lookup_key not in rrf_scores:
                rrf_scores[lookup_key] = 0.0
                doc_store[lookup_key] = {"payload": payload, "dense_score": hit.score, "bm25_score": 0.0}
            
            rrf_scores[lookup_key] += dense_weight / (k + (rank + 1))

    #get all bm25 results from all active collections for our query,so that we can use for retrival.
    for col_name, (bm25_idx, bm25_scores, df) in all_bm25_results.items():
        for rank, idx in enumerate(bm25_idx):
            if idx >= len(df): continue
            row_df = df.iloc[idx]
            payload = row_df.to_dict()
            key = payload.get("text", "")
            if not key: continue
            # combine collection + text,to skip duplicates.
            lookup_key = f"{col_name}_{key}"

            if lookup_key not in rrf_scores:
                rrf_scores[lookup_key] = 0.0
                doc_store[lookup_key] = {"payload": payload, "dense_score": 0.0, "bm25_score": bm25_scores[idx]}
            else:
                doc_store[lookup_key]["bm25_score"] = bm25_scores[idx]
            
            rrf_scores[lookup_key] += bm25_weight / (k + (rank + 1))

    # sort candidates across all collections globally,so that we can take top results 
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_store[key] for key in sorted_keys]


def rerank(query, candidates, reranker, top_n=8):
    # rerank candidates using Cross-Encoder with domian -penalties ,we will penalize for dictionary / vocabulary so that , for definations and vocabulary giving lower priority is better.
    pairs = [[query, item["payload"]["text"]] for item in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for item, score in zip(candidates, scores):
        source = item["payload"].get("source", "")
       ## we will penalize the vocabulary anmd dictionary domains chunks if the score is <0 we will penalize more
        if source in ["dictionary", "vocabulary"]:
             if score > 0:
                 score = score * 0.8
             else:
                 score=score * 1.2
        reranked.append((score, item))

    reranked.sort(reverse=True, key=lambda x: x[0])
    return reranked[:top_n]

## we will build contect and citations from retrieved chunks
def build_context(top_chunks, tokenizer, max_tokens=6000):
    """Assembles text context while respecting explicit token budgets."""
    context = ""
    citations=[]
    current_tokens=0
    for score, item in top_chunks:
        payload = item["payload"]
        
         # extract explicit citation parameters
        chunk_str = (
            f"SPEC: {payload.get('spec', 'N/A')}\n"
            f"SECTION: {payload.get('section', 'N/A')}\n"
            f"TITLE: {payload.get('title', 'N/A')}\n\n"
            f"{payload.get('text', '')}\n"
            f"{'='*50}\n"
        )
        chunk_tokens=len(tokenizer.encode(chunk_str,add_special_tokens=False))
        # check context budget ,because this context needs to go into llm.
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
        "content": f"""  Context: {context}
                          Question: {query}

        Options:
        option 1: {options.get("option 1","")}
        option 2: {options.get("option 2","")}
        option 3: {options.get("option 3","")}
        option 4: {options.get("option 4","")}
        option 5: {options.get("option 5","")}
        
        Identify the correct option based on the provided content,also you can use your knowledge if it is relevant and your are sure about it. Format your output exactly like: option <number>: <text> in text setion there should be only answer(i.e text comes from correct options)"""
    }
]
    # need to convert to chat template as we are asking our llm.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(llm.device)
## for inference 
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=128, do_sample=False)
        
    generated = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    
   # after genaration of output from LLM,we will add citations of data from where we had picked them.
    answer += "\n\nSources:\n"
    for c in citations:
        answer += f"- TS {c['spec']} | Section {c['section']} | {c['title']}\n"
    
    return answer

## it gives out only option ,which is easy to compare
def normalize_answer(text):
    """Step 5: Standardizes generated option patterns for strict checking."""
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
## As we will decide dynamically based on query to take which collections for 
def route_query_with_llm(query,tokenizer,llm,sw):
    collection_definitions = """
    - "data_plane": General vocabularies, dictionaries, and broad network data data analytics frameworks. Holds: 21.905, 23.288.
    - "multi_access_edge": Multi-access Edge Computing (MEC) applications, frameworks, hosting platforms, and edge radio indicators. Holds: MEC003, MEC011, MEC012.
    - "l2-l3-protocol-logic": 3GPP layer 2 and layer 3 control protocols, UE idle/active states, scheduling reports, and mobility configurations. Holds: 38.300, 38.304, 38.306, 38.321, 38.322, 38.323, 38.331.
    - "physical_mac_layer": 3GPP lower physical layers and control loop procedures, including slot mapping, channel bit modulation, beamforming control, and channel quality matrices. Holds: 38.211, 38.212, 38.213, 38.214.
    - "o_ran_kpi": O-RAN telemetry reporting models, performance management, and open fronthaul data stream tracking counters. Holds: O-RAN E2SM-RC, O-RAN Fronthaul CUS-Plane.
    - "o_ran_management": O-RAN high-level platform controllers, interface messaging configurations, automation workflows, and architectural use cases. Holds: A1TD, E2AP, Non-RT-RIC-ARCH, Near-RT-RIC-ARCH (RICARCH), SMO-ARCH, A1 Use Case Requirements.
    - "ran_architecture": Multi-node 3GPP network splits, backhaul routing controls, and legacy 4G anchor leg management. Holds: 38.401, 38.410, 38.413, 36.300.
    """

    available_collections = [
        "data_plane", "multi_access_edge", "l2-l3-protocol-logic", 
        "physical_mac_layer", "o_ran_kpi", "o_ran_management", "ran_architecture"
    ]
    
    system_prompt = f"""You are a specialized telecom routing classifier. Analyze the user's query and map it to the most relevant database collection(s) based on the strict specification directory below.

    Specification Map: {collection_definitions}

    Rules:
    1. Choose ONLY from this explicit list: {available_collections}
    2. You can output more than one collection separated by a comma if the query bridges multiple specification domains (e.g., lower layer control loops intersecting with MAC scheduling).
    3. Output ONLY the plain comma-separated text list of collection names. Do not include markdown formatting, brackets, or backticks.
    
    Example: "Explain F1-C interface routing drops during context setup" -> ran_architecture"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}"}
    ]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(text,return_tensors="pt").to(llm.device)
    with torch.no_grad():
        outputs=llm.generate(**inputs,max_new_tokens=32,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    
    raw_response=tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
    pred_coll=[c.strip()  for c in raw_response.split(",") if c.strip() in available_collections]
    # pred_coll=[]
    if not pred_coll:
        if sw==0:
           pred_coll=[
        "data_plane",
        "physical_mac_layer", "o_ran_kpi","ran_architecture"
    ]   
        else:
            pred_coll=[
        "data_plane",
        "physical_mac_layer", "multi_access_edge", "l2-l3-protocol-logic", 
         "o_ran_management"
    ]
    if "data_plane" not in pred_coll:
       pred_coll.append("data_plane")
    return pred_coll
    # return available_collections
    
def evaluate(eval_df, embed_model, client, reranker, tokenizer, llm):
    """Orchestrates testing loop execution, metric tracking, and results parsing."""
    correct = 0
    evaluation_results = []

    for idx, row in eval_df.iterrows():
        query = row["question"]
        expected = row["answer"]
        explanation = row["explanation"] # Fixed: Pull explanation from dataset

        target_collections = route_query_with_llm(query, tokenizer, llm, sw=1)           
        print("=" * 100)
        print(f"QUERY: {query}")
        print(f"SEARCHING ACROSS COLLECTIONS: {target_collections}")
        print("=" * 100)
        
        all_dense_results = {}
        all_bm25_results = {}
        
        for col_name in target_collections:
            df, bm25 = get_collection_resources(col_name)
            if df is None or bm25 is None:
                continue
            
            query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()
            dense_res = client.query_points(collection_name=col_name, query=query_vector, limit=TOP_K_PER_COLLECTION).points
            all_dense_results[col_name] = dense_res
            
            query_tokens = query.lower().split()
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_PER_COLLECTION]
            all_bm25_results[col_name] = (bm25_idx, bm25_scores, df)
            
        candidates = multi_collection_rrf(all_dense_results, all_bm25_results, k=60)
        reranked = rerank(query, candidates, reranker, top_n=50)
        top8 = reranked[:8]
        
        # Keep tracking raw chunks for accurate context metrics downstream
        contexts_list = [item["payload"].get("text", "") for _, item in top8]
        
        context, citations = build_context(top8, tokenizer, max_tokens=CONTEXT_TOKEN_BUDGET)
        options = {f"option {i}": row.get(f"option {i}", "") for i in range(1, 6)}
        answer = generate_answer(query, options, context, citations, tokenizer, llm)
        
        print("predicted answer:", answer)
        print("expected answer:", expected)

        norm_pred = normalize_answer(answer)
        norm_exp = normalize_answer(expected)
        is_correct = (norm_pred == norm_exp)

        if is_correct:
            correct += 1

        specs8 = [item["payload"].get("spec", "") for _, item in top8]
        
        # Fixed Schema to match downstream evaluation requirements perfectly
        evaluation_results.append({
            "question": query,
            "ground_truth": expected,
            "explanation": explanation,        # Fixed: Saved text explanation for Context Recall
            "prediction": answer,
            "normalized_ground_truth": norm_exp,
            "normalized_prediction": norm_pred,
            "exact_match_correct": 1 if is_correct else 0, # Metric optimization tracking
            "retrieved_specs": specs8,
            "contexts_list": contexts_list,    # Fixed: Storing as an explicit clean string array
            "context_used": context
        })

    print(f"\nNumber of correct results: {correct}")
    print(f"Accuracy: {correct / len(eval_df):.4f}")
    return evaluation_results


if __name__ == "__main__":
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
    client = QdrantClient(path=QDRANT_PATH)
    
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

    results = evaluate(eval_df, embed_model, client, reranker, tokenizer, llm)

    # FIXED: Write as pure JSONL lines to sync up with the Ragas evaluation parser script
    print(f"Committing validation records cleanly to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Saved {len(results)} records in JSONL layout.")

