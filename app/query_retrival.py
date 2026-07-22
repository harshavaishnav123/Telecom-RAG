import os
import re
import json
import pickle
import torch
import pandas as pd
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, CrossEncoder

# defining the qdrant path,model and bm25 directory
QDRANT_PATH = "./qdrant_db"
BM25_DIR = "./bm25_indices"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# initializing the top k per collection and context token budget
TOP_K_PER_COLLECTION = 30 
CONTEXT_TOKEN_BUDGET = 8000

# directory initialization for bm25 caches
os.makedirs(BM25_DIR, exist_ok=True)

# data collection of all the chunk files and bm25 files
COLLECTION_REGISTRY = {
    "data_plane": {
        "chunks_file": "..../data_plane_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "data_plane_bm25.pkl")
    },
    "multi_access_edge": {
        "chunks_file": "..../edge_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "edge_bm25.pkl")
    },
    "l2-l3-protocol-logic": {
        "chunks_file": "..../l_2_3_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "l2_l3_bm25.pkl")
    },
    "physical_mac_layer": {
        "chunks_file": "..../mac_layer_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "mac_layer_bm25.pkl")
    },
    "o_ran_kpi": {
        "chunks_file": "..../o-kpi_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "o_ran_kpi_bm25.pkl")
    },
    "o_ran_management": {
        "chunks_file": "..../o-query_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "o_ran_management_bm25.pkl")
    },
    "ran_architecture": {
        "chunks_file": "..../ran_arch_chunks.jsonl",
        "bm25_file": os.path.join(BM25_DIR, "ran_architecture_bm25.pkl")
    }
}

LOADED_RESOURCES = {}

# function for getting all the collection resources
def get_collection_resources(collection_name):
    """Loads or instantiates dataframes and BM25 objects on demand (Cached in RAM)."""
    if collection_name in LOADED_RESOURCES:
        return LOADED_RESOURCES[collection_name]
    
    if collection_name not in COLLECTION_REGISTRY:
        return None, None
        
    # loading the collection name from the data files
    config = COLLECTION_REGISTRY[collection_name]
    # loading the chunks file in a dataframe
    df = pd.read_json(config["chunks_file"], lines=True)
    # filling all the missing values
    documents = df["text"].fillna("").astype(str).tolist()
    
    if os.path.exists(config["bm25_file"]):
        with open(config["bm25_file"], "rb") as f:
            # loading bm25
            bm25 = pickle.load(f)
    else:
        tokenized = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)
        with open(config["bm25_file"], "wb") as f:
            # dumping into the bm25
            pickle.dump(bm25, f)
            
    LOADED_RESOURCES[collection_name] = (df, bm25)
    return df, bm25

# function for combining dense and bm25 search results from multiple collections
def multi_collection_rrf(all_dense_results, all_bm25_results, k=60):
    """
        Executes Reciprocal Rank Fusion across multiple independent collection streams.
        all_dense_results: Dict of {collection_name: [qdrant_points]}
        all_bm25_results: Dict of {collection_name: (top_indices, scores, dataframe)}
    """
    rrf_scores = {}
    doc_store = {}
    dense_weight = 1.2
    bm25_weight = 0.8

    # fuse all dense results from all active collections
    for col_name, dense_res in all_dense_results.items():
        for rank, hit in enumerate(dense_res):
            payload = hit.payload
            key = payload.get("text", "")
            if not key: continue

            # creating the unique identifier by combining collection name and text
            lookup_key = f"{col_name}_{key}"
            
            if lookup_key not in rrf_scores:
                rrf_scores[lookup_key] = 0.0
                doc_store[lookup_key] = {"payload": payload, "dense_score": hit.score, "bm25_score": 0.0}
            
            rrf_scores[lookup_key] += dense_weight / (k + (rank + 1))

    # fusing all bm25 results from all active collections
    for col_name, (bm25_idx, bm25_scores, df) in all_bm25_results.items():
        for rank, idx in enumerate(bm25_idx):
            if idx >= len(df): continue
            row_df = df.iloc[idx]
            payload = row_df.to_dict()
            key = payload.get("text", "")
            if not key: continue

            lookup_key = f"{col_name}_{key}"

            if lookup_key not in rrf_scores:
                rrf_scores[lookup_key] = 0.0
                doc_store[lookup_key] = {"payload": payload, "dense_score": 0.0, "bm25_score": bm25_scores[idx]}
            else:
                doc_store[lookup_key]["bm25_score"] = bm25_scores[idx]
            
            rrf_scores[lookup_key] += bm25_weight / (k + (rank + 1))

    # sorting candidates across all collections
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_store[key] for key in sorted_keys]

# function for reranking the retrieved candidates using a cross-encoder reranker
def rerank(query, candidates, reranker, top_n=50):
    if not candidates: return []
    pairs = [[query, item["payload"]["text"]] for item in candidates]
    scores = reranker.predict(pairs)
    reranked = []
    for item, score in zip(candidates, scores):
        if item["payload"].get("source") in ["dictionary", "vocabulary"]:
            score = score * 0.8 if score > 0 else score * 1.2      
        reranked.append((score, item))
    reranked.sort(reverse=True, key=lambda x: x[0])
    return reranked[:top_n]

# function for building context using top_chunks
def build_context(top_chunks, tokenizer, max_tokens=6000):
    context, citations = "", []
    current_tokens = 0
    for score, item in top_chunks:
        payload = item["payload"]
        chunk_str = (
            f"SPEC: {payload.get('spec', 'N/A')} | SECTION: {payload.get('section', 'N/A')} | TITLE: {payload.get('title', 'N/A')}\n"
            f"{payload.get('text', '')}\n"
            f"----------------------------------------\n"
        )
        chunk_tokens = len(tokenizer.encode(chunk_str, add_special_tokens=False))
        if current_tokens + chunk_tokens > max_tokens:
            break
        context += chunk_str
        current_tokens += chunk_tokens
        # generating the citations
        citations.append({"source": payload.get("source"," "),
                          "release_file": payload.get("release_file",""),
                          "group": payload.get("group",""),
                          "spec": payload.get("spec", ""), 
                          "section": payload.get("section", ""),
                          "title": payload.get("title", "")})
    return context, citations

# function for generating the answer
def generate_answer(query, context, citations, tokenizer, llm):
    messages = [
        {
            "role": "system",
            "content": """You are an expert in 3GPP and O-RAN standards. 
                Your task is to answer the query based on the provided context.

                Guidelines:
                1. Analyze the context to find the answer based on logically supported by the facts.
                2. Output the result with easy understanding with facts including it ,also only for query with a good explanation .
                """
        },
        {
            "role": "user",
            "content": f"""
            Context:
            {context}

            Question:
            {query}

            Also you can use your knowledge if it is relevant and your are sure about it."""
        }
    ]
    # applying the tokenizer on the text
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        # generating using llm
        outputs = llm.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
    # decoding the answer from a tokenizer
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    answer += "\n\nSources:\n"
    seen = set()
    for c in citations:
        cit_str = f"- TS {c['spec']} | Section {c['section']} | {c['title']}"
        if cit_str not in seen:
            answer += cit_str + "\n"
            seen.add(cit_str)
    return answer

# function for routing the query with llm
def route_query_with_llm(query,tokenizer,llm,sw):
    # defining all the collection definitions
    collection_definitions = """
        - "data_plane": General vocabularies, dictionaries, and broad network data data analytics frameworks. Holds: 21.905, 23.288.
        - "multi_access_edge": Multi-access Edge Computing (MEC) applications, frameworks, hosting platforms, and edge radio indicators. Holds: MEC003, MEC011, MEC012.
        - "l2-l3-protocol-logic": 3GPP layer 2 and layer 3 control protocols, UE idle/active states, scheduling reports, and mobility configurations. Holds: 38.300, 38.304, 38.306, 38.321, 38.322, 38.323, 38.331.
        - "physical_mac_layer": 3GPP lower physical layers and control loop procedures, including slot mapping, channel bit modulation, beamforming control, and channel quality matrices. Holds: 38.211, 38.212, 38.213, 38.214.
        - "o_ran_kpi": O-RAN telemetry reporting models, performance management, and open fronthaul data stream tracking counters. Holds: O-RAN E2SM-RC, O-RAN Fronthaul CUS-Plane.
        - "o_ran_management": O-RAN high-level platform controllers, interface messaging configurations, automation workflows, and architectural use cases. Holds: A1TD, E2AP, Non-RT-RIC-ARCH, Near-RT-RIC-ARCH (RICARCH), SMO-ARCH, A1 Use Case Requirements.
        - "ran_architecture": Multi-node 3GPP network splits, backhaul routing controls, and legacy 4G anchor leg management. Holds: 38.401, 38.410, 38.413, 36.300.
    """

    # defining all the available collections
    available_collections = [
        "data_plane", "multi_access_edge", "l2-l3-protocol-logic", 
        "physical_mac_layer", "o_ran_kpi", "o_ran_management", "ran_architecture"
    ]
    
    # system prompt for generating answer
    system_prompt = f"""
            You are a specialized telecom routing classifier. Analyze the user's query and map it to the most relevant database collection(s) based on the strict specification directory below.
            Specification Map:
            {collection_definitions}

            Rules:
            1. Choose ONLY from this explicit list: {available_collections}
            2. You can output more than one collection separated by a comma if the query bridges multiple specification domains (e.g., lower layer control loops intersecting with MAC scheduling).
            3. Output ONLY the plain comma-separated text list of collection names. Do not include markdown formatting, brackets, or backticks.

            Example: "Explain F1-C interface routing drops during context setup" -> ran_architecture"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}"}
    ]
    # applying tokenizer with the text
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(text,return_tensors="pt").to(llm.device)
    with torch.no_grad():
        # generating output by llm
        outputs=llm.generate(**inputs,max_new_tokens=32,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    
    # decoding raw response by a tokenizer
    raw_response=tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
    pred_coll=[c.strip()  for c in raw_response.split(",") if c.strip() in available_collections]
    if not pred_coll:
        if sw==0:
           pred_coll=["data_plane","physical_mac_layer", "o_ran_kpi","ran_architecture"]   
        else:
            pred_coll=["data_plane","physical_mac_layer", "multi_access_edge", "l2-l3-protocol-logic", "o_ran_management"]
    if "data_plane" not in pred_coll:
       pred_coll.append("data_plane")
    return pred_coll

# function for making retrieval of a query
def make_retrival(eval_data,sw):
    print("Initializing Unified Search Pipeline Engine...")
    # loading the embedding model
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    # loading cross-encoder using reranker
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
    # loading the client using qdrant client path
    client = QdrantClient(path=QDRANT_PATH)
    # loading the llm and the tokenizer using a model
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # initializing the llm for reference
    FastLanguageModel.for_inference(llm)
    query = eval_data
    
    target_collections = route_query_with_llm(query,tokenizer,llm,sw)           
    print("=" * 100)
    print(f"QUERY: {query}")
    print(f"SEARCHING ACROSS COLLECTIONS: {target_collections}")
    print("=" * 100)
    
    # local containers to accumulate hits from multiple targets
    all_dense_results = {}
    all_bm25_results = {}
    
    # parallel loop across all targeted collection spaces
    for col_name in target_collections:
        df, bm25 = get_collection_resources(col_name)
        if df is None or bm25 is None:
            continue
        
        # fetching dense vector results
        query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()
        dense_res = client.query_points(collection_name=col_name, query=query_vector, limit=TOP_K_PER_COLLECTION).points
        all_dense_results[col_name] = dense_res
        
        # fetching sparse bm25 keyword results
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_PER_COLLECTION]
        all_bm25_results[col_name] = (bm25_idx, bm25_scores, df)
        
    # cross-collection data fusion via rrf
    candidates = multi_collection_rrf(all_dense_results, all_bm25_results, k=60)
    # deep cross-encoder filtration, context assembly and text formulation
    reranked = rerank(query, candidates, reranker, top_n=50)
    top8 = reranked[:8]
    
    context, citations = build_context(top8, tokenizer, max_tokens=CONTEXT_TOKEN_BUDGET)
    answer = generate_answer(query, context, citations, tokenizer, llm)
    # print("\nPREDICTED ANSWER:\n", answer)
    return answer