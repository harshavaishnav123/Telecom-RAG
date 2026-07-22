import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# initializing the qdrant path and the batch size
QDRANT_PATH = "./qdrant_db"
BATCH_SIZE = 64

# defining the source chunk files and matching them with their target collections
DATA_PIPELINES = [
    {
        "input_file": r"..../data_plane_chunks.jsonl",
        "collection_name": "data_plane",
        "group_fallback": "Data Plane"
    },
    {
        "input_file": r"..../edge_chunks.jsonl",
        "collection_name": "multi_access_edge",
        "group_fallback": "Multi-access Edge"
    }, 
    {
        "input_file": r"..../l_2_3_chunks.jsonl",
        "collection_name": "l2-l3-protocol-logic",
        "group_fallback": "L2\/L3 Protocol Logic"
    },
     {
        "input_file": r"..../mac_layer_chunks.jsonl",
        "collection_name": "physical_mac_layer",
        "group_fallback": "Physical\/MAC Layer"
    },
      {
        "input_file": r"..../o-kpi_chunks.jsonl",
        "collection_name": "o_ran_kpi",
        "group_fallback": "O-RAN KPI"
    },
       {
        "input_file": r"..../o-query_chunks.jsonl",
        "collection_name": "o_ran_management",
        "group_fallback": "O-RAN MANAGEMENT"
    },
       {
        "input_file": r"..../ran_arch_chunks.jsonl",
        "collection_name": "ran_architecture",
        "group_fallback": "RAN ARCHITECTURE"
    }
]

# laoding the embedding model
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5",device="cuda")
VECTOR_SIZE = model.get_sentence_embedding_dimension()
print("Embedding dimension:", VECTOR_SIZE)

# initializing the qdrantclient path
client = QdrantClient(path=QDRANT_PATH)

# running the pipeline for each dataset
for pipeline in DATA_PIPELINES:
    input_path = Path(pipeline["input_file"])
    col_name = pipeline["collection_name"]
    group_fallback = pipeline["group_fallback"]
    
    print("\n" + "="*60)
    print(f"PROCESSING: {input_path.name} -> COLLECTION: {col_name}")
    print("="*60)
    
    if not input_path.exists():
        print(f"Skipping: Chunk file not found at {input_path}")
        continue

    # handling collection initialization
    existing = [c.name for c in client.get_collections().collections]
    if col_name in existing:
        print(f"Resetting existing collection: {col_name}")
        client.delete_collection(col_name)

    client.create_collection(collection_name=col_name,vectors_config=VectorParams(size=VECTOR_SIZE,distance=Distance.COSINE))

    # reading the data chunks
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {input_path.name}")

    # embedding loop and vector db
    point_id = 0

    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Indexing {col_name}"):
        batch = chunks[start : start + BATCH_SIZE]

        # filtering the valid batch
        valid_batch = [row for row in batch if "text" in row and isinstance(row["text"], str) and row["text"].strip()]
        
        if not valid_batch:
            continue

        texts = [row["text"] for row in valid_batch]

        # creating the dense embedding space matrix
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        points = []

        for row, embedding in zip(valid_batch, embeddings):
            payload = {
                "chunk_id": row.get("chunk_id", ""),
                "source": row.get("source", ""),
                "group": row.get("group", group_fallback),
                "release_file": row.get("release_file", ""),
                "spec": row.get("spec", ""),
                "section": row.get("section", ""),
                "title": row.get("title", ""),
                "text": row["text"]
            }

            points.append(PointStruct(id=point_id,vector=embedding.tolist(),payload=payload))
            point_id += 1

        client.upsert(collection_name=col_name,points=points)

    print(f"Finished indexing target collection: {col_name}")

print("\nAll targeted datasets have been indexed successfully into distinct collections!")