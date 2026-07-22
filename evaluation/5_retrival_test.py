from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
# process:
#     Query
#  ↓
#  Embedding
#  ↓
# Qdrant
#  ↓
# Top-K


QDRANT_PATH = "/home/shreya/Harsha/qdrant_db"
COLLECTION_NAME = "telecom_rag"

TOP_K = 10
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

client = QdrantClient( path=QDRANT_PATH)

# test_queries = [

#     # RRC
#     "What is RRC Reconfiguration?",
#     "How does RRC Connection Reconfiguration work?",
#     "What is SRB1?",
#     "What is SRB2?",
#     "What is SRB3?",
#     "What is an RRC message?",
#     "How does RRC Resume work?",
#     "How does RRC Setup work?",
#     "What is RRC Release?",
#     "How does RRC Inactive state work?",

#     # Mobility
#     "How does handover work?",
#     "What is inter-gNB handover?",
#     "What is Xn handover?",
#     "What is conditional handover?",
#     "How does cell reselection work?",
#     "What is mobility robustness optimization?",

#     # NGAP / Core
#     "What is NGAP?",
#     "What is AMF?",
#     "What is UPF?",
#     "What is SMF?",
#     "How does PDU Session Establishment work?",
#     "How does Registration procedure work?",
#     "What is NAS signalling?",

#     # QoS
#     "What is a QoS Flow?",
#     "What is 5QI?",
#     "What is GBR?",
#     "What is non-GBR traffic?",
#     "How are QoS flows mapped to DRBs?",
#     "What is reflective QoS?",

#     # Layer 2
#     "What is PDCP?",
#     "What is RLC AM?",
#     "What is RLC UM?",
#     "What is MAC CE?",
#     "What is SDAP?",
#     "How does HARQ work?",
#     "What is packet duplication?",

#     # Radio Resource Management
#     "What is SCell?",
#     "What is PCell?",
#     "What is PSCell?",
#     "What is carrier aggregation?",
#     "What is dual connectivity?",
#     "What is EN-DC?",
#     "What is NR-DC?",

#     # Measurements
#     "What is CQI?",
#     "What is PMI?",
#     "What is RI?",
#     "How does beam management work?",
#     "How does measurement reporting work?",

#     # Physical Layer
#     "What is PRB?",
#     "What is numerology in 5G?",
#     "What is SSB?",
#     "What is PDCCH?",
#     "What is PDSCH?",
#     "What is PUSCH?",
#     "What is PRACH?",

#     # O-RAN
#     "What is E2AP?",
#     "What is E2SM-KPM?",
#     "What is E2SM-RC?",
#     "What is a Near-RT RIC?",
#     "What is a Non-RT RIC?",
#     "How does the E2 interface work?",

#     # Network Slicing
#     "What is network slicing?",
#     "How is slice selection performed?",
#     "What is S-NSSAI?",
#     "What is NSSAI?",

#     # Security
#     "How does 5G security work?",
#     "What is AS security?",
#     "What is NAS security?",
#     "How are security keys derived?",
#     "What is K_gNB?",

#     # Advanced Features
#     "What is sidelink communication?",
#     "What is V2X?",
#     "What is URLLC?",
#     "What is mMTC?",
#     "What is eMBB?",
#     "How does packet duplication improve reliability?"
# ]
test_queries=[
    "During an inter-gNB handover, how are security keys updated and which RRC messages are exchanged between the UE and network?",
    "How does a Near-RT RIC use E2AP procedures to control an E2 Node?",
    "What is the difference between SRB1, SRB2 and SRB3?",
    "How does a UE move from idle state to exchanging user data with the network?",
    "How can a Near-RT RIC influence handover decisions in a 5G network?",
    
]

for query in test_queries:
    print("=" * 100)
    print("QUERY:", query)
    print("=" * 100)

    query_vector = model.encode(query,normalize_embeddings=True)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=TOP_K).points
    
    seen = set()
    unique_results = []
    for hit in results:
        payload = hit.payload
    
        key = (payload.get("spec"),payload.get("section"))
        if key in seen:
            continue
    
        seen.add(key)
        unique_results.append(hit)
    
    print("\nTop Results")
    print("=" * 100)
    
    for rank, hit in enumerate(unique_results, start=1):
    
        payload = hit.payload
    
        print(f"\nRank #{rank}")
        print( "Score:",round(hit.score, 4))
        print("Spec:", payload.get("spec"))
        print("Release:",payload.get("release_file"))
        print("Section:",payload.get("section"))
        print("Title:",payload.get("title"))
        print("-" * 80)
        print(payload.get("text","")[:1000])
        print("=" * 100)
        print()
   
    