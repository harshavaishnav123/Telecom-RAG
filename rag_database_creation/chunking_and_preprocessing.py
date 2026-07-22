import json
import pandas as pd
import re
from transformers import AutoTokenizer

# declaring the required input and output files
INPUT_FILE = r"..../ran_arch.jsonl"
CLEAN_OUTPUT = r"..../ran_arch_cleaned.jsonl"
CHUNK_OUTPUT = r"..../ran_arch_chunks.jsonl"

# initializing the chunk size and chunk overlap
CHUNK_SIZE = 1000
OVERLAP = 200

# getting the required data from the input file
data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print("=" * 80)
print("Loaded documents:", len(data))
print("=" * 80)

# initializing the data as a dataframe
df = pd.DataFrame(data)

# if doc id donot exist then we are creating the column called doc_id
if "doc_id" not in df.columns:

    df["doc_id"] = (df["release_file"].astype(str) + "_"
        + df["section"].astype(str)+"_"+ df["title"].astype(str))

# dropping duplicates based on doc_id
df = df.drop_duplicates(subset=["doc_id"])
print("After dedup:", len(df))
INVALID_TEXTS = {"", "void","void."}
# removing the invalid texts from the document
df = df[~df["text"].fillna("").str.lower().str.strip().isin(INVALID_TEXTS)]
print("After void removal:", len(df))

# function for cleaning the text for handling missing values,etc
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Applying the function on df['text']
df["text"] = df["text"].apply(clean_text)

# writing all the clean rows into the clean output file
with open(CLEAN_OUTPUT, "w", encoding="utf-8") as f:
    for row in df.to_dict(orient="records"):
        f.write(json.dumps(row,ensure_ascii=False)+ "\n")
print("Saved cleaned file:")
print(CLEAN_OUTPUT)

# loading the BAAI tokenizer
tokenizer = AutoTokenizer.from_pretrained( "BAAI/bge-large-en-v1.5")

# function for chunking the text
def chunk_text( text, chunk_size=CHUNK_SIZE,overlap=OVERLAP):

    if overlap >= chunk_size:
        raise ValueError("OVERLAP must be strictly less than CHUNK_SIZE.")
    
    # encoding tokens from tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk = tokenizer.decode(chunk_tokens,skip_special_tokens=True)
        chunks.append(chunk)
        # for avoiding duplicating tail chunks we do this
        if end >= len(tokens):
            break
        start += step
    return chunks

# this loop collects all the chunked records in the required format
chunk_records = []
for _, row in df.iterrows():
    text = row["text"]
    text_chunks = chunk_text(text)
    for idx, chunk in enumerate(text_chunks):
        token_count = len(tokenizer.encode(chunk,add_special_tokens=False))
        doc_id=str(row.get("release_file",""))+"-"+ str(row.get("section",""))
        chunk_records.append({
                             "chunk_id":f"{doc_id}_{idx}",                             
                             "source":row.get("source","3gpp"),                             
                             "release_file":row.get("release_file", ""),                             
                             "spec": row.get("spec",""),
                             "group": row.get("group",""),                             
                             "section":row.get("section",""),                             
                             "title":row.get("title",""),
                             "token_count": token_count,                             
                             "text":chunk})

print("=" * 80)
print("Generated chunks:", len(chunk_records))
print("=" * 80)

# in this loop the chunked records are added to the chunk output file
with open(CHUNK_OUTPUT,"w",encoding="utf-8") as f:
    for chunk in chunk_records:
        f.write(json.dumps(chunk,ensure_ascii=False)+ "\n")
print("=" * 80)
print("Saved chunks:")
print(CHUNK_OUTPUT)
print("=" * 80)