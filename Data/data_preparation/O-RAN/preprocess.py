import pandas as pd

INPUT_FILE = r"....\data.jsonl"
OUTPUT_FILE = r"....\data.jsonl"

df = pd.read_json(
    INPUT_FILE,
    lines=True
)

before = len(df)

df = df.drop_duplicates(subset=["title", "text"])

df = df.drop(columns=["doc_id", "level","version"],errors="ignore")

after = len(df)

print("Before:", before)
print("After :", after)
print("Removed:", before - after)

df.to_json(OUTPUT_FILE,orient="records",lines=True,force_ascii=False)