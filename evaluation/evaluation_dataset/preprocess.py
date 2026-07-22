import pandas as pd

INPUT_FILE = r"C:\nihal\PROJECT\Telecom project\Data\Final\data.jsonl"
OUTPUT_FILE = r"C:\nihal\PROJECT\Telecom project\Data\Final\data.jsonl"

df = pd.read_json(INPUT_FILE,lines=True)

before = len(df)

df = df.drop_duplicates(subset=["title", "text"])
after = len(df)

print("Before:", before)
print("After :", after)
print("Removed:", before - after)

df.to_json(OUTPUT_FILE,orient="records",lines=True,force_ascii=False)