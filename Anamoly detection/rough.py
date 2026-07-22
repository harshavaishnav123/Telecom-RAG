import pandas as pd
import json

df = pd.read_csv(r"C:\nihal\PROJECT\Telecom project\Anamoly detection\data\queries.csv")

with open(r"C:\nihal\PROJECT\Telecom project\Anamoly detection\data\r.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        obj = {
            "question": row["query"]
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")