import json

INPUT_FILE = r"....\all_data.jsonl"
OUTPUT_FILE = r"....\dictio.jsonl"
OUTPUT_FILE1 = r"....\waste.jsonl"
OUTPUT_FILE2= r"....\waste1.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout,\
     open(OUTPUT_FILE1, "w", encoding="utf-8") as fout1,\
     open(OUTPUT_FILE2, "w", encoding="utf-8") as fout2:

    for line in fin:

        row = json.loads(line)
        arr1=["2","650"]
        if str(row.get("section")) in arr1 or str(row.get("title"))=="Scope":
            fout1.write(json.dumps(row,ensure_ascii=False) + "\n")
        elif row.get("section")=="5":
            fout2.write(json.dumps(row,ensure_ascii=False) + "\n")
        else:
          fout.write(json.dumps(row,ensure_ascii=False) + "\n")

print("Done")