import json

INPUT_FILE = r"....\all_data.jsonl"
OUTPUT_FILE = r"....\data.jsonl"
OUTPUT_FILE1 = r"....\vocab.jsonl"
OUTPUT_FILE2= r"....\waste.jsonl"
OUTPUT_FILE3= r"....\terms.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout,\
     open(OUTPUT_FILE1, "w", encoding="utf-8") as fout1,\
     open(OUTPUT_FILE2, "w", encoding="utf-8") as fout2,\
     open(OUTPUT_FILE3, "w", encoding="utf-8") as fout3:

    for line in fin:

        row = json.loads(line)
        arr1=["Normative references","Informative references","Control action","Offset","Annex:","Scope","Introduction","Current type definitions","QoE target","QoS target","Load balancing targets","Energy saving- targets"]
        if str(row.get("section")) =="Modal verbs terminology" or str(row.get("title")) in arr1 :
            fout2.write(json.dumps(row,ensure_ascii=False) + "\n")
        elif row.get("title")=="Terms":
             fout3.write(json.dumps(row,ensure_ascii=False) + "\n")
        elif row.get("title")=="Abbreviations" or row.get("title")=="Symbols":
             fout1.write(json.dumps(row,ensure_ascii=False) + "\n")     
        else:
          fout.write(json.dumps(row,ensure_ascii=False) + "\n")

print("Done")