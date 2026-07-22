import json

INPUT_FILE =    r"....\all_data.jsonl"
OUTPUT_FILE =   r"....\data.jsonl"
OUTPUT_FILE1 =  r"....\vocab.jsonl"
OUTPUT_FILE2=   r"....\waste.jsonl"
OUTPUT_FILE3=   r"....\terms.jsonl"
OUTPUT_FILE4=   r"....\terms1.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout,\
     open(OUTPUT_FILE1, "w", encoding="utf-8") as fout1,\
     open(OUTPUT_FILE2, "w", encoding="utf-8") as fout2,\
     open(OUTPUT_FILE3, "w", encoding="utf-8") as fout3,\
     open(OUTPUT_FILE4, "w", encoding="utf-8") as fout4:

    for line in fin:

        row = json.loads(line)
        arr1=["Normative references","Informative references","Control action","Offset"]
        arr2=["650","1","2"]
        if str(row.get("section")) in arr2 or str(row.get("title")) in arr1 :
            fout2.write(json.dumps(row,ensure_ascii=False) + "\n")
        elif row.get("section")=="3.2":
            fout3.write(json.dumps(row,ensure_ascii=False) + "\n")
        elif row.get("section")=="3":
            fout1.write(json.dumps(row,ensure_ascii=False) + "\n") 
        elif row.get("section")=="3.1" or row.get("section")=="3.3":
            fout4.write(json.dumps(row,ensure_ascii=False) + "\n")     
        else:
            fout.write(json.dumps(row,ensure_ascii=False) + "\n")

print("Done")