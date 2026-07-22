import json

raw_input_path = "/home/shreya/Harsha/data/error_analysis_qa.jsonl"
repaired_output_path = "/home/shreya/Harsha/data/error_analysis_qa.jsonl"

cleaned_records = []
with open(raw_input_path, "r", encoding="utf-8") as f:
    # Read entire text, strip global array wrappers, split by object closures
    raw_text = f.read().strip()
    if raw_text.startswith("["):
        raw_text = raw_text[1:]
    if raw_text.endswith("]"):
        raw_text = raw_text[:-1]
        
    # Split records by structural boundaries while filtering out blank spaces
    parts = raw_text.split("},")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not part.endswith("}"):
            part += "}"
            
        try:
            cleaned_records.append(json.loads(part))
        except json.JSONDecodeError as e:
            print(f"Skipping heavily malformed record block: {part[:60]}... Error: {e}")

# Save back to disk as a perfect, valid JSONL file
with open(repaired_output_path, "w", encoding="utf-8") as out:
    for record in cleaned_records:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Sanitization complete! Use '{repaired_output_path}' as your new script input.")