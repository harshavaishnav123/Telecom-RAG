import json
import re

INPUT_FILE =  r"....\vocab.jsonl"
OUTPUT_FILE = r"....\telecom_dictionary.jsonl"

telecom_dict = {}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        release_file = row.get("release_file", "")
        spec = row.get("spec", "")
        group = row.get("group", "")
        section = row.get("section", "")
        source = row.get("source", "")
        text = row.get("text", "")

        text = text.replace("\\t", "\t").replace("\\n", "\n")

        for entry in text.split("\n"):
            entry = entry.strip()
            if not entry:
                continue

            match_abbrev = re.match(r"^([A-Za-z0-9_\-\(\)/\s\.]+?)(?:\t+|\s{2,})(.+)$", entry)
            
            match_definition = re.match(r"^([A-Za-z0-9_\-\(\)/\s\.\_]+?)\s*:\s*(.+)$", entry)

            if match_abbrev:
                acronym = match_abbrev.group(1).strip()
                definition = match_abbrev.group(2).strip()
            elif match_definition:
                acronym = match_definition.group(1).strip()
                definition = match_definition.group(2).strip()
            else:
                continue

            if len(acronym) > 40 or acronym == "0-9" or acronym.lower().startswith("for the purposes"):
                continue

            if acronym not in telecom_dict or len(definition) > len(telecom_dict[acronym]["text"]):
                telecom_dict[acronym] = {
                    "doc_id": f"{spec}_{acronym}",
                    "source": source,
                    "spec": spec,
                    "group": group,
                    "release_file": release_file,
                    "section": section,
                    "title": acronym,
                    "text": definition,
                }

print(f"Unique terms processed: {len(telecom_dict)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for value in sorted(telecom_dict.values(), key=lambda x: x["title"]):
        f.write(json.dumps(value, ensure_ascii=False) + "\n")

print(f"Saved cleanly to: {OUTPUT_FILE}")