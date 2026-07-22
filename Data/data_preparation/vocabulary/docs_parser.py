from docx import Document
from pathlib import Path
import re
import json

ROOT = r"....\23288"
OUTPUT_FILE = r"....\all_data.jsonl"
docx_files = list(Path(ROOT).rglob("*.docx"))

print("DOCX count:", len(docx_files))

def get_spec_number(filename):

    match = re.search(r"(\d{5})", filename)
    if not match:
        return "unknown"
    digits = match.group(1)
    return f"{digits[:2]}.{digits[2:]}"


SECTION_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")

SPEC_DOC_PATTERN = re.compile(r"^\d{5}.*\.docx$",re.IGNORECASE)

def parse_docx(docx_path):

    doc = Document(docx_path)

    spec = get_spec_number(Path(docx_path).stem)
    release_file = Path(docx_path).stem
    sections = []

    current_section = None
    current_title = None
    buffer = []

    started = False

    SECTION_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)[\t ]+(.+)$')

    for para in doc.paragraphs:

        text = para.text.strip()

        if not text:
            continue

        started = True

        if not started:
            continue

        match = SECTION_PATTERN.match(text)

        if match:

            if current_section is not None:

                section_text = "\n".join(buffer).strip()

                INVALID_TEXTS = {"","void","void."}

                if section_text.lower().strip() in INVALID_TEXTS:
                      continue
                
                sections.append({
                    "doc_id": f"{release_file}_{current_section}",
                    "source": "3gpp",
                    "spec": spec,
                    "group": "Data Plane",
                    "release_file": release_file,
                    "section": current_section,
                    "title": current_title,
                    "level": current_section.count(".") + 1,
                    "text": section_text
                })

            current_section = match.group(1)
            current_title = match.group(2)

            buffer = []

        else:

            buffer.append(text)

    if current_section is not None:

        section_text = "\n".join(buffer).strip()

        INVALID_TEXTS = {"","void","void."}

        if section_text.lower().strip() not in INVALID_TEXTS:

             sections.append({
                "doc_id": f"{release_file}_{current_section}",
                "source": "3gpp",
                "spec": spec,
                "group": "Data Plane",
                "release_file": release_file,
                "section": current_section,
                "title": current_title,
                "level": current_section.count(".") + 1,
                "text": section_text
            })

    unique_sections = {}

    for sec in sections:
    
        key = (sec["release_file"],sec["section"])
    
        unique_sections[key] = sec
    
    return list(unique_sections.values())

docx_files = list(Path(ROOT).rglob("*.docx"))

print(f"Found {len(docx_files)} DOCX files")

processed = 0
written = 0

with open(OUTPUT_FILE,"w",encoding="utf-8") as fout:

    for docx_file in docx_files:

        filename = docx_file.name

        if "cover sheet" in filename.lower():
            continue

        try:

            sections = parse_docx(str(docx_file))

            print(f"{filename} -> {len(sections)} sections")

            for sec in sections:

                fout.write(json.dumps(sec,ensure_ascii=False) + "\n")

                written += 1

            processed += 1

        except Exception as e:

            print(f"Failed: {filename}")

            print(e)

print("\nDone")
print("Processed files:", processed)
print("Written sections:", written)
print("Output:", OUTPUT_FILE)