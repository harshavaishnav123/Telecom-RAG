import json
import re
from pathlib import Path
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

ROOT = r"....\edge"
OUTPUT_FILE = r"....\all_data.jsonl"

def get_oran_spec(filename):
    stem = Path(filename).stem
    match = re.match(r"([A-Za-z0-9\-]+)-R\d+-v[\d\.]+", stem, re.IGNORECASE)
    return match.group(1).upper() if match else stem

def get_version(filename):
    stem = Path(filename).stem
    match = re.search(r"(v[\d\.]+)", stem, re.IGNORECASE)
    return match.group(1) if match else ""

def iter_block_items(doc_obj):
    """
        Yield each paragraph and table element in the exact order 
        they appear in the document layout.
    """
    if hasattr(doc_obj, "element") and hasattr(doc_obj.element, "body"):
        parent_elm = doc_obj.element.body
    else:
        raise ValueError("Provided object is not a valid docx Document instance.")

    for child in parent_elm.iterchildren():
        if child.tag.endswith('p'):
            yield "paragraph", Paragraph(child, doc_obj)
        elif child.tag.endswith('tbl'):
            yield "table", Table(child, doc_obj)

def extract_table_text(table):
    """Converts a Word table into a readable structured text format."""
    table_text = []
    for row in table.rows:
        row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
        if row_cells:
            table_text.append(" | ".join(row_cells))
    return "\n".join(table_text)

def parse_docx_completely(docx_path):
    doc = Document(docx_path)
    release_file = Path(docx_path).stem
    spec = get_oran_spec(release_file)
    version = get_version(release_file)

    sections = []
    current_section = None
    current_title = None
    current_level = 1
    buffer = []
    started = False

    heading_regex = re.compile(r'^(\d+(?:\.\d+)*)[\t ]+(.+)$')

    for item_type, item in iter_block_items(doc):
        if item_type == "table":
            if started and current_section:
                t_text = extract_table_text(item)
                if t_text:
                    buffer.append(t_text)
            continue

        para = item
        text = para.text.strip()
        if not text:
            continue

        style_name = para.style.name if para.style else ""

        if text.lower() in ["foreword", "1\tscope", "1 scope"]:
            started = True
            if text.lower() == "foreword":
                continue

        if not started:
            continue

        is_heading = style_name.startswith("Heading") or heading_regex.match(text)

        if is_heading:
            if current_section:
                section_text = "\n".join(buffer).strip()
                if section_text and section_text.lower() not in {"void", "void."}:
                    sections.append({
                        "doc_id": f"{release_file}_{current_section}",
                        "source": "mec",
                        "spec": spec,
                        "group": "Multi-access Edge",
                        "release_file": release_file,
                        "version": version,
                        "section": current_section,
                        "title": current_title,
                        "level": current_level,
                        "text": section_text
                    })

            match = heading_regex.match(text)
            if match:
                current_section = match.group(1)
                current_title = match.group(2)
            else:
                current_section = text
                current_title = text

            try:
                current_level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                current_level = current_section.count(".") + 1

            buffer = []
        else:
            buffer.append(text)

    if current_section:
        section_text = "\n".join(buffer).strip()
        if section_text and section_text.lower() not in {"void", "void."}:
            sections.append({
                "doc_id": f"{release_file}_{current_section}",
                "source": "mec",
                "spec": spec,
                "group": "Multi-access Edge",
                "release_file": release_file,
                "version": version,
                "section": current_section,
                "title": current_title,
                "level": current_level,
                "text": section_text
            })

    unique_sections = {}
    for sec in sections:
        key = (sec["release_file"], sec["section"])
        unique_sections[key] = sec

    return list(unique_sections.values())

if __name__ == "__main__":
    docx_files = list(Path(ROOT).rglob("*.docx"))
    print(f"Found {len(docx_files)} DOCX files matching execution directory.")

    processed = 0
    written = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for docx_file in docx_files:
            if "cover sheet" in docx_file.name.lower():
                continue
            try:
                sections = parse_docx_completely(docx_file)
                print(f"{docx_file.name} -> Extracted {len(sections)} complete sections (including tables).")
                
                for sec in sections:
                    fout.write(json.dumps(sec, ensure_ascii=False) + "\n")
                    written += 1
                processed += 1
            except Exception as e:
                print(f"Failed parsing file: {docx_file.name}. Error: {e}")

    print("\n--- Processing Finished ---")
    print("Processed Files Count:", processed)
    print("Total Records Exported:", written)
    print("Target Destination JSONL:", OUTPUT_FILE)