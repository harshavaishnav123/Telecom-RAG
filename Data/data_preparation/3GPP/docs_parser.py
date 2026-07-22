from docx import Document
from docx.oxml import CT_P, CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
from pathlib import Path
import re
import json

ROOT = r"....\RAN_ARCH"
OUTPUT_FILE = r"....\all_data.jsonl"
docx_files = list(Path(ROOT).rglob("*.docx"))

print("DOCX count:", len(docx_files))

def get_spec_number(filename):
    match = re.search(r"(\d{5})", filename)
    if not match:
        return "unknown"
    digits = match.group(1)
    return f"{digits[:2]}.{digits[2:]}"

def iter_block_items(parent):
    """
        Yield each paragraph and table element within the document body 
        in the order they appear, safely avoiding all isinstance type bugs.
    """
    if hasattr(parent, 'element') and hasattr(parent.element, 'body'):
        parent_elm = parent.element.body
    else:
        raise ValueError("Unsupported parent type for block iteration")

    for child in parent_elm.iterchildren():
        tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        class_name = child.__class__.__name__
        
        if class_name == "CT_P" or tag_name == "p":
            yield Paragraph(child, parent)
        elif class_name == "CT_Tbl" or tag_name == "tbl":
            yield Table(child, parent)

def table_to_markdown(table):
    """
        Converts a python-docx table object into a Markdown string representation.
    """
    markdown_lines = []
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        
        markdown_lines.append("| " + " | ".join(cells) + " |")
        
        if i == 0 and len(table.rows) > 1:
            markdown_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
            
    return "\n".join(markdown_lines)

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

    for block in iter_block_items(doc):
        
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue

            started = True
            
            match = SECTION_PATTERN.match(text)
            if match:
                if current_section is not None:
                    section_text = "\n".join(buffer).strip()
                    INVALID_TEXTS = {"", "void", "void."}

                    if section_text.lower().strip() not in INVALID_TEXTS:
                        sections.append({
                            "doc_id": f"{release_file}_{current_section}",
                            "source": "3gpp",
                            "spec": spec,
                            "group": "RAN ARCHITECTURE",
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
                
        elif isinstance(block, Table) and started:
            table_text = table_to_markdown(block)
            if table_text.strip():
                buffer.append("\n" + table_text + "\n")

    if current_section is not None:
        section_text = "\n".join(buffer).strip()
        INVALID_TEXTS = {"", "void", "void."}

        if section_text.lower().strip() not in INVALID_TEXTS:
            sections.append({
                "doc_id": f"{release_file}_{current_section}",
                "source": "3gpp",
                "spec": spec,
                "group": "RAN ARCHITECTURE",
                "release_file": release_file,
                "section": current_section,
                "title": current_title,
                "level": current_section.count(".") + 1,
                "text": section_text
            })

    unique_sections = {}
    for sec in sections:
        key = (sec["release_file"], sec["section"])
        unique_sections[key] = sec
    
    return list(unique_sections.values())

docx_files = list(Path(ROOT).rglob("*.docx"))
print(f"Found {len(docx_files)} DOCX files")

processed = 0
written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for docx_file in docx_files:
        filename = docx_file.name

        if "cover sheet" in filename.lower():
            continue

        try:
            sections = parse_docx(str(docx_file))
            print(f"{filename} -> {len(sections)} sections")

            for sec in sections:
                fout.write(json.dumps(sec, ensure_ascii=False) + "\n")
                written += 1

            processed += 1

        except Exception as e:
            print(f"Failed: {filename}")
            print(e)

print("\nDone")
print("Processed files:", processed)
print("Written sections:", written)
print("Output:", OUTPUT_FILE)