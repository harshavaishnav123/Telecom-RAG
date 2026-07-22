import zipfile
import os
from docx import Document
from pathlib import Path
import re
import win32com.client 
from pathlib import Path
import shutil
import win32com
import os

# converting the .doc file into the .docx file
def convert_doc_to_docx(doc_path):

    word = win32com.client.dynamic.Dispatch("Word.Application")
    
    word.Visible = False

    doc = word.Documents.Open(str(doc_path))

    docx_path = (str(doc_path) + "x")

    doc.SaveAs(docx_path,FileFormat=16)

    doc.Close()

    return docx_path

# function for unzipping the file
def unzip_file(zip_path, extract_to):
    # validating the zip file path
    if not os.path.isfile(zip_path):
        print("Error: The specified ZIP file does not exist.")
        return
    
    # ensuring the destination directory exist
    try:
        os.makedirs(extract_to, exist_ok=True)
    except OSError as e:
        print(f"Error: Unable to create destination directory. Details: {e}")
        return

    # trying to extract the files
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Files successfully extracted to: {extract_to}")
    except zipfile.BadZipFile:
        print("Error: The file is not a valid ZIP archive.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    zip_path = r"....\38213"
    extract_to = r"....\38213" 
    docx_files = list(Path(zip_path).rglob("*.zip"))
    for x in docx_files:
      unzip_file(x, extract_to)
    for doc_file in Path(extract_to).rglob("*.doc") :
      if doc_file not in Path(extract_to).rglob("*.docx"):
        convert_doc_to_docx(doc_file)
    
    for file in Path(extract_to).rglob("*"):

       if file.suffix.lower() in [".zip", ".doc"]:
        file.unlink()
        print("Deleted:", file)