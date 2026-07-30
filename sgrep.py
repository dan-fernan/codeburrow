
import sys
import subprocess
import chromadb
import ollama
from pathlib import Path
from typing import List, Dict, Any
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from helper import collect_file_names

PYTHON_LANGUAGE = Language(tspython.language())
# tspython.language() returns a pointer to the compiled C def'n for python
# Language() takes that pointer and turns it into a python object, exposing it to high level methods

parser = Parser(PYTHON_LANGUAGE)
# Instantiates the engine that executes the parsing process

DB_PATH = Path("./.sgreb_db")
EMBEDDING_MODEL = "nomic-embed-text"

def ast_parse_chunk(file_path: Path):
    if not file_path.exists():
        return []
    
    code_bytes = file_path.read_bytes()
    tree = parser.parse(code_bytes)
    root_node = tree.root_node

    chunks = []
    for child in root_node.children:
        if child.type in ["function_definition", "async_function_definition", "class_definition", "decorated_definition"]:
            chunk_text = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
            chunks.append({
                "text": chunk_text,
                "start_line": child.start_point[0],
                "end_line": child.end_point[0]
            })

    if not chunks:
        text_content = code_bytes.decode("utf-8")
        chunks.append({
            "text": text_content,
            "start_line": 0,
            "end_line": text_content.count("\n")
        })

    return chunks

def get_changed_files() -> List[Path]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True # Captures output to stdout obj, decode stdout as str, and raises a calledprocesserror if exited with a non-zero return code
        )
        changed_files = []
        for line in result.stdout.splitlines(): 
            if not line.strip():
                continue
            status = line[:2]
            filename = line[3:]

            if any(s in status for s in ["M", "A", "?", "R"]):
                changed_files.append(Path(filename))
        return changed_files
    except subprocess.CalledProcessError:
        print("[-] Not a git repository or git failed, falling back to full scan.")
        return collect_file_names()
        
def index_codebase(force_full: bool = False):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="code_semantic_index")

    is_empty = collection.count() == 0

    if force_full or is_empty:
        print("[*] Cold start or full re-index requested. Scanning entire codebase...")
        target_files = collect_file_names()
    else:
        target_files = get_changed_files()
        if not target_files:
            print("[*] No file changes via git, index is up to date.")
            return

    print(f"Processing {len(target_files)} file(s)...")

    for file_path in target_files:
        if not file_path.exists():
            continue

        ast_chunks = ast_parse_chunk(file_path)
        if not ast_chunks:
            continue

        texts = [chunk["text"] for chunk in ast_chunks]
        response = ollama.embed(model=EMBEDDING_MODEL, input=texts)

        collection.upsert(
            ids = [f"{file_path}_chunk_{i}" for i in range(len(ast_chunks))],
            embeddings = response.embeddings,
            metadatas=[{"file": str(file_path), "start_line": c["start_line"], "end_line": c["end_line"]}for c in ast_chunks],
            documents = texts
        )

    print("[+] Indexing complete.")










print(ast_parse_chunk(Path("./test.py")))
print(get_changed_files())

