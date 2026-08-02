import re
import argparse
import subprocess
import chromadb
import ollama
from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from helper import collect_file_names
from consts import PARSERS, CHUNK_NODE_TYPES, DB_PATH, EMBEDDING_MODEL, POSTCOMMIT_MARKER, INTERPRETER_HOOK_LINES

def ast_parse_chunk(file_path: Path):
    if not file_path.exists():
        return []

    suffix = file_path.suffix
    parser = PARSERS.get(suffix)

    node_types = CHUNK_NODE_TYPES.get(suffix)
    code_bytes = file_path.read_bytes()
    tree = parser.parse(code_bytes)
    root_node = tree.root_node

    chunks = []
    for child in root_node.children:
        if child.type in node_types:
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

def get_changed_files() -> List[Tuple[Path, str]]:
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

            if "R" in status:
                old_path, _, new_path = filename.partition(" -> ") # partition on file rename, catch old file/new file via porcelain, old file is caught in the early chunk deletion
                changed_files.append((Path(old_path), "D"))
                changed_files.append((Path(new_path), "A"))
            elif any(s in status for s in ["M", "A", "?", "D"]):
                changed_files.append((Path(filename), status))
        return changed_files
    except subprocess.CalledProcessError:
        print("[-] Not a git repository or git failed, falling back to full scan.")
        return [(f, "A") for f in collect_file_names()]
        
def index_codebase(force_full: bool = False):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="code_semantic_index")

    is_empty = collection.count() == 0

    if force_full or is_empty:
        print("[*] Cold start or full re-index requested. Scanning entire codebase...")
        target_files = [(f, "A") for f in collect_file_names()]
    else:
        target_files = get_changed_files()
        if not target_files:
            print("[*] No file changes via git, index is up to date.")
            return

    print(f"Processing {len(target_files)} file(s)...")

    for file_path, status in target_files:
        if not file_path.exists():
            if "D" in status:
                collection.delete(where={"file": str(file_path)})
                print(f"[-] Removed deleted file from index: {file_path}")
            continue

        ast_chunks = ast_parse_chunk(file_path)
        if not ast_chunks:
            continue

        texts = [chunk["text"] for chunk in ast_chunks]
        embed_inputs = [f"search_document: {t}" for t in texts]
        response = ollama.embed(model=EMBEDDING_MODEL, input=embed_inputs)

        collection.delete(where={"file": str(file_path)})
        collection.upsert(
            ids = [f"{file_path}_chunk_{i}" for i in range(len(ast_chunks))],
            embeddings = response.embeddings,
            metadatas=[{"file": str(file_path), "start_line": c["start_line"], "end_line": c["end_line"]}for c in ast_chunks],
            documents = texts
        )

    print("[+] Indexing complete.")

def hybrid_search(query: str, top_k: int = 3):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="code_semantic_index")

    all_data = collection.get(include=["metadatas","documents","embeddings"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    embeddings = all_data["embeddings"]
    ids = all_data["ids"]

    if not documents:
        print("[-] Index is empty. Run 'index' command first.")
        return

    # ------ BM25 (keyword) half --------
    tokenized_corpus = [re.findall(r"\w+", doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # ------ semantic (vector) half ------
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=f"search_query: {query}").embeddings[0]
    semantic_results = collection.query(query_embeddings=[query_embedding], n_results=len(ids))

    # ------ fuse rankings via Reciprocal Rank Fusion ---
    bm25_ranked_ids = [ids[i] for i in bm25_scores.argsort()[::-1]]
    semantic_ranked_ids = semantic_results["ids"][0]

    k = 60
    fused_scores = {}

    for rank, doc_id in enumerate(bm25_ranked_ids):
        fused_scores[doc_id] = 1.0 / (k + rank)
    for rank, doc_id in enumerate(semantic_ranked_ids):
        fused_scores[doc_id] += 1.0 / (k + rank)

    top_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    print(f"\n=== Hybrid Search Query Results for: '{query}' ===")

    for rank, doc_id in enumerate(top_ids):
        idx = ids.index(doc_id)
        metadata = metadatas[idx]
        signature = documents[idx].strip().splitlines()[0]
        print(f"\n[{rank+1}] {metadata['file']}:{metadata['start_line'] + 1}  {signature}")

def ensure_gitignore_entry(entry: str):
    toplevel = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True
    ).stdout.strip())

    gitignore_path = toplevel / ".gitignore"
    existing = gitignore_path.read_text() if gitignore_path.exists() else ""
    if entry in existing.splitlines():
        return

    with gitignore_path.open("a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"{entry}\n")
    print(f"[+] Added '{entry}' to {gitignore_path}")

def detect_interpreter(shebang_line: str):
    if not shebang_line.startswith("#!"):
        return None
    parts = shebang_line[2:].split()
    if not parts:
        return None
    interpreter_path = parts[-1] if parts[0].endswith("env") else parts[0]
    return Path(interpreter_path).name

def install_postcommit_hook():
    try:
        hooks_dir = Path(subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            capture_output=True, text=True, check=True
        ).stdout.strip())
    except subprocess.CalledProcessError:
        print("[-] Not inside a git repository.")
        return

    ensure_gitignore_entry(f"{DB_PATH}/")
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "post-commit"

    if hook_path.exists():
        existing = hook_path.read_text()
        if POSTCOMMIT_MARKER in existing:
            print(f"[*] codeburrow post-commit hook already installed at {hook_path}")
            return

        first_line = existing.splitlines()[0] if existing else ""
        interpreter = detect_interpreter(first_line)
        hook_line = INTERPRETER_HOOK_LINES.get(interpreter)

        if hook_line is None:
            print(f"[-] Existing post-commit hook uses an unrecognized interpreter ({first_line or 'no shebang'}). "
                  f"Add this manually: {INTERPRETER_HOOK_LINES['bash'].splitlines()[1]}")
            return

        with hook_path.open("a") as f:
            f.write(f"\n{hook_line}")
        print(f"[+] Appended codeburrow indexing to existing hook: {hook_path}")
    else:
        hook_line = INTERPRETER_HOOK_LINES["bash"]
        hook_path.write_text(f"#!/bin/bash\n{hook_line}")
        print(f"[+] Installed post-commit hook: {hook_path}")

    hook_path.chmod(hook_path.stat().st_mode | 0o111)

def main():
    parser = argparse.ArgumentParser(prog="codeburrow", description="Hybrid semantic + keyword code search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index the current repository")
    index_parser.add_argument("--force", action="store_true", help="Force a full re-index instead of an incremental one")

    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("query")

    subparsers.add_parser("postcommit-install", help="Install a git post-commit hook that re-indexes on every commit")

    args = parser.parse_args()

    if args.command == "index":
        index_codebase(force_full=args.force)
    elif args.command == "search":
        hybrid_search(args.query)
    elif args.command == "postcommit-install":
        install_postcommit_hook()

if __name__ == "__main__":
    main()

