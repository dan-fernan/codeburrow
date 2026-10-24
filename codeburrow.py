import re
import json
import argparse
import subprocess
import chromadb
import voyageai
import numpy as np
from pathlib import Path
from typing import List, Optional
from rank_bm25 import BM25Okapi
from helper import collect_file_names, is_indexable
from consts import PARSERS, CHUNK_NODE_TYPES, CLASS_NODE_TYPES, METHOD_NODE_TYPES, DB_PATH, INDEX_STATE_PATH, EMBEDDING_MODEL, RRF_K, SEMANTIC_WEIGHT, BM25_WEIGHT, POSTCOMMIT_MARKER, INTERPRETER_HOOK_LINES

VOYAGE_CLIENT = voyageai.Client()

def make_chunk(code_bytes: bytes, node):
    return {
        "text": code_bytes[node.start_byte:node.end_byte].decode("utf-8"),
        "start_line": node.start_point[0],
        "end_line": node.end_point[0]
    }

def find_class(node):
    if node.type in CLASS_NODE_TYPES:
        return node
    if node.type in ("decorated_definition", "export_statement"): # @decorator class Foo / export class Foo
        for child in node.named_children:
            if child.type in CLASS_NODE_TYPES:
                return child
    return None

def get_methods(class_node):
    body = class_node.child_by_field_name("body")
    methods = []
    for child in (body.named_children if body else []):
        if child.type in METHOD_NODE_TYPES:
            methods.append(child)
            nested_class = find_class(child)
            if nested_class:
                methods += get_methods(nested_class)
    return methods

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
            chunks.append(make_chunk(code_bytes, child))
            class_node = find_class(child)
            if class_node:
                chunks += [make_chunk(code_bytes, method) for method in get_methods(class_node)]

    if not chunks:
        text_content = code_bytes.decode("utf-8")
        if text_content.strip():
            chunks.append({
                "text": text_content,
                "start_line": 0,
                "end_line": text_content.count("\n")
            })

    return chunks

def run_git(*args) -> List[str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()

def get_head_commit() -> Optional[str]:
    try:
        return run_git("rev-parse", "HEAD")[0]
    except subprocess.CalledProcessError:
        return None # not a git repo, or no commits yet

def load_index_state():
    if not INDEX_STATE_PATH.exists():
        return None
    return json.loads(INDEX_STATE_PATH.read_text())

def save_index_state(commit: str, dirty_files: List[Path]):
    INDEX_STATE_PATH.write_text(json.dumps({"commit": commit, "dirty": sorted(str(f) for f in dirty_files)}))

def get_working_tree_changes() -> List[Path]:
    changed = []
    for line in run_git("status", "--porcelain", "--untracked-files=all"): # list files inside new untracked dirs, not the dir itself
        if not line.strip():
            continue
        filename = line[3:]
        if "R" in line[:2]:
            old_path, _, new_path = filename.partition(" -> ")
            changed += [Path(old_path), Path(new_path)]
        else:
            changed.append(Path(filename))
    return changed

def get_changed_files(state) -> Optional[List[Path]]:
    try:
        run_git("cat-file", "-e", f"{state['commit']}^{{commit}}")
    except subprocess.CalledProcessError:
        return None # last indexed commit is gone (e.g. rebased away), caller falls back to a full scan

    # committed since last index + uncommitted now + uncommitted at last index (catches reverted edits)
    changed = set(get_working_tree_changes()) | {Path(f) for f in state["dirty"]}
    for line in run_git("diff", "--name-status", "-M", state["commit"], "HEAD"):
        changed.update(Path(p) for p in line.split("\t")[1:]) # renames list both old and new path
    return [f for f in changed if is_indexable(f)]

def index_codebase(force_full: bool = False):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="code_semantic_index")

    head = get_head_commit()
    dirty = get_working_tree_changes() if head else []
    state = load_index_state()

    target_files = None
    if not force_full and collection.count() > 0 and state is not None:
        target_files = get_changed_files(state)

    if target_files is None:
        print("[*] Cold start or full re-index requested. Scanning entire codebase...")
        target_files = collect_file_names()
        indexed = {m["file"] for m in collection.get(include=["metadatas"])["metadatas"]}
        for stale in indexed - {str(f) for f in target_files}:
            collection.delete(where={"file": stale})
    elif not target_files:
        print("[*] No file changes via git, index is up to date.")
    else:
        print(f"Processing {len(target_files)} file(s)...")

    for file_path in target_files:
        collection.delete(where={"file": str(file_path)})
        if not file_path.is_file():
            print(f"[-] Removed deleted file from index: {file_path}")
            continue

        ast_chunks = ast_parse_chunk(file_path)
        if not ast_chunks:
            continue

        texts = [chunk["text"] for chunk in ast_chunks]
        response = VOYAGE_CLIENT.embed(texts, model=EMBEDDING_MODEL, input_type="document")

        collection.upsert(
            ids = [f"{file_path}_chunk_{i}" for i in range(len(ast_chunks))],
            embeddings = response.embeddings,
            metadatas=[{"file": str(file_path), "start_line": c["start_line"], "end_line": c["end_line"]}for c in ast_chunks],
            documents = texts
        )

    if head:
        save_index_state(head, dirty)
    print("[+] Indexing complete.")

def tokenize(text: str) -> List[str]:
    tokens = []
    for word in re.findall(r"\w+", text):
        tokens.append(word.lower())
        subwords = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", word) # parse_header_links -> parse/header/links, getHTTPResponse -> get/HTTP/Response
        if len(subwords) > 1:
            tokens += [s.lower() for s in subwords]
    return tokens

def rank_chunks(query: str, documents: List[str], embeddings, query_embedding) -> List[int]:
    n = len(documents)

    # ------ BM25 (keyword) half --------
    bm25 = BM25Okapi([tokenize(doc) for doc in documents])
    # argsort sorts ascending, so negate scores to get chunk positions best-first; "stable" keeps ties in original order
    bm25_ranked = np.argsort(-bm25.get_scores(tokenize(query)), kind="stable")

    # ------ semantic (vector) half ------
    # matrix @ vector = dot product of the query with every chunk embedding in one step, i.e.
    # [sum(e * q for e, q in zip(chunk, query_embedding)) for chunk in embeddings]
    # voyage embeddings are unit-length, so dot product = cosine similarity (same order chroma's L2 distance gives)
    similarities = np.asarray(embeddings) @ np.asarray(query_embedding)
    semantic_ranked = np.argsort(-similarities, kind="stable")

    # ------ weighted Reciprocal Rank Fusion ------
    # each chunk earns weight / (k + rank) from each ranking. fused[ranked] += points[rank] is equivalent to:
    #   for rank, idx in enumerate(ranked): fused[idx] += weight / (RRF_K + rank)
    rrf_points = 1.0 / (RRF_K + np.arange(n))
    fused = np.zeros(n)
    fused[bm25_ranked] += BM25_WEIGHT * rrf_points
    fused[semantic_ranked] += SEMANTIC_WEIGHT * rrf_points
    fused_ranked = np.argsort(-fused, kind="stable")

    # chunks containing the query verbatim go first (error messages, string literals), fused order kept within each group
    needle = query.lower()
    exact = [i for i in fused_ranked if needle in documents[i].lower()]
    exact_set = set(exact)
    return exact + [i for i in fused_ranked if i not in exact_set]

def hybrid_search(query: str, top_k: int = 3):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="code_semantic_index")

    all_data = collection.get(include=["metadatas","documents","embeddings"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    embeddings = all_data["embeddings"]

    if not documents:
        print("[-] Index is empty. Run 'index' command first.")
        return

    query_embedding = VOYAGE_CLIENT.embed([query], model=EMBEDDING_MODEL, input_type="query").embeddings[0]
    top_indices = rank_chunks(query, documents, embeddings, query_embedding)[:top_k]

    print(f"\n=== Hybrid Search Query Results for: '{query}' ===")

    for rank, idx in enumerate(top_indices):
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

