# codeburrow

Hybrid semantic + keyword code search with AST-aware chunking. codeburrow indexes a codebase by parsing it into function/class-level chunks with [tree-sitter](https://tree-sitter.github.io/tree-sitter/), embeds those chunks with [Voyage AI](https://www.voyageai.com/), and searches over them by fusing keyword (BM25) and semantic (vector) rankings — so `codeburrow search "auth middleware"` finds relevant code whether or not it contains the literal words "auth" or "middleware".

## Features

- **Hybrid retrieval** — combines BM25 keyword scoring with Voyage AI semantic embeddings via Reciprocal Rank Fusion, so results aren't limited to exact keyword matches or purely semantic guesses.
- **AST-aware chunking** — uses tree-sitter to parse source files into function/class-level chunks (rather than arbitrary line splits) for more coherent, accurately-scoped embeddings.
- **Multi-language support** — Python, JavaScript, JSX, TypeScript, and TSX out of the box.
- **Git-aware incremental indexing** — after the initial index, only changed files are re-embedded on each run (via `git status`), including correct handling of renamed and deleted files, so the index never accumulates stale entries.
- **Self-installing git hook** — `codeburrow postcommit-install` wires up a `post-commit` hook that automatically re-indexes after every commit, detecting and safely extending whatever interpreter (shell, Python, Node, Ruby) an existing hook already uses.

## Requirements

- Python >= 3.9
- A [Voyage AI](https://www.voyageai.com/) API key (embeddings are generated via Voyage's hosted API, not run locally)

## Installation

```sh
git clone <this-repo-url>
cd codeburrow
pip install -e .
```

This installs `codeburrow` as a command on your `PATH`. Editable install (`-e`) means future `git pull`s pick up code changes immediately with no reinstall — you only need to re-run `pipx install -e .` if the project's dependencies themselves change.

## Setup

Set your Voyage AI API key in your shell profile so it's available in every session:

```sh
# ~/.zshrc or ~/.bashrc
export VOYAGE_API_KEY="your-key-here"
```

Then reload your shell (`source ~/.zshrc`) or open a new terminal.

## Usage

Run from inside any project you want to index and search — codeburrow operates on whatever directory you invoke it from, not on itself.

```sh
cd /path/to/your/project

# Build the initial index (full scan on first run)
codeburrow index

# Force a full re-index instead of an incremental one
codeburrow index --force

# Search the index
codeburrow search "auth middleware"

# Optional: auto-reindex on every future commit
codeburrow postcommit-install
```

Running `codeburrow postcommit-install` also adds `.codeburrow_db/` (the local vector index) to your project's `.gitignore`, so the index itself is never committed.

## How it works

1. **Chunking** — `collect_file_names()` walks the repo (skipping common noise directories like `node_modules`, `.git`, `dist`, etc.) and each supported file is parsed with tree-sitter into function/class-level chunks.
2. **Embedding** — each chunk is embedded via Voyage AI's `voyage-code-3` model (code-specialized) and stored in a local [ChromaDB](https://www.trychroma.com/) collection.
3. **Search** — a query is scored two ways: BM25 keyword matching against chunk text, and cosine similarity against the query's embedding. Both rankings are merged via Reciprocal Rank Fusion, and the top results are printed with `file:line` and a signature preview.
4. **Incremental updates** — after the first index, `codeburrow index` reads `git status --porcelain` and only re-embeds files that changed, correctly deleting index entries for removed/renamed files rather than leaving them stale.

## Supported file types

`.py`, `.js`, `.jsx`, `.ts`, `.tsx`