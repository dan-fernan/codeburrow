
import sys
import subprocess
import chromadb
import ollama
from pathlib import Path
from typing import List, Dict, Any
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PYTHON_LANGUAGE = Language(tspython.language())
# tspython.language() returns a pointer to the compiled C def'n for python
# Language() takes that pointer and turns it into a python object, exposing it to high level methods

parser = Parser(PYTHON_LANGUAGE)
# Instantiates the engine that executes the parsing process

DB_PATH = Path("./.sgreb_db")
EMBEDDING_MODEL = "nomic-embed-text"

def ast_parse_chunk(file_path: Path) -> List:
    if not file_path.exists():
        return []
    
    code_bytes = file_path.read_bytes()
    tree = parser.parse(code_bytes)

    return tree

print(ast_parse_chunk("./test.py"))
