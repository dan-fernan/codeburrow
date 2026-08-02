import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser
from pathlib import Path

DB_PATH = Path("./.codeburrow_db")
EMBEDDING_MODEL = "voyage-code-3"
POSTCOMMIT_MARKER = "codeburrow-postcommit"

INTERPRETER_HOOK_LINES = {
    "sh": f"# {POSTCOMMIT_MARKER}\ncodeburrow index >/dev/null 2>&1 &\n",
    "bash": f"# {POSTCOMMIT_MARKER}\ncodeburrow index >/dev/null 2>&1 &\n",
    "zsh": f"# {POSTCOMMIT_MARKER}\ncodeburrow index >/dev/null 2>&1 &\n",
    "python": f"# {POSTCOMMIT_MARKER}\nimport subprocess; subprocess.Popen([\"codeburrow\", \"index\"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "python3": f"# {POSTCOMMIT_MARKER}\nimport subprocess; subprocess.Popen([\"codeburrow\", \"index\"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "node": f"// {POSTCOMMIT_MARKER}\nrequire(\"child_process\").spawn(\"codeburrow\", [\"index\"], {{ stdio: \"ignore\", detached: true }}).unref();\n",
    "ruby": f"# {POSTCOMMIT_MARKER}\nspawn(\"codeburrow index\", out: \"/dev/null\", err: \"/dev/null\")\n",
}

LANGUAGES = {
    ".py": Language(tspython.language()),
    ".js": Language(tsjavascript.language()),
    ".jsx": Language(tsjavascript.language()),
    ".ts": Language(tstypescript.language_typescript()),
    ".tsx": Language(tstypescript.language_tsx())
}

PARSERS = {suffix: Parser(lang) for suffix, lang in LANGUAGES.items()}


CHUNK_NODE_TYPES = {
    ".py": {"function_definition", "async_function_definition", "class_definition", "decorated_definition"},
    ".js": {"function_declaration", "class_declaration", "method_definition", "lexical_declaration", "export_statement"},
    ".jsx": {"function_declaration", "class_declaration", "method_definition", "lexical_declaration", "export_statement"},
    ".ts": {"function_declaration", "class_declaration", "method_definition", "lexical_declaration", "export_statement", "interface_declaration"},
    ".tsx": {"function_declaration", "class_declaration", "method_definition", "lexical_declaration", "export_statement", "interface_declaration"},
}

IGNORED_DIRS = {
    ".git", "venv", ".venv", "env", "node_modules", "__pycache__",
    ".codeburrow_db", "dist", "build", ".next", ".nuxt", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".turbo", ".parcel-cache",
}

EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx"
}