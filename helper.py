from pathlib import Path

def collect_file_names():
    return [Path(p) for p in Path(".").rglob("*.py") if ".git" not in p.parts and ".sgrep_db" not in p.parts and "venv" not in p.parts]