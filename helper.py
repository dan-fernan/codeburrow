import os
from pathlib import Path
from consts import IGNORED_DIRS, EXTS

def collect_file_names():
    files = []
    for dirpath, dirnames, filenames in os.walk("."):
      dirnames[:] = [dir for dir in dirnames if dir not in IGNORED_DIRS]
      for f in filenames:
         if Path(f).suffix in EXTS:
            files.append(Path(dirpath) / f)

    return files