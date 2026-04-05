import os
from typing import List


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def list_source_files(root: str, exts: List[str] = None) -> List[str]:
    exts = exts or [".md", ".txt", ".py", ".rst", ".json"]
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            name_low = fn.lower()
            if any(name_low.endswith(e) for e in exts):
                files.append(os.path.join(dirpath, fn))
                continue
            # include common README/LICENSE files without extension
            if name_low in ("readme", "readme.md", "readme.txt", "license", "license.md"):
                files.append(os.path.join(dirpath, fn))
                continue
    return files
