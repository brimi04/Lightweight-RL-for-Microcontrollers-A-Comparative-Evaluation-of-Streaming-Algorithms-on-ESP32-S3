"""
verify_manifest.py -- Check that every file of the repository matches MANIFEST_SHA256.csv.

Reports files that are missing, modified, or present but not listed. Read-only.
Usage (from the repository root):  python tools/verify_manifest.py
Exit status 0 only if every listed file is present and unchanged.
"""
import csv
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "MANIFEST_SHA256.csv")
# Not part of the archived snapshot (generated locally or by git).
IGNORED_DIRS = {".git", "_reproduction", "__pycache__", ".venv", ".venv-exp", ".venv-ana"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    listed, missing, modified = set(), [], []
    for r in rows:
        p = r["path"]
        listed.add(p)
        full = os.path.join(REPO, *p.split("/"))
        if not os.path.isfile(full):
            missing.append(p)
        elif sha256(full) != r["sha256"]:
            modified.append(p)
    unlisted = []
    for d, dirs, files in os.walk(REPO):
        dirs[:] = [x for x in dirs if x not in IGNORED_DIRS]
        for fn in files:
            rel = os.path.relpath(os.path.join(d, fn), REPO).replace(os.sep, "/")
            if rel not in listed and rel != "MANIFEST_SHA256.csv":
                unlisted.append(rel)
    print(f"{len(rows)} files listed; {len(missing)} missing; {len(modified)} modified; {len(unlisted)} not listed.")
    for name, lst in (("MISSING", missing), ("MODIFIED", modified), ("NOT LISTED", unlisted)):
        for p in lst[:50]:
            print(f"  {name}: {p}")
    if modified and any(p.endswith((".csv", ".txt", ".md", ".json", ".log")) for p in modified):
        print("\nHint: if text files differ, check that git did not convert line endings "
              "(the repository's .gitattributes disables conversion; see README).")
    sys.exit(0 if not (missing or modified) else 1)


if __name__ == "__main__":
    main()
