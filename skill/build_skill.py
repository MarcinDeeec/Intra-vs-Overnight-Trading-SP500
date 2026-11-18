#!/usr/bin/env python3
"""Build and lint the Agensi / Agent-Skills package.

    python skill/build_skill.py            # writes dist/sp500-overnight-vs-intraday.zip

The single source of truth for the script is the repo root; it is copied into
scripts/ inside the ZIP. Pure standard library.
"""
import os
import re
import sys
import zipfile

NAME = "sp500-overnight-vs-intraday"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILL_DIR = os.path.join(HERE, NAME)
SCRIPT = os.path.join(ROOT, "sp500_overnight_intraday.py")
LICENSE = os.path.join(ROOT, "LICENSE")
OUT = os.path.join(ROOT, "dist", NAME + ".zip")

MAX_SCRIPT_BYTES = 20_000          # Agensi: keep each script's logic under ~20 KB
MAX_SKILL_LINES = 500
FORBIDDEN = [r"os\.system", r"shell\s*=\s*True", r"chmod\s+777", r"ignore (all )?previous instructions",
             r"do not tell the user", r"\bsecretly\b", r"curl[^\n]*\|\s*(ba)?sh", r"base64\.b64decode"]


def lint(files):
    problems = []
    skill = open(os.path.join(SKILL_DIR, "SKILL.md"), encoding="utf-8").read()
    m = re.match(r"^---\n(.*?)\n---\n", skill, re.S)
    if not m:
        return ["SKILL.md has no frontmatter"]
    fm = m.group(1)
    name = re.search(r"^name:\s*(.+)$", fm, re.M)
    desc = re.search(r"^description:\s*(.+)$", fm, re.M)
    if not name or name.group(1).strip() != NAME:
        problems.append("frontmatter name must equal the folder name: " + NAME)
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", NAME) or len(NAME) > 64:
        problems.append("name must be lowercase kebab-case, <= 64 chars")
    if not desc or not desc.group(1).strip():
        problems.append("missing description")
    elif len(desc.group(1).strip()) > 1024:
        problems.append("description longer than 1024 characters")
    if skill.count("\n") > MAX_SKILL_LINES:
        problems.append("SKILL.md longer than 500 lines")
    if os.path.getsize(SCRIPT) > MAX_SCRIPT_BYTES:
        problems.append(f"script is {os.path.getsize(SCRIPT)} bytes (> {MAX_SCRIPT_BYTES})")
    for arc, src in files:
        if arc.endswith((".md", ".py")):
            text = open(src, encoding="utf-8").read()
            for pat in FORBIDDEN:
                if re.search(pat, text, re.I):
                    problems.append(f"{arc}: forbidden pattern /{pat}/")
    return problems


def main():
    files = []
    for dirpath, dirnames, filenames in os.walk(SKILL_DIR):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            src = os.path.join(dirpath, f)
            files.append((os.path.join(NAME, os.path.relpath(src, SKILL_DIR)), src))
    files.append((f"{NAME}/scripts/sp500_overnight_intraday.py", SCRIPT))
    files.append((f"{NAME}/LICENSE", LICENSE))

    problems = lint(files)
    if problems:
        print("Lint failed:\n  - " + "\n  - ".join(problems))
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in sorted(files):
            z.write(src, arc)
    print(f"OK  {os.path.relpath(OUT, ROOT)}  ({os.path.getsize(OUT)} bytes, {len(files)} files)")
    for arc, _ in sorted(files):
        print("   ", arc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
