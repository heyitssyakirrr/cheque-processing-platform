"""Build the uat-008 branch: main with the Cheque_Processing. prefix on every project import.

On SVDWH008 the project sits in /pythonITD/dwh_312/Cheque_Processing, next to TransactionODD and
FileReader, and is imported as a package from dwh_312, so its imports read
"from Cheque_Processing.app.settings import ...". main keeps the plain imports
("from app.settings import ..."). This tool makes the prefixed copy, so files can be copied from
the uat-008 branch to SVDWH008 as they are:

    python tools/build_uat_branch.py            build from the current commit, check, push uat-008
    python tools/build_uat_branch.py --no-push  build and check only

Before committing it checks the copy the way SVDWH008 lays it out: from a folder that holds
Cheque_Processing, every Cheque_Processing.app import must point at a file that exists, every .py
must compile, and no plain "app." import may be left.
uat-008 is rebuilt from main each time (force-pushed); never edit it by hand.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PREFIX = "Cheque_Processing"
BRANCH = "uat-008"
PROJECT = Path(__file__).resolve().parents[1]
SKIP = {".venv", "__pycache__", ".git", "data", "tools"}

_RULES = [
    (re.compile(r"^(\s*)from app(?=[. ])", re.MULTILINE), rf"\1from {PREFIX}.app"),
    (re.compile(r"^(\s*)import app(?=[.\s]|$)", re.MULTILINE), rf"\1import {PREFIX}.app"),
    (re.compile(r'"app\.main:app"'), f'"{PREFIX}.app.main:app"'),
]
# run.py run as a file (python Cheque_Processing/run.py from dwh_312) only has Cheque_Processing on
# sys.path; put the folder above it there too so the prefixed imports are found. (run.py already
# imports sys and Path above this point.)
_RUN_PY_PATH = "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # uat-008: dwh_312, for Cheque_Processing.*\n\n"


def add_prefix(text: str) -> str:
    """The same source with the Cheque_Processing. prefix; running it twice changes nothing more."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def git(*args: str, cwd: Path = PROJECT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if not SKIP.intersection(path.relative_to(root).parts):
            yield path


def prefix_tree(root: Path) -> list[str]:
    changed = []
    for path in python_files(root):
        text = path.read_text(encoding="utf-8")
        new = add_prefix(text)
        if path.relative_to(root).as_posix() == "run.py" and "uat-008: dwh_312" not in new:
            anchor = f"\nfrom {PREFIX}.app.settings import settings"
            if anchor not in new:
                raise SystemExit("run.py: could not find where to add the sys.path line")
            new = new.replace(anchor, "\n" + _RUN_PY_PATH + anchor.lstrip("\n"), 1)
        if new != text:
            path.write_text(new, encoding="utf-8", newline="")
            changed.append(path.relative_to(root).as_posix())
    return changed


def check_tree(tree: Path) -> None:
    """From <tmp>/Cheque_Processing, as on SVDWH008: compile, and resolve every project import."""
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / PREFIX
        shutil.copytree(tree, package, ignore=shutil.ignore_patterns(".git", "__pycache__", "data", ".venv"))
        for path in python_files(package):
            name = path.relative_to(package).as_posix()
            try:
                module = ast.parse(path.read_text(encoding="utf-8"), filename=name)
            except SyntaxError as error:
                problems.append(f"{name}: {error}")
                continue
            for node in ast.walk(module):
                targets = []
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    targets = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                    base = [node.module]
                elif isinstance(node, ast.Import):
                    base = targets = [a.name for a in node.names]
                else:
                    continue
                for target in base:
                    if target == "app" or target.startswith("app."):
                        problems.append(f"{name}:{node.lineno}: plain import {target}")
                    elif target.startswith(f"{PREFIX}."):
                        candidates = {target} | {t for t in targets if t.startswith(target)}
                        if not any(_resolves(Path(tmp), t) for t in candidates):
                            problems.append(f"{name}:{node.lineno}: {target} does not exist")
    if problems:
        print("\n".join(f"  {p}" for p in problems))
        raise SystemExit(f"The prefixed copy has {len(problems)} problem(s); {BRANCH} was not built")
    print("  Checked: every file compiles and every Cheque_Processing.app import resolves")


def _resolves(root: Path, dotted: str) -> bool:
    path = root.joinpath(*dotted.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-push", action="store_true", help="build and check only")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        raise SystemExit("Commit or stash your changes first: uat-008 is built from the last commit")
    source = git("rev-parse", "--short", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        git("worktree", "add", "--force", "-B", BRANCH, str(tree), "HEAD")
        try:
            changed = prefix_tree(tree)
            print(f"Prefixed {len(changed)} file(s) from main {source}: {', '.join(changed)}")
            check_tree(tree)
            git("add", "-A", cwd=tree)
            git("commit", "-q", "-m", f"uat-008: {PREFIX}. prefix on the imports of main {source}\n\n"
                "Built by tools/build_uat_branch.py; never edit this branch by hand.", cwd=tree)
            if not args.no_push:
                git("push", "-q", "-f", "origin", BRANCH, cwd=tree)
                print(f"Pushed {BRANCH}: {git('rev-parse', '--short', 'HEAD', cwd=tree)}")
        finally:
            git("worktree", "remove", "--force", str(tree))
    return 0


if __name__ == "__main__":
    sys.exit(main())
