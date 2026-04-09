#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
from typing import NamedTuple


ARCHIVE_FILE = "all.txt"
BEGIN = "===BEGIN_FILE==="
END = "===END_FILE==="

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
}

INCLUDED_FILE_EXTENSIONS = {
    ".py",
    ".md",
    ".ini",
    ".cfg",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".js",
    ".css",
}

INCLUDED_FILE_NAMES = {}


class GitignoreRule(NamedTuple):
    pattern: str
    negation: bool
    anchored: bool
    dir_only: bool


class GitignoreParser:
    """Simple .gitignore pattern parser and matcher."""

    def __init__(self, root: Path):
        self.root = root
        self.rules: list[GitignoreRule] = []
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        gitignore_path = self.root / ".gitignore"
        if not gitignore_path.exists():
            return

        for line in gitignore_path.read_text(encoding="utf-8").splitlines():
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue

            negation = line.startswith("!")
            if negation:
                line = line[1:]

            anchored = line.startswith("/")
            if anchored:
                line = line[1:]

            dir_only = line.endswith("/")
            if dir_only:
                line = line[:-1]

            self.rules.append(GitignoreRule(
                pattern=line,
                negation=negation,
                anchored=anchored,
                dir_only=dir_only,
            ))

    def _match_rule(self, rule: GitignoreRule, rel_path: Path, is_dir: bool) -> bool | None:
        """Returns True if matches, False if not, None if rule doesn't apply."""
        rel_str = str(rel_path).replace("\\", "/")
        parts = rel_str.split("/")

        if rule.dir_only:
            if not is_dir:
                # Check if file is inside ignored directory
                for i, part in enumerate(parts[:-1]):  # Exclude the file itself
                    if fnmatch.fnmatch(part, rule.pattern):
                        return True
            return None

        if rule.anchored:
            if fnmatch.fnmatch(rel_str, rule.pattern):
                return True
            if fnmatch.fnmatch(parts[0], rule.pattern) and not is_dir:
                return True
        else:
            for i, part in enumerate(parts):
                rest = "/".join(parts[i:])
                if fnmatch.fnmatch(part, rule.pattern):
                    return True
                if fnmatch.fnmatch(rest, rule.pattern):
                    return True

        return None

    def is_ignored(self, path: Path) -> bool:
        """Check if a path is ignored by .gitignore rules."""
        try:
            rel_path = path.relative_to(self.root)
        except ValueError:
            return False

        is_dir = path.is_dir()
        ignored = False

        for rule in self.rules:
            result = self._match_rule(rule, rel_path, is_dir)
            if result is True:
                ignored = not rule.negation

        return ignored


def should_skip_path(path: Path, root: Path, gitignore: GitignoreParser) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    if gitignore.is_ignored(path):
        return True
    return False


def should_include_path(path: Path) -> bool:
    if path.name in INCLUDED_FILE_NAMES:
        return True
    return path.suffix.lower() in INCLUDED_FILE_EXTENSIONS


def collect_files(root: Path, *, archive_path: Path | None = None) -> list[Path]:
    files: list[Path] = []
    archive_resolved = archive_path.resolve() if archive_path is not None else None
    default_archive_resolved = (root / ARCHIVE_FILE).resolve()
    gitignore = GitignoreParser(root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip_path(path, root, gitignore):
            continue
        if not should_include_path(path):
            continue
        resolved_path = path.resolve()
        if resolved_path == default_archive_resolved:
            continue
        if archive_resolved is not None and resolved_path == archive_resolved:
            continue
        files.append(path)

    return sorted(set(files), key=lambda p: str(p.relative_to(root)))


def pack(root: Path, output_file: Path) -> None:
    files = collect_files(root, archive_path=output_file)
    with output_file.open("w", encoding="utf-8", newline="") as out:
        for path in files:
            rel = path.relative_to(root).as_posix()
            content = path.read_text(encoding="utf-8")
            out.write(f"{BEGIN}\t{rel}\t{len(content)}\n")
            out.write(content)
            out.write("\n")
            out.write(f"{END}\n")

    print(f"Packed {len(files)} files into {output_file}")


def unpack(root: Path, input_file: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Archive file not found: {input_file}")

    restored = 0

    with input_file.open("r", encoding="utf-8") as src:
        while True:
            header = src.readline()
            if not header:
                break
            if not header.startswith(f"{BEGIN}\t"):
                raise ValueError("Invalid archive format: malformed BEGIN header")

            payload = header.rstrip("\n").split("\t", maxsplit=2)
            if len(payload) != 3:
                raise ValueError("Invalid archive format: malformed BEGIN payload")
            _, rel_path, raw_len = payload
            content_len = int(raw_len)

            content = src.read(content_len)
            if len(content) != content_len:
                raise ValueError("Invalid archive format: truncated file content")

            separator = src.read(1)
            if separator != "\n":
                raise ValueError("Invalid archive format: missing content separator")

            end_line = src.readline().rstrip("\n")
            if end_line != END:
                raise ValueError("Invalid archive format: missing END marker")

            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            restored += 1

    print(f"Restored {restored} files from {input_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack project text sources/config/assets (Python, HTML/JS/CSS, JSON/TOML,"
            " Markdown, WELLTRACK .inc, etc.) to all.txt and unpack back."
        )
    )
    parser.add_argument(
        "mode",
        choices=("pack", "unpack"),
        help="pack: create all.txt, unpack: restore files from all.txt",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help=f"Archive file path (default: <root>/{ARCHIVE_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    archive = (
        (root / ARCHIVE_FILE) if args.archive is None else Path(args.archive).resolve()
    )

    if args.mode == "pack":
        pack(root, archive)
    else:
        unpack(root, archive)


if __name__ == "__main__":
    main()
