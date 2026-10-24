
from pathlib import Path

from markdown_parser import (
    INCLUDE_PATTERN,
    resolve_markdown,
)

CORPUS_DIR = Path("data/corpus")
RAW_DOCS_DIR = Path("data/raw/docs")


def main():
    documents = sorted(CORPUS_DIR.rglob("*.md"))

    total_references = 0
    unresolved = 0
    total_original_chars = 0
    total_resolved_chars = 0

    for document in documents:
        relative_path = document.relative_to(CORPUS_DIR)
        original_path = RAW_DOCS_DIR / relative_path

        markdown = document.read_text(encoding="utf-8")

        references = len(INCLUDE_PATTERN.findall(markdown))
        total_references += references

        resolved = resolve_markdown(
            markdown,
            original_path,
        )

        remaining = len(INCLUDE_PATTERN.findall(resolved))
        unresolved += remaining

        total_original_chars += len(markdown)
        total_resolved_chars += len(resolved)

        print(
            f"{relative_path}: "
            f"{references} references resolved"
        )

    print("\n=== PARSER TEST ===")
    print(f"Documents: {len(documents)}")
    print(f"References found: {total_references}")
    print(f"Unresolved references: {unresolved}")
    print(f"Original characters: {total_original_chars:,}")
    print(f"Resolved characters: {total_resolved_chars:,}")


if __name__ == "__main__":
    main()
