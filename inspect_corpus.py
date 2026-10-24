from pathlib import Path
import re
from collections import Counter

CORPUS_DIR = Path("data/corpus")

INCLUDE_PATTERN = re.compile(r"\{![^}]+!\}")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
CODE_FENCE_PATTERN = re.compile(r"^```", re.MULTILINE)


def main():
    files = sorted(CORPUS_DIR.rglob("*.md"))

    if not files:
        raise RuntimeError("No Markdown files found in data/corpus")

    total_chars = 0
    total_headings = 0
    total_includes = 0
    total_fences = 0
    files_with_includes = []
    include_examples = []
    sizes = []

    for path in files:
        text = path.read_text(encoding="utf-8")

        headings = HEADING_PATTERN.findall(text)
        includes = INCLUDE_PATTERN.findall(text)
        fences = CODE_FENCE_PATTERN.findall(text)

        total_chars += len(text)
        total_headings += len(headings)
        total_includes += len(includes)
        total_fences += len(fences)

        sizes.append((len(text), path.relative_to(CORPUS_DIR)))

        if includes:
            files_with_includes.append(str(path.relative_to(CORPUS_DIR)))
            include_examples.extend(includes[:2])

    print("=== CORPUS INSPECTION ===")
    print(f"Documents: {len(files)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Markdown headings: {total_headings}")
    print(f"Include directives: {total_includes}")
    print(f"Files containing includes: {len(files_with_includes)}")
    print(f"Code fences: {total_fences}")

    print("\n=== LARGEST DOCUMENTS ===")
    for size, path in sorted(sizes, reverse=True)[:5]:
        print(f"{size:>8,} chars  {path}")

    print("\n=== FILES WITH INCLUDE DIRECTIVES ===")
    for path in files_with_includes[:10]:
        print(f"- {path}")

    print("\n=== EXAMPLE INCLUDE DIRECTIVES ===")
    for example in include_examples[:5]:
        print(example)


if __name__ == "__main__":
    main()
