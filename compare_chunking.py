
from pathlib import Path
from collections import Counter
import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from markdown_parser import resolve_markdown
from config import CONFIG

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
TOKENIZER = tiktoken.get_encoding("cl100k_base")

CONFIGURATIONS = {
    "A": {"chunk_size": 256, "chunk_overlap": 60},
    "B": {"chunk_size": 150, "chunk_overlap": 20},
}

def token_count(text):
    return len(TOKENIZER.encode(text))



def split_sections(markdown):
    """
    Split Markdown by headings while ignoring headings
    inside fenced code blocks.
    """
    sections = []
    heading_stack = []
    current_lines = []
    in_code_block = False
    fence_character = None
    fence_length = 0

    def flush():
        content = "\n".join(current_lines).strip()

        if content:
            sections.append({
                "heading": " > ".join(
                    heading for _, heading in heading_stack
                ) or "Introduction",
                "content": content,
            })

    for line in markdown.splitlines():
        stripped = line.lstrip()

        fence = re.match(r"^(`{3,}|~{3,})", stripped)

        if fence:
            marker = fence.group(1)
            character = marker[0]

            if not in_code_block:
                in_code_block = True
                fence_character = character
                fence_length = len(marker)

            elif (
                character == fence_character
                and len(marker) >= fence_length
            ):
                in_code_block = False
                fence_character = None
                fence_length = 0

        match = None

        if not in_code_block:
            match = HEADING_PATTERN.match(line)

        if match:
            flush()
            current_lines = []

            level = len(match.group(1))
            heading = match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading))

        current_lines.append(line)

    flush()
    return sections



def build_chunks(chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    corpus_dir = Path(CONFIG["corpus_dir"])
    raw_docs_dir = Path(CONFIG["raw_docs_dir"])
    chunks = []

    for path in sorted(corpus_dir.rglob("*.md")):
        relative_path = path.relative_to(corpus_dir)
        original_path = raw_docs_dir / relative_path

        markdown = resolve_markdown(
            path.read_text(encoding="utf-8"),
            original_path,
        )

        sections = split_sections(markdown)

        for section in sections:
            for content in splitter.split_text(section["content"]):
                chunks.append({
                    "source": relative_path.as_posix(),
                    "heading": section["heading"],
                    "text": content,
                    "tokens": token_count(content),
                })

    return chunks


def summarize(label, chunks):
    sizes = [chunk["tokens"] for chunk in chunks]

    print(f"\n=== CONFIGURATION {label} ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average tokens: {sum(sizes) / len(sizes):.1f}")
    print(f"Smallest chunk: {min(sizes)}")
    print(f"Largest chunk: {max(sizes)}")
    print(f"Chunks under 40 tokens: {sum(s < 40 for s in sizes)}")

    print("\nLargest sources by chunk count:")
    counts = Counter(chunk["source"] for chunk in chunks)

    for source, count in counts.most_common(5):
        print(f"  {source}: {count}")

def inspect_suspicious_heading():
    path = Path("data/corpus/tutorial/security/oauth2-jwt.md")

    markdown = resolve_markdown(
        path.read_text(encoding="utf-8"),
        Path("data/raw/docs/tutorial/security/oauth2-jwt.md"),
    )

    lines = markdown.splitlines()

    for index, line in enumerate(lines):
        if "openssl rand -hex 32" in line:
            print("\n=== SUSPICIOUS HEADING CONTEXT ===")

            start = max(0, index - 8)
            end = min(len(lines), index + 9)

            for i in range(start, end):
                print(f"{i + 1:>4}: {lines[i]!r}")


def main():
    #inspect_suspicious_heading()

    results = {}

    for label, settings in CONFIGURATIONS.items():
        chunks = build_chunks(**settings)
        results[label] = chunks
        summarize(label, chunks)

    print("\n=== SAMPLE CHUNKS: CONFIGURATION A ===")

    examples = [
        "tutorial/security/oauth2-jwt.md",
        "tutorial/dependencies/index.md",
    ]

    for source in examples:
        matches = [
            chunk for chunk in results["A"]
            if chunk["source"] == source
        ]

        if matches:
            example = max(matches, key=lambda c: c["tokens"])
            print(f"\nSource: {example['source']}")
            print(f"Heading: {example['heading']}")
            print(f"Tokens: {example['tokens']}")
            print(example["text"][:700])
            print("...")


if __name__ == "__main__":
    main()
