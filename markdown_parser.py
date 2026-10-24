
from pathlib import Path
import re


RAW_DOCS_SRC = Path("data/raw/docs_src")

# Example:
# {* ../../docs_src/first_steps/tutorial001_py310.py hl[3] *}
INCLUDE_PATTERN = re.compile(
    r"\{\*\s*(?P<path>\S+\.py)"
    r"(?P<options>[^{}]*?)\s*\*\}"
)

LINE_PATTERN = re.compile(r"\bln\[([^\]]+)\]")


def select_lines(content: str, specification: str) -> str:
    """
    Select one-based line numbers and inclusive ranges.

    Examples:
        ln[5]
        ln[5:10]
        ln[2,5:8,12]
    """
    lines = content.splitlines()
    selected = []

    for part in specification.split(","):
        part = part.strip()

        if ":" in part:
            start, end = map(int, part.split(":", 1))
            numbers = range(start, end + 1)
        else:
            numbers = [int(part)]

        for number in numbers:
            if not 1 <= number <= len(lines):
                raise ValueError(
                    f"Line {number} is outside source "
                    f"range 1-{len(lines)}"
                )

            selected.append(lines[number - 1])

    return "\n".join(selected)



def resolve_markdown(
    markdown: str,
    document_path: Path,
) -> str:
    """
    Replace FastAPI Python references with actual source code.
    Preserve existing code fences and wrap standalone includes.
    """

    def replace_include(match: re.Match) -> str:
        relative_path = match.group("path")
        options = match.group("options")

        source_path = (
            RAW_DOCS_SRC.parent
            / "docs"
            / "tutorial"
            / relative_path
        ).resolve()

        allowed_root = RAW_DOCS_SRC.resolve()

        if not source_path.is_relative_to(allowed_root):
            raise ValueError(
                f"Include outside docs_src: {relative_path}"
            )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing FastAPI example: {source_path}"
            )

        content = source_path.read_text(encoding="utf-8")

        line_match = LINE_PATTERN.search(options)

        if line_match:
            content = select_lines(
                content,
                line_match.group(1),
            )

        # Determine whether the include is inside a Markdown fence.
        preceding_text = markdown[:match.start()]
        fence_matches = re.findall(
            r"^\s*(`{3,}|~{3,})",
            preceding_text,
            flags=re.MULTILINE,
        )

        inside_fence = len(fence_matches) % 2 == 1

        if inside_fence:
            return content.rstrip("\n")

        return f"\n```python\n{content.rstrip()}\n```\n"

    return INCLUDE_PATTERN.sub(replace_include, markdown)



if __name__ == "__main__":
    sample = (
        "```python\n"
        "{* ../../docs_src/first_steps/tutorial001_py310.py *}\n"
        "```"
    )

    document = Path(
        "data/raw/docs/tutorial/first-steps.md"
    )

    print(resolve_markdown(sample, document))
