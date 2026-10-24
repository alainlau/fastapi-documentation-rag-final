
from pathlib import Path
import shutil

RAW_DIR = Path("data/raw/docs")
CORPUS_DIR = Path("data/corpus")

# Initial document selection, based on our Week 1 proposal.
# Paths are relative to the official FastAPI English docs directory.
SELECTED_FILES = [
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/body.md",
    "tutorial/body-fields.md",
    "tutorial/body-multiple-params.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/path-params-numeric-validations.md",
    "tutorial/response-model.md",
    "tutorial/response-status-code.md",
    "tutorial/handling-errors.md",
    "tutorial/dependencies/index.md",
    "tutorial/dependencies/classes-as-dependencies.md",
    "tutorial/dependencies/sub-dependencies.md",
    "tutorial/dependencies/dependencies-in-path-operation-decorators.md",
    "tutorial/security/first-steps.md",
    "tutorial/security/get-current-user.md",
    "tutorial/security/simple-oauth2.md",
    "tutorial/security/oauth2-jwt.md",
    "tutorial/sql-databases.md",
]

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing documentation directory: {RAW_DIR}")

    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)

    copied = []
    missing = []

    for relative_path in SELECTED_FILES:
        source = RAW_DIR / relative_path
        destination = CORPUS_DIR / relative_path

        if not source.is_file():
            missing.append(relative_path)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative_path)

    print(f"Selected documents: {len(SELECTED_FILES)}")
    print(f"Copied documents:   {len(copied)}")
    print(f"Missing documents:  {len(missing)}")

    if missing:
        print("\nMissing files:")
        for path in missing:
            print(f"  - {path}")

    print("\nCorpus location:", CORPUS_DIR.resolve())


if __name__ == "__main__":
    main()
