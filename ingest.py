
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

from config import CONFIG
from compare_chunking import build_chunks



def enforce_embedding_limit(chunks, model):
    """
    Split oversized chunks using MiniLM token offsets.

    Unlike decoding WordPiece token IDs, slicing the
    original text preserves Markdown, capitalization,
    Python code, and punctuation.
    """
    tokenizer = model.tokenizer

    max_tokens = min(
        CONFIG["max_embedding_tokens"],
        model.max_seq_length,
    )

    special_tokens = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    available = max_tokens - special_tokens

    overlap = min(40, available // 4)
    step = available - overlap

    result = []

    for chunk in chunks:
        text = chunk["text"]

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )

        offsets = [
            (start, end)
            for start, end in encoded["offset_mapping"]
            if end > start
        ]

        if len(offsets) <= available:
            result.append(chunk)
            continue

        for start in range(0, len(offsets), step):
            window = offsets[start:start + available]

            if not window:
                break

            char_start = window[0][0]
            char_end = window[-1][1]

            part_text = text[char_start:char_end].strip()

            if part_text:
                result.append({
                    **chunk,
                    "text": part_text,
                    "tokens": len(
                        tokenizer.encode(
                            part_text,
                            add_special_tokens=True,
                        )
                    ),
                })

            if start + available >= len(offsets):
                break

    return result



def main():
    print("Loading embedding model...")
    model = SentenceTransformer(CONFIG["embedding_model"])

    print("Preparing chunks...")
    chunks = build_chunks(
        CONFIG["chunk_size"],
        CONFIG["chunk_overlap"],
    )

    original_count = len(chunks)

    chunks = enforce_embedding_limit(chunks, model)

    print(f"Chunks before embedding limit: {original_count}")
    print(f"Chunks after embedding limit: {len(chunks)}")

    # MiniLM has a limited input length.
    # Check actual tokenizer lengths, including special tokens.
    token_lengths = [
        len(
            model.tokenizer.encode(
                chunk["text"],
                add_special_tokens=True,
            )
        )
        for chunk in chunks
    ]

    print(f"Chunks prepared: {len(chunks)}")
    print(f"Maximum embedding tokens: {max(token_lengths)}")
    print(f"Model token limit: {model.max_seq_length}")

    if max(token_lengths) > model.max_seq_length:
        raise RuntimeError(
            "Some chunks exceed the embedding model token limit. "
            "Reduce chunk_size before indexing."
        )

    client = chromadb.PersistentClient(
        path=CONFIG["chroma_dir"]
    )

    # Recreate collection so repeated ingestion does not
    # leave stale chunks from previous configurations.
    try:
        client.delete_collection(CONFIG["collection_name"])
    except chromadb.errors.NotFoundError:
        pass

    collection = client.create_collection(
        name=CONFIG["collection_name"],
        configuration={
            "hnsw": {
                "space": CONFIG["similarity_metric"]
            }
        },
        metadata={
            "embedding_model": CONFIG["embedding_model"],
            "chunk_size": CONFIG["chunk_size"],
            "chunk_overlap": CONFIG["chunk_overlap"],
        },
    )

    print("Embedding and indexing chunks...")

    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = []
        metadatas = []

        for index, chunk in enumerate(batch, start=start):
            chunk_id = hashlib.sha256(
                f"{chunk['source']}:{chunk['heading']}:{index}".encode()
            ).hexdigest()[:16]

            ids.append(chunk_id)

            metadatas.append({
                "source": chunk["source"],
                "heading": chunk["heading"],
                "chunk_index": index,
                "tokens": chunk["tokens"],
                "source_url": (
                    "https://github.com/fastapi/fastapi/blob/"
                    "master/docs/en/docs/"
                    f"{chunk['source']}"
                ),
            })

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"Indexed {min(start + batch_size, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    print("\n=== INDEXING COMPLETE ===")
    print(f"Collection: {CONFIG['collection_name']}")
    print(f"Indexed chunks: {collection.count()}")
    print(f"Storage: {Path(CONFIG['chroma_dir']).resolve()}")


if __name__ == "__main__":
    main()
