import re

def fixed_size_chunks(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks


def sentence_chunks(
    text: str,
    max_sentences: int = 3,
) -> list[str]:
    if max_sentences <= 0:
        raise ValueError("max_sentences must be greater than 0")

    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return [
        " ".join(sentences[i:i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]


def paragraph_chunks(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    return chunks or [text]
