import argparse
import re
from pathlib import Path


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9+#\.]*", text)


def normalize_token(token: str) -> str:
    token = token.strip()
    normalized = token.lower()
    replacements = {
        "c++": "c",
        "cpp": "c",
        "c#": "c sharp",
        "f#": "f sharp",
    }
    return replacements.get(normalized, normalized)


def collect_words(args) -> list[str]:
    words: set[str] = set()
    for item in args.word:
        normalized = normalize_token(item)
        for token in tokenize_text(normalized):
            words.add(token)

    for path in args.text_file:
        text = Path(path).read_text(encoding="utf-8")
        for token in tokenize_text(text):
            normalized = normalize_token(token)
            for piece in tokenize_text(normalized):
                words.add(piece)

    return sorted(words)


def ensure_nltk_data() -> None:
    import nltk

    for package in [
        "averaged_perceptron_tagger_eng",
        "cmudict",
    ]:
        try:
            nltk.data.find(package)
        except LookupError:
            nltk.download(package, quiet=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a custom G2P lexicon using g2p-en.")
    parser.add_argument("--word", action="append", default=[], help="Word or term to add. Can be repeated.")
    parser.add_argument("--text-file", action="append", default=[], help="Text file to scan for candidate words.")
    parser.add_argument("--output", required=True, help="Output lexicon file path.")
    args = parser.parse_args()

    ensure_nltk_data()
    from g2p_en import G2p

    words = collect_words(args)
    if not words:
        raise SystemExit("No words were provided. Use --word or --text-file.")

    g2p = G2p()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for word in words:
        phones = [item for item in g2p(word) if item != " "]
        if not phones:
          continue
        lines.append(f"{word.upper()}\t{' '.join(phones)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {output_path}")


if __name__ == "__main__":
    main()
