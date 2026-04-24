from pathlib import Path
import spacy



INPUT_DIR = Path("data/ba_artikel_txt")
OUTPUT_DIR = Path("data/ba_artikel_processed/NOUN-ADJ-VERB-PROP")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load("de_dep_news_trf")

def preprocess(text: str) -> str:
    doc = nlp(text)

    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.pos_ in {"NOUN", "ADJ", "VERB", "PROP"}
                
    ]

    return " ".join(tokens)

count = 0

for path in INPUT_DIR.rglob("*.txt"):
    raw = path.read_text(encoding="utf-8")
    processed = preprocess(raw)

    out_path = OUTPUT_DIR / path.name
    out_path.write_text(processed, encoding="utf-8")

    count += 1
    print(f"Wrote {out_path}")

print(f"Done. Processed {count} files.")