"""
Prepare the data for the website version (run once, after fetch_data.py).

Reads  knowledge/*.txt  and  data/kepler.csv  and creates:
  - knowledge.json : the articles, cut into small passages
  - kepler.csv     : a copy of the NASA table (the site trains its model from it)

Uses only Python's standard library, so it runs on any computer.

Run:  python build_knowledge.py
"""
import glob
import json
import os
import re
import shutil

CHUNK_WORDS = 150
OVERLAP_WORDS = 30
MAX_CHARS_PER_ARTICLE = 15000  # keep the site fast: use the start of each article

# Wikipedia sections that are not useful as answers
STOP_HEADINGS = {
    "see also", "references", "notes", "external links", "further reading",
    "footnotes", "bibliography", "sources", "citations",
}


def clean_article(text):
    """Drop reference-style sections and turn '== Heading ==' lines into plain text."""
    kept = []
    for line in text.split("\n"):
        m = re.match(r"^=+\s*(.*?)\s*=+$", line.strip())
        if m:
            if m.group(1).lower() in STOP_HEADINGS:
                break
            kept.append(m.group(1) + ".")
        else:
            kept.append(line)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        piece = " ".join(words[start:start + size]).strip()
        if len(piece) > 80:
            chunks.append(piece)
        start += size - overlap
    return chunks


def main():
    files = sorted(glob.glob("knowledge/*.txt"))
    if not files:
        print("No files in knowledge/. Run fetch_data.py first.")
        return

    passages = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        title, _, body = raw.partition("\n")
        title = title.strip()
        body = clean_article(body)[:MAX_CHARS_PER_ARTICLE]
        for c in chunk_words(body):
            passages.append({"t": title, "x": c})
        print(f"  {title}: {len([p for p in passages if p['t'] == title])} passages")

    with open("knowledge.json", "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = os.path.getsize("knowledge.json") / 1024
    print(f"\nknowledge.json: {len(passages)} passages ({size_kb:.0f} KB)")

    if os.path.exists("data/kepler.csv"):
        shutil.copyfile("data/kepler.csv", "kepler.csv")
        print("kepler.csv: copied")
    else:
        print("data/kepler.csv not found. Run fetch_data.py first.")


if __name__ == "__main__":
    main()
