"""
AI Text Analyzer
-----------------
Reads a document (.txt, .pdf, or .docx), runs it through a pretrained
spaCy NLP model (real AI/NLP, not rule-based) to extract named entities
and word-level linguistic stats, and separately computes the frequency
of a target set of letters {a, w, e, r} in the raw text.

Outputs:
  1. letter_counts.csv   -> letter, count               (as requested)
  2. entities.csv        -> entity_text, entity_label    (AI/NLP output)
  3. word_stats.csv      -> word, lemma, pos, count      (AI/NLP output)

Usage:
    python3 main.py --input sample.txt
    python3 main.py --input report.pdf --output letters.csv
    python3 main.py --input essay.docx --entities-output ents.csv --word-output words.csv
"""

import argparse
import csv
import os
import sys
from collections import Counter

TARGET_LETTERS = ["a", "w", "e", "r"]


# ---------- File reading ----------

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_pdf(path):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def extract_text_from_docx(path):
    import docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .txt, .pdf, or .docx")


# ---------- Letter frequency (as requested) ----------

def analyze_letters(text):
    text_lower = text.lower()
    counts = Counter(ch for ch in text_lower if ch in TARGET_LETTERS)
    return {letter: counts.get(letter, 0) for letter in TARGET_LETTERS}


def write_letter_csv(letter_counts, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "count"])
        for letter in TARGET_LETTERS:
            writer.writerow([letter, letter_counts[letter]])


# ---------- Real AI/NLP analysis using spaCy ----------

def load_nlp_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print(
            "spaCy model 'en_core_web_sm' not found.\n"
            "Install it with: python3 -m spacy download en_core_web_sm"
        )
        sys.exit(1)


def analyze_with_ai(text, nlp):
    # spaCy truncates very long docs by default; raise the limit for big files
    nlp.max_length = max(nlp.max_length, len(text) + 1000)
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    word_counter = Counter()
    word_info = {}
    for token in doc:
        if token.is_alpha:
            key = token.text.lower()
            word_counter[key] += 1
            word_info[key] = (token.lemma_.lower(), token.pos_)

    return entities, word_counter, word_info


def write_entities_csv(entities, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_text", "entity_label"])
        for text, label in entities:
            writer.writerow([text, label])


def write_word_stats_csv(word_counter, word_info, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "lemma", "pos", "count"])
        for word, count in word_counter.most_common():
            lemma, pos = word_info[word]
            writer.writerow([word, lemma, pos, count])


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="AI-powered text analyzer: NLP entity/word stats + target letter frequency."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input file (.txt, .pdf, .docx)")
    parser.add_argument("--output", "-o", default="letter_counts.csv", help="Output CSV for letter counts")
    parser.add_argument("--entities-output", default="entities.csv", help="Output CSV for named entities (AI)")
    parser.add_argument("--word-output", default="word_stats.csv", help="Output CSV for word/POS stats (AI)")
    parser.add_argument("--skip-ai", action="store_true", help="Skip the spaCy AI analysis, only do letter counts")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found -> {args.input}")
        sys.exit(1)

    print(f"Reading file: {args.input}")
    text = extract_text(args.input)

    print("Counting target letters: a, w, e, r ...")
    letter_counts = analyze_letters(text)
    write_letter_csv(letter_counts, args.output)
    print(f"Saved -> {args.output}")
    for letter, count in letter_counts.items():
        print(f"  {letter}: {count}")

    if not args.skip_ai:
        print("\nRunning AI/NLP analysis (spaCy pretrained model)...")
        nlp = load_nlp_model()
        entities, word_counter, word_info = analyze_with_ai(text, nlp)

        write_entities_csv(entities, args.entities_output)
        print(f"Saved -> {args.entities_output} ({len(entities)} entities found)")

        write_word_stats_csv(word_counter, word_info, args.word_output)
        print(f"Saved -> {args.word_output} ({len(word_counter)} unique words)")

    print("\nDone.")


if __name__ == "__main__":
    main()
