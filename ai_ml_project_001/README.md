# AI Text Analyzer 🧠📄

A small AI/NLP project that reads a document (`.txt`, `.pdf`, or `.docx`) and produces
structured CSV outputs using a **pretrained spaCy language model** (real neural NLP —
named entity recognition + part-of-speech tagging), alongside a targeted letter-frequency
count.

100% free and local — no API keys, no paid services, no internet needed after setup.

## What it does

Given any input file, it produces up to three CSV files:

| File | Contents | Powered by |
|---|---|---|
| `letter_counts.csv` | Frequency of the letters `a`, `w`, `e`, `r` (lowercase) in the text | Plain Python |
| `entities.csv` | Named entities detected in the text (people, places, orgs, dates, etc.) | spaCy AI model |
| `word_stats.csv` | Every word with its lemma, part-of-speech tag, and frequency | spaCy AI model |

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd ai-letter-analyzer

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy AI model (one-time, ~12MB, free)
python3 -m spacy download en_core_web_sm
```

## Usage

```bash
python3 main.py --input sample.txt
```

This will create:
- `letter_counts.csv`
- `entities.csv`
- `word_stats.csv`

### Custom file / output paths

```bash
python3 main.py --input mydoc.pdf --output letters.csv --entities-output ents.csv --word-output words.csv
```

### Supported input types
- `.txt`
- `.pdf`
- `.docx`

### Skip the AI step (letter count only, faster)

```bash
python3 main.py --input sample.txt --skip-ai
```

## Example output (`letter_counts.csv`)

```
letter,count
a,12
w,5
e,15
r,10
```

## Example output (`entities.csv`)

```
entity_text,entity_label
Warren,PERSON
Sarah,PERSON
Australia,GPE
```

## Project structure

```
ai-letter-analyzer/
├── main.py
├── requirements.txt
├── sample.txt
└── README.md
```

## Why this counts as an AI project

The letter-frequency counting is plain Python logic — but the entity extraction and
part-of-speech tagging are produced by `en_core_web_sm`, a pretrained neural NLP model
shipped with spaCy. It performs tokenization, POS tagging, and named entity recognition
using a trained neural network, not hand-written rules.

## License

MIT — free to use, modify, and share.
