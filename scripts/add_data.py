import re
from pathlib import Path
from typing import List, Dict

import pandas as pd

# Hindi NLP
import stanza as download_stanza
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# English NLP
import spacy

# =========================
# Config & initializations
# =========================

DEV_START, DEV_END = "\u0900", "\u097F"

def is_dev_char(ch: str) -> bool:
    return DEV_START <= ch <= DEV_END

def is_devanagari(s: str) -> bool:
    return any(is_dev_char(c) for c in s)

# Remove <Hindi>…</Hindi> tags (any case/spacing/attributes)
TAG_RE = re.compile(r'<\s*/?\s*Hindi\b[^>]*>', flags=re.IGNORECASE)

def remove_hindi_tags(text: str) -> str:
    text = TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

# =========== TOKENIZATION RULE ===========
# Keep ONLY letter sequences:
# - Devanagari spans (will filter non-letters post transliteration)
# - English A–Z sequences (single letters allowed)
# Drop everything else (numbers, URLs, punctuation, emojis).
TOK_RE = re.compile(
    r"[\u0900-\u097F]+|"   # Devanagari span
    r"[A-Za-z]+"           # English letters (single letter or word)
)

# After romanizing Hindi, restrict to a–z only (letters only), lowercase
ONLY_ASCII_LETTERS = re.compile(r"[^a-z]+")

def clean_roman_letters(s: str) -> str:
    return ONLY_ASCII_LETTERS.sub("", s.lower())

# ===== Pipelines (NO LEMMAS, NO STOPWORDS) =====
_HI_NLP = download_stanza.Pipeline(
    lang="hi",
    processors="tokenize,pos",  # no lemma
    use_gpu=False,
    tokenize_pretokenized=False
)
_EN_NLP = spacy.load("en_core_web_sm")

# =========================
# Language-specific tagging
# =========================

def tag_hindi(hi_tokens: List[str]) -> List[Dict]:
    """
    Input: list of Devanagari tokens (letters only after filtering).
    Output (same order): dicts with
        - token_dev: original Devanagari token
        - token_roman: romanized (SLP1) lowercase letters only
        - pos: UPOS from stanza (fallback 'X')
        - lang: 'hi'
    """
    out = []
    if not hi_tokens:
        return out

    # POS via stanza (best effort alignment by simple join)
    doc = _HI_NLP(" ".join(hi_tokens))
    words = []
    for sent in doc.sentences:
        words.extend(sent.words)

    for orig_tok, w in zip(hi_tokens, words):
        pos = w.upos or "X"

        # Romanize then filter to letters only
        roman_raw = transliterate(orig_tok, sanscript.DEVANAGARI, sanscript.SLP1)
        roman = clean_roman_letters(roman_raw)
        # If roman becomes empty (e.g., token was punctuation-like), skip
        if not roman:
            continue

        out.append({
            "token_dev": orig_tok,
            "token_roman": roman,
            "pos": pos,
            "lang": "hi"
        })
    return out


def tag_english(en_tokens: List[str]) -> List[Dict]:
    """
    Input: list of ASCII-letter tokens (filtered to letters only).
    Output (same order): dicts with
        - token_en: lowercase letters only
        - pos: POS from spaCy (fallback 'X')
        - lang: 'en'
    """
    out = []
    if not en_tokens:
        return out

    # POS via spaCy (best effort alignment)
    doc = _EN_NLP(" ".join(en_tokens))
    toks = [t for t in doc if not t.is_space]

    for orig_tok, t in zip(en_tokens, toks):
        if not orig_tok.isalpha():
            # Safety: only letters allowed
            continue
        token_lc = orig_tok.lower()
        pos = t.pos_ or "X"

        out.append({
            "token_en": token_lc,
            "pos": pos,
            "lang": "en"
        })
    return out

# =========================
# Main per-utterance logic
# =========================

def process_utterance(utt: str) -> Dict[str, List]:
    """
    Pipeline:
      - Remove <Hindi> tags.
      - Tokenize to letter-only sequences (English letters or Devanagari).
      - Drop everything else (numbers, URLs, punctuation, emojis).
      - POS tag Hindi (stanza) and English (spaCy).
      - Reconstruct original order.

    Outputs (aligned lists, lowercase where applicable):
      tokens_roman : English tokens (lowercase) + Hindi romanized (lowercase, letters only)
      tokens_dev   : Hindi original Devanagari for Hindi tokens; English repeats lowercase word
      pos_tags     : POS tag per token
      token_languages : 'en' or 'hi' only
    """
    utt = remove_hindi_tags(utt)
    raw_toks = TOK_RE.findall(utt)  # already only letter sequences

    # Separate by script
    hi_seq = [t for t in raw_toks if is_devanagari(t)]
    en_seq = [t for t in raw_toks if not is_devanagari(t)]  # by regex, these are A–Z only

    hi_tagged = tag_hindi(hi_seq)
    en_tagged = tag_english(en_seq)

    # Rebuild in original order
    hi_i = 0
    en_i = 0
    tokens_roman: List[str] = []
    tokens_dev:   List[str] = []
    pos_tags:     List[str] = []
    langs:        List[str] = []

    for tok in raw_toks:
        if is_devanagari(tok):
            if hi_i >= len(hi_tagged):
                # Fallback: romanize + POS unknown
                roman = clean_roman_letters(transliterate(tok, sanscript.DEVANAGARI, sanscript.SLP1))
                if not roman:
                    continue
                tokens_roman.append(roman)
                tokens_dev.append(tok)
                pos_tags.append("X")
                langs.append("hi")
            else:
                item = hi_tagged[hi_i]; hi_i += 1
                tokens_roman.append(item["token_roman"])
                tokens_dev.append(item["token_dev"])
                pos_tags.append(item["pos"])
                langs.append("hi")
        else:
            # English (letters only)
            if not tok.isalpha():
                continue
            if en_i >= len(en_tagged):
                token_lc = tok.lower()
                tokens_roman.append(token_lc)
                tokens_dev.append(token_lc)  # mirror for alignment
                pos_tags.append("X")
                langs.append("en")
            else:
                item = en_tagged[en_i]; en_i += 1
                token_lc = item["token_en"]
                tokens_roman.append(token_lc)
                tokens_dev.append(token_lc)
                pos_tags.append(item["pos"])
                langs.append("en")

    return {
        "tokens_roman": tokens_roman,
        "tokens_dev": tokens_dev,
        "pos_tags": pos_tags,
        "token_languages": langs
    }

# =========================
# File-level processing
# =========================

def process_file(
    input_path: str,
    output_path: str,
    text_col: str = "utterance"
) -> pd.DataFrame:
    """
    Reads CSV/JSON with column `text_col`.
    Writes CSV with:
      - tokens_roman, tokens_dev, pos_tags, token_languages
      - num_tokens, num_english_tokens, num_hindi_tokens, is_code_switched
    Only letters retained (no numbers/URLs/punct). Single-letter tokens allowed.
    """
    print(f"Loading: {input_path}")
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path)

    tokens_roman_col, tokens_dev_col, pos_col, langs_col = [], [], [], []

    for i, row in df.iterrows():
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}...")
        text = str(row.get(text_col, ""))
        rec = process_utterance(text)
        tokens_roman_col.append(rec["tokens_roman"])
        tokens_dev_col.append(rec["tokens_dev"])
        pos_col.append(rec["pos_tags"])
        langs_col.append(rec["token_languages"])

    out_df = df.copy()
    out_df["tokens_roman"] = tokens_roman_col
    out_df["tokens_dev"] = tokens_dev_col
    out_df["pos_tags"] = pos_col
    out_df["token_languages"] = langs_col

    out_df["num_tokens"] = out_df["tokens_roman"].apply(len)
    out_df["num_english_tokens"] = out_df["token_languages"].apply(lambda x: x.count("en"))
    out_df["num_hindi_tokens"] = out_df["token_languages"].apply(lambda x: x.count("hi"))
    out_df["is_code_switched"] = (out_df["num_hindi_tokens"] > 0) & (out_df["num_english_tokens"] > 0)

    # Remove any ratio columns if present
    for col in ["hindi_ratio", "ratio"]:
        if col in out_df.columns:
            out_df = out_df.drop(columns=[col])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved: {output_path}")

    # Sample
    print("\nSAMPLE ROWS")
    for i in range(min(5, len(out_df))):
        print(f"\nRow {i+1}")
        print("tokens_roman:", out_df.iloc[i]["tokens_roman"][:12])
        print("tokens_dev  :", out_df.iloc[i]["tokens_dev"][:12])
        print("pos         :", out_df.iloc[i]["pos_tags"][:12])
        print("langs       :", out_df.iloc[i]["token_languages"][:12])

    return out_df

# =========
# __main__
# =========
if __name__ == "__main__":
    # Change these paths as needed
    input_path = "data/processed/codeswitch_by_sentences_v02.csv"
    output_path = "data/processed/sentences_with_tags_v02.csv"

    df = process_file(input_path, output_path, text_col="utterance")
    print(df["is_code_switched"].sum(), "/", len(df), "are code-switched")
    print("English tokens:", df["num_english_tokens"].sum())
    print("Hindi tokens  :", df["num_hindi_tokens"].sum())
    print("\n✓ Done.")
