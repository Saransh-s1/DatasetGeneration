import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# Hindi NLP
import stanza
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import stopwordsiso as stopiso

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

# Tokenizer: keep word-like, numbers, Devanagari spans, and single punctuation symbols.
# This preserves original order so we can re-insert processed outputs correctly.
TOK_RE = re.compile(
    r"[\u0900-\u097F]+|"              # Devanagari span
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"      # English word (with optional apostrophe segment)
    r"\d+(?:[.,]\d+)?|"               # number
    r"https?://\S+|www\.\S+|"         # URLs
    r"[^\s\w]"                        # single non-space, non-word char (punct, emoji, symbols)
)

NUM_RE   = re.compile(r"^\d+(?:[.,]\d+)?$")
PUNCT_RE = re.compile(r"^\W+$")

# Build NLP pipelines
_HI_NLP = stanza.Pipeline(
    lang="hi",
    processors="tokenize,pos,lemma",
    use_gpu=False,
    tokenize_pretokenized=False
)
_EN_NLP = spacy.load("en_core_web_sm")

# Stopwords
_HI_STOP_DEV = set(stopiso.stopwords("hi"))     # Devanagari
_EN_STOP     = _EN_NLP.Defaults.stop_words      # ASCII (lemmas lowercased)


# =========================
# Language-specific tagging
# =========================

def tag_hindi(hi_tokens: List[str]) -> List[Dict]:
    """
    Input: list of Devanagari tokens (no punctuation).
    Output: list of dicts in the SAME order, each containing:
        orig_dev, token_slp1, lemma_dev, lemma_slp1, pos, lang='hi', drop (bool)
    We remove stopwords by *lemma in Devanagari* (drop=True).
    """
    out = []
    if not hi_tokens:
        return out

    # Stanza expects text; join with spaces
    hi_text = " ".join(hi_tokens)
    doc = _HI_NLP(hi_text)

    # Collect words in order
    words: List[stanza.models.common.doc.Word] = []
    for sent in doc.sentences:
        words.extend(sent.words)

    # Stanza will re-tokenize Hindi; in most cases the sequential order matches the token list we built.
    # We align by order (zip). If lengths differ, zip will truncate extra tokens.
    for orig_tok, w in zip(hi_tokens, words):
        lemma_dev = w.lemma or w.text
        pos = w.upos or "X"

        # Stopword removal by lemma (devanagari)
        drop = lemma_dev in _HI_STOP_DEV

        out.append({
            "orig_dev": orig_tok,
            "token_slp1": transliterate(orig_tok, sanscript.DEVANAGARI, sanscript.SLP1),
            "lemma_dev": lemma_dev,
            "lemma_slp1": transliterate(lemma_dev, sanscript.DEVANAGARI, sanscript.SLP1),
            "pos": pos,
            "lang": "hi",
            "drop": drop
        })
    return out


def tag_english(en_tokens: List[str]) -> List[Dict]:
    """
    Input: list of ASCII word tokens (no punctuation).
    Output: list of dicts in order:
        orig_en, token_en, lemma_en, pos, lang='en', drop (bool)
    We remove stopwords by *lemma (lowercased)*.
    """
    out = []
    if not en_tokens:
        return out

    en_text = " ".join(en_tokens)
    doc = _EN_NLP(en_text)

    # spaCy will tokenize again; align by order with the provided words (skip spaces)
    toks = [t for t in doc if not t.is_space]

    for orig_tok, t in zip(en_tokens, toks):
        if not (t.is_alpha or t.like_num):
            # Skip weird tokens here; punctuation handled outside
            lemma = t.lemma_.lower() if t.lemma_ else t.text.lower()
            out.append({
                "orig_en": orig_tok,
                "token_en": t.text,
                "lemma_en": lemma,
                "pos": t.pos_ or "X",
                "lang": "en",
                "drop": False
            })
            continue

        lemma = (t.lemma_ or t.text).lower()
        drop = (t.is_alpha and lemma in _EN_STOP)

        out.append({
            "orig_en": orig_tok,
            "token_en": t.text,
            "lemma_en": lemma,
            "pos": t.pos_ or "X",
            "lang": "en",
            "drop": drop
        })
    return out


# =========================
# Main per-utterance logic
# =========================

def process_utterance(utt: str) -> Dict[str, List]:
    """
    For a single utterance:
      1) Tokenize preserving order (words, numbers, punct).
      2) Collect Hindi-only and English-only word tokens (in order).
      3) Tag each group separately (Hindi via Stanza, English via spaCy).
      4) Reconstruct the final stream in original order:
            - Replace Hindi words with processed (drop if stopword).
            - Replace English words with processed (drop if stopword).
            - Keep numbers/punct/others with simple tags.
    Returns dict with lists: tokens, lemmas, pos_tags, token_languages
    (All ASCII; Hindi transliterated to SLP1)
    """
    raw_toks = TOK_RE.findall(utt)

    # Sequences for in-language tagging
    hi_seq = [t for t in raw_toks if is_devanagari(t)]
    en_seq = [t for t in raw_toks if (not is_devanagari(t) and re.match(r"^[A-Za-z]", t))]

    # Tagging
    hi_tagged = tag_hindi(hi_seq)
    en_tagged = tag_english(en_seq)

    # Pointers for sequential consumption
    hi_i = 0
    en_i = 0

    final_tokens: List[str] = []
    final_lemmas: List[str] = []
    final_pos: List[str] = []
    final_langs: List[str] = []

    for tok in raw_toks:
        if is_devanagari(tok):
            if hi_i >= len(hi_tagged):
                # Fallback: include raw transliteration as unknown
                final_tokens.append(transliterate(tok, sanscript.DEVANAGARI, sanscript.SLP1))
                final_lemmas.append(transliterate(tok, sanscript.DEVANAGARI, sanscript.SLP1))
                final_pos.append("X")
                final_langs.append("hi")
            else:
                item = hi_tagged[hi_i]; hi_i += 1
                if item["drop"]:
                    # Drop Hindi stopword
                    continue
                final_tokens.append(item["token_slp1"])
                final_lemmas.append(item["lemma_slp1"])
                final_pos.append(item["pos"])
                final_langs.append("hi")

        elif re.match(r"^[A-Za-z]", tok):
            if en_i >= len(en_tagged):
                # Fallback: include as unknown
                final_tokens.append(tok)
                final_lemmas.append(tok.lower())
                final_pos.append("X")
                final_langs.append("en")
            else:
                item = en_tagged[en_i]; en_i += 1
                if item["drop"]:
                    # Drop English stopword
                    continue
                final_tokens.append(item["token_en"])
                final_lemmas.append(item["lemma_en"])
                final_pos.append(item["pos"])
                final_langs.append("en")

        else:
            # numbers / punctuation / symbols / URLs
            if NUM_RE.match(tok):
                pos = "NUM"
            elif PUNCT_RE.match(tok):
                pos = "PUNCT"
            else:
                pos = "SYM"
            final_tokens.append(tok)
            final_lemmas.append(tok)
            final_pos.append(pos)
            final_langs.append("other")

    return {
        "tokens": final_tokens,
        "lemmas": final_lemmas,
        "pos_tags": final_pos,
        "token_languages": final_langs
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
    Writes a combined CSV with both languages processed and merged per row.
    """
    print(f"Loading: {input_path}")
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path)

    tokens_col, lemmas_col, pos_col, langs_col = [], [], [], []

    for i, row in df.iterrows():
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}...")
        text = str(row.get(text_col, ""))
        rec = process_utterance(text)
        tokens_col.append(rec["tokens"])
        lemmas_col.append(rec["lemmas"])
        pos_col.append(rec["pos_tags"])
        langs_col.append(rec["token_languages"])

    out_df = df.copy()
    out_df["tokens"] = tokens_col
    out_df["lemmas"] = lemmas_col
    out_df["pos_tags"] = pos_col
    out_df["token_languages"] = langs_col

    out_df["num_tokens"] = out_df["tokens"].apply(len)
    out_df["num_english_tokens"] = out_df["token_languages"].apply(lambda x: x.count("en"))
    out_df["num_hindi_tokens"] = out_df["token_languages"].apply(lambda x: x.count("hi"))
    out_df["is_code_switched"] = (out_df["num_hindi_tokens"] > 0) & (out_df["num_english_tokens"] > 0)
    out_df["hindi_ratio"] = out_df.apply(
        lambda r: (r["num_hindi_tokens"] / r["num_tokens"]) if r["num_tokens"] else 0.0,
        axis=1
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved: {output_path}")

    # Small sample
    print("\nSAMPLE ROWS")
    for i in range(min(5, len(out_df))):
        print(f"\nRow {i+1}")
        print("tokens:", out_df.iloc[i]["tokens"][:12])
        print("lemmas:", out_df.iloc[i]["lemmas"][:12])
        print("pos   :", out_df.iloc[i]["pos_tags"][:12])
        print("langs :", out_df.iloc[i]["token_languages"][:12])

    return out_df


# =========
# __main__
# =========
if __name__ == "__main__":
    # Change these paths as needed
    input_path = "data/processed/codeswitch_v01.csv"
    output_path = "data/processed/codeswitch_v01_combined.csv"

    process_file(input_path, output_path, text_col="utterance")
    print("\n✓ Done.")
