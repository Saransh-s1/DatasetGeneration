import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# Hindi NLP
import stanza as download_stanza
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

# --- Strip <Hindi> tags (case-insensitive, with/without attributes) ---
LANG_TAG_RE = re.compile(r'</?\s*hindi\b[^>]*>', flags=re.I)

def strip_hindi_tags(text: str) -> str:
    return LANG_TAG_RE.sub('', text)

# --- Big-O (capture & normalize) ---
# Make sure O(log n) is one token, regardless of spaces/case inside.
BIGO_TOKEN_RE = r"[Oo]\s*\(\s*log\s+[A-Za-z0-9]+\s*\)"  # e.g., O(log n), o ( log N )
BIGO_NORMALIZE_INNER_SPACES_RE = re.compile(r"\s+")

def is_big_o(tok: str) -> bool:
    return re.fullmatch(BIGO_TOKEN_RE, tok) is not None

def normalize_big_o(tok: str) -> str:
    # Force leading 'O', remove extra spaces after 'O', squash inner spaces to single,
    # then remove spaces right after '(' and before ')'
    # Examples:
    #  "o ( log   n )" -> "O(log n)"
    tok = tok.strip()
    # Uppercase the initial 'O'
    tok = re.sub(r'^[oO]\s*\(', 'O(', tok)
    # Condense internal runs of whitespace
    inside = tok[2:-1] if tok.startswith('O(') and tok.endswith(')') else tok
    inside = BIGO_NORMALIZE_INNER_SPACES_RE.sub(' ', inside)
    return f"O({inside})" if tok.startswith('O(') and tok.endswith(')') else tok

# Tokenizer:
#  - Big-O pattern FIRST to keep "O(log n)" as ONE TOKEN
#  - Keep Devanagari spans
#  - English word possibly with apostrophes and hyphenated segments kept together
#  - Numbers, URLs, single punctuation symbols
TOK_RE = re.compile(
    rf"{BIGO_TOKEN_RE}|"                 # Big-O as a single token
    r"[\u0900-\u097F]+|"                # Devanagari span
    r"[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+)*|"  # English word + optional apostrophe + hyphens
    r"\d+(?:[.,]\d+)?|"                 # number
    r"https?://\S+|www\.\S+|"           # URLs
    r"[^\s\w]"                          # single non-space, non-word char (punct, emoji, symbols)
)

NUM_RE   = re.compile(r"^\d+(?:[.,]\d+)?$")
PUNCT_RE = re.compile(r"^\W+$")

# Build NLP pipelines
_HI_NLP = download_stanza.Pipeline(
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
    words: List[download_stanza.models.common.doc.Word] = []
    for sent in doc.sentences:
        words.extend(sent.words)

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
    Also normalizes Big-O tokens to canonical 'O(log n)' form.
    """
    out = []
    if not en_tokens:
        return out

    en_text = " ".join(en_tokens)
    doc = _EN_NLP(en_text)

    toks = [t for t in doc if not t.is_space]

    for orig_tok, t in zip(en_tokens, toks):
        surface = t.text

        # Preserve/normalize Big-O as a unit
        if is_big_o(surface):
            tok_norm = normalize_big_o(surface)
            out.append({
                "orig_en": orig_tok,
                "token_en": tok_norm,
                "lemma_en": tok_norm,   # treat Big-O as a fixed form
                "pos": "SYM",
                "lang": "en",
                "drop": False
            })
            continue

        if not (t.is_alpha or t.like_num or '-' in surface):
            # Non standard token (urls/punct handled elsewhere), just pass through
            lemma = t.lemma_.lower() if t.lemma_ else surface.lower()
            out.append({
                "orig_en": orig_tok,
                "token_en": surface,
                "lemma_en": lemma,
                "pos": t.pos_ or "X",
                "lang": "en",
                "drop": False
            })
            continue

        # Lemmatize; keep hyphenated compounds as-is for token,
        # lemma from spaCy (lowercased). We do NOT split hyphenated forms.
        lemma = (t.lemma_ or surface).lower()

        # Stopword removal by lemma (only for pure alphabetic words)
        drop = (surface.replace('-', '').isalpha() and lemma in _EN_STOP)

        out.append({
            "orig_en": orig_tok,
            "token_en": surface,
            "lemma_en": lemma if not is_big_o(surface) else normalize_big_o(surface),
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
      - Strip <Hindi> tags
      - Tokenize preserving order (words, numbers, punct), with special handling:
          * Big-O kept as single token and normalized
          * Hyphenated English words kept as single tokens
      - Tag Hindi (Stanza) and English (spaCy) separately
      - Reconstruct final stream in original order (drop stopwords)
    Returns dict with lists: tokens, lemmas, pos_tags, token_languages
    (All ASCII; Hindi transliterated to SLP1)
    """
    # 0) Remove language tags
    utt = strip_hindi_tags(utt)

    # 1) Tokenize
    raw_toks = TOK_RE.findall(utt)

    # 2) Prepare language-specific sequences
    hi_seq = [t for t in raw_toks if is_devanagari(t)]
    en_seq = [t for t in raw_toks if (not is_devanagari(t) and re.match(r"^[A-Za-z]", t)) or is_big_o(t)]

    # 3) Tagging
    hi_tagged = tag_hindi(hi_seq)
    en_tagged = tag_english(en_seq)

    # 4) Reconstruct
    hi_i = 0
    en_i = 0

    final_tokens: List[str] = []
    final_lemmas: List[str] = []
    final_pos: List[str] = []
    final_langs: List[str] = []

    for tok in raw_toks:
        if is_devanagari(tok):
            if hi_i >= len(hi_tagged):
                # Fallback
                final_tokens.append(transliterate(tok, sanscript.DEVANAGARI, sanscript.SLP1))
                final_lemmas.append(transliterate(tok, sanscript.DEVANAGARI, sanscript.SLP1))
                final_pos.append("X")
                final_langs.append("hi")
            else:
                item = hi_tagged[hi_i]; hi_i += 1
                if item["drop"]:
                    continue
                final_tokens.append(item["token_slp1"])
                final_lemmas.append(item["lemma_slp1"])
                final_pos.append(item["pos"])
                final_langs.append("hi")

        elif is_big_o(tok) or re.match(r"^[A-Za-z]", tok):
            if en_i >= len(en_tagged):
                # Fallback
                tok_out = normalize_big_o(tok) if is_big_o(tok) else tok
                final_tokens.append(tok_out)
                final_lemmas.append(tok_out.lower() if not is_big_o(tok_out) else tok_out)
                final_pos.append("SYM" if is_big_o(tok_out) else "X")
                final_langs.append("en")
            else:
                item = en_tagged[en_i]; en_i += 1
                if item["drop"]:
                    continue
                # Normalize Big-O once more for safety
                tok_out = normalize_big_o(item["token_en"]) if is_big_o(item["token_en"]) else item["token_en"]
                lem_out = item["lemma_en"]
                if is_big_o(tok_out):
                    lem_out = tok_out
                final_tokens.append(tok_out)
                final_lemmas.append(lem_out)
                final_pos.append(item["pos"] if not is_big_o(tok_out) else "SYM")
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
