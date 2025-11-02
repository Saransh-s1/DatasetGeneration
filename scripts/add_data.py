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

# Remove <Hindi>…</Hindi> tags (any case/spacing/attributes)
TAG_RE = re.compile(r'<\s*/?\s*Hindi\b[^>]*>', flags=re.IGNORECASE)

def remove_hindi_tags(text: str) -> str:
    text = TAG_RE.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

# Tokenizer: keep word-like, numbers, Devanagari spans, and single punctuation symbols.
TOK_RE = re.compile(
    r"[\u0900-\u097F]+|"              # Devanagari span
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"      # English word (with optional apostrophe segment)
    r"\d+(?:[.,]\d+)?|"               # number
    r"https?://\S+|www\.\S+|"         # URLs
    r"[^\s\w]"                        # single non-space, non-word char (punct, emoji, symbols)
)

NUM_RE   = re.compile(r"^\d+(?:[.,]\d+)?$")
PUNCT_RE = re.compile(r"^\W+$")

# Keep only lowercase ascii letters/digits in tokens (after transliteration for Hindi)
# This also removes hyphens, apostrophes, punctuation, emoji, etc.
KEEP_ASCII = re.compile(r"[^a-z0-9]+")

def clean_ascii_token(s: str) -> str:
    s = s.lower()
    s = KEEP_ASCII.sub("", s)
    return s

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
        token_slp1, lemma_slp1, pos, lang='hi', drop (bool)
    - Transliterate to SLP1, lowercase, strip special chars to [a-z0-9].
    - Drop if lemma (Devanagari) is a stopword.
    """
    out = []
    if not hi_tokens:
        return out

    hi_text = " ".join(hi_tokens)
    doc = _HI_NLP(hi_text)

    words: List[download_stanza.models.common.doc.Word] = []
    for sent in doc.sentences:
        words.extend(sent.words)

    for orig_tok, w in zip(hi_tokens, words):
        lemma_dev = w.lemma or w.text
        pos = w.upos or "X"
        drop = lemma_dev in _HI_STOP_DEV

        token_slp1_raw = transliterate(orig_tok, sanscript.DEVANAGARI, sanscript.SLP1)
        lemma_slp1_raw = transliterate(lemma_dev, sanscript.DEVANAGARI, sanscript.SLP1)

        token_slp1 = clean_ascii_token(token_slp1_raw)
        lemma_slp1 = clean_ascii_token(lemma_slp1_raw)

        # If cleaning nukes the token, skip it.
        if not token_slp1:
            continue

        out.append({
            "token_slp1": token_slp1,
            "lemma_slp1": lemma_slp1 if lemma_slp1 else token_slp1,
            "pos": pos,
            "lang": "hi",
            "drop": drop
        })
    return out


def tag_english(en_tokens: List[str]) -> List[Dict]:
    """
    Input: list of ASCII word tokens (no punctuation).
    Output: list of dicts in order:
        token_en, lemma_en, pos, lang='en', drop (bool)
    - Lowercase and strip special characters to [a-z0-9].
    - Remove stopwords by lemma (lowercased).
    """
    out = []
    if not en_tokens:
        return out

    en_text = " ".join(en_tokens)
    doc = _EN_NLP(en_text)
    toks = [t for t in doc if not t.is_space]

    for orig_tok, t in zip(en_tokens, toks):
        # Clean both surface and lemma
        token_clean = clean_ascii_token(t.text)
        lemma_clean = clean_ascii_token(t.lemma_ or t.text)

        # If cleaning nukes the token, skip it.
        if not token_clean:
            continue

        # Stopword removal: only if it’s alphabetic (after cleaning) and in stop set
        drop = (lemma_clean.isalpha() and lemma_clean in _EN_STOP)

        out.append({
            "token_en": token_clean,
            "lemma_en": lemma_clean if lemma_clean else token_clean,
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
    Pipeline:
      - Remove <Hindi> tags.
      - Tokenize.
      - Tag Hindi (Devanagari-only) and English (ASCII-leading) separately.
      - Reconstruct in original order, dropping punctuation and any token that
        becomes empty after cleaning, and dropping stopwords.
      - Outputs are all lowercase; Hindi in SLP1 with only [a-z0-9].
    """
    # 1) Remove <Hindi> tags
    utt = remove_hindi_tags(utt)

    # 2) Tokenize
    raw_toks = TOK_RE.findall(utt)

    # 3) Prepare sequences for tagging
    hi_seq = [t for t in raw_toks if is_devanagari(t)]
    en_seq = [t for t in raw_toks if (not is_devanagari(t) and re.match(r"^[A-Za-z]", t))]

    # 4) Tag
    hi_tagged = tag_hindi(hi_seq)
    en_tagged = tag_english(en_seq)

    # 5) Rebuild in order
    hi_i = 0
    en_i = 0

    final_tokens: List[str] = []
    final_lemmas: List[str] = []
    final_pos: List[str] = []
    final_langs: List[str] = []

    for tok in raw_toks:
        if is_devanagari(tok):
            if hi_i >= len(hi_tagged):
                # Fallback: transliterate, clean, skip if empty
                t = clean_ascii_token(transliterate(tok, sanscript.DEVANAGARI, sanscript.SLP1))
                if not t:
                    continue
                final_tokens.append(t)
                final_lemmas.append(t)
                final_pos.append("X")
                final_langs.append("hi")
            else:
                item = hi_tagged[hi_i]; hi_i += 1
                if item["drop"]:
                    continue
                t = item["token_slp1"]
                l = item["lemma_slp1"] if item["lemma_slp1"] else t
                if not t:
                    continue
                final_tokens.append(t)
                final_lemmas.append(l)
                final_pos.append(item["pos"])
                final_langs.append("hi")

        elif re.match(r"^[A-Za-z]", tok):
            if en_i >= len(en_tagged):
                t = clean_ascii_token(tok)
                if not t:
                    continue
                final_tokens.append(t)
                final_lemmas.append(t)
                final_pos.append("X")
                final_langs.append("en")
            else:
                item = en_tagged[en_i]; en_i += 1
                if item["drop"]:
                    continue
                t = item["token_en"]
                l = item["lemma_en"] if item["lemma_en"] else t
                if not t:
                    continue
                final_tokens.append(t)
                final_lemmas.append(l)
                final_pos.append(item["pos"])
                final_langs.append("en")

        else:
            # numbers / punctuation / symbols / URLs
            # Punctuation & symbols are dropped entirely (special-char removal).
            if NUM_RE.match(tok):
                # keep numbers; normalize to digits only (drop commas/periods)
                t = KEEP_ASCII.sub("", tok.lower())
                if not t:
                    continue
                final_tokens.append(t)
                final_lemmas.append(t)
                final_pos.append("NUM")
                final_langs.append("other")
            else:
                # drop everything else here (punct/emoji/URLs)
                continue

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
    Writes a combined CSV with cleaned, lowercased, special-char-stripped tokens/lemmas.
    Drops any ratio columns from the final CSV.
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

    # DO NOT compute hindi_ratio; also drop any ratio columns if they exist
    for col in ["hindi_ratio", "ratio"]:
        if col in out_df.columns:
            out_df = out_df.drop(columns=[col])

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
