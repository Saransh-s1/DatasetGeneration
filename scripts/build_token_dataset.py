# /src/01_build_token_dataset.py
import ast
import re
import pandas as pd
from pathlib import Path

RAW = Path("data/processed/final_dataset_v01.csv")
OUT = Path("data/processed/tokens_labeled.csv")

def parse_list_cell(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return ast.literal_eval(x)

def is_punct(tok: str) -> int:
    if tok is None:
        return 0
    tok = tok.strip()
    return int(bool(re.fullmatch(r"\W+", tok)))

def main():
    df = pd.read_csv(RAW)

    # Parse list-like columns
    for col in ["tokens_roman", "pos_tags", "token_languages"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_cell)
        else:
            raise ValueError(f"Missing required column: {col}")

    # Sanity: enforce aligned lengths
    ok = (df["tokens_roman"].str.len() == df["pos_tags"].str.len()) & \
         (df["tokens_roman"].str.len() == df["token_languages"].str.len())
    if not ok.all():
        # Keep only well-formed rows
        df = df[ok].copy()

    rows = []
    for _, r in df.iterrows():
        toks = [t.lower() for t in r["tokens_roman"]]
        poses = r["pos_tags"]
        langs = r["token_languages"]
        n = len(toks)

        for i in range(n):
            curr_tok = toks[i]
            curr_pos = poses[i]
            curr_lang = langs[i]
            prev_tok = toks[i-1] if i > 0 else ""
            prev_pos = poses[i-1] if i > 0 else "NONE"
            prev_lang = langs[i-1] if i > 0 else "NONE"

            # Label: switch at token i if previous token exists and language changes
            y_switch = int(i > 0 and curr_lang != prev_lang)

            rows.append({
                "conversation_id": r.get("conversation_id"),
                "utterance_id": r.get("utterance_id"),
                "speaker": r.get("speaker"),
                "generation_strategy": r.get("generation_strategy"),
                "position_idx": i,
                "position_frac": (i / (n - 1)) if n > 1 else 0.0,
                "curr_tok": curr_tok,
                "prev_tok": prev_tok,
                "curr_pos": curr_pos,
                "prev_pos": prev_pos,
                "curr_len": len(curr_tok),
                "prev_len": len(prev_tok),
                "is_punct": is_punct(curr_tok),
                "prev_is_punct": is_punct(prev_tok),
                "prev_lang": prev_lang,
                "curr_lang": curr_lang,
                "y_switch": y_switch,
            })

    token_df = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    token_df.to_csv(OUT, index=False)
    print(f"Saved token-level dataset: {OUT}  ({len(token_df)} rows)")

if __name__ == "__main__":
    main()
