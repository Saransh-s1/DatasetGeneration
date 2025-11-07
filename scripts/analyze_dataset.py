# /src/02_analyze_dataset.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INP = Path("data/processed/tokens_labeled.csv")
FIGDIR = Path("reports/figures")

def main():
    df = pd.read_csv(INP)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # Totals
    total_tokens = len(df)
    total_utts = df["utterance_id"].nunique()

    # Mixed-utterance rate: utterance has at least one switch
    utt_switch = df.groupby("utterance_id")["y_switch"].max()
    mixed_rate = float(utt_switch.mean())

    # Language proportions (by token)
    lang_props = df["curr_lang"].value_counts(normalize=True).sort_values(ascending=False)

    print("=== Dataset Stats ===")
    print(f"Total tokens: {total_tokens}")
    print(f"Total utterances: {total_utts}")
    print(f"Mixed-utterance rate: {mixed_rate:.3f}  (target 0.20–0.30)")
    print("Language proportions:", lang_props.to_dict())

    # 1) Language proportion bar
    ax = lang_props.plot(kind="bar")
    ax.set_title("Language Proportion (Tokens)")
    ax.set_ylabel("Fraction")
    plt.tight_layout()
    plt.savefig(FIGDIR / "lang_proportion.png")
    plt.clf()

    # 2) Histogram of switch positions (position_frac where y_switch==1)
    df[df["y_switch"] == 1]["position_frac"].plot(kind="hist", bins=20)
    plt.title("Histogram of Switch Positions (fraction of utterance)")
    plt.xlabel("Position (0=start, 1=end)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "switch_position_hist.png")
    plt.clf()

    # 3) Utterance length distribution
    utt_len = df.groupby("utterance_id")["position_idx"].max() + 1
    utt_len.plot(kind="hist", bins=20)
    plt.title("Distribution of Utterance Lengths (tokens)")
    plt.xlabel("Tokens per utterance")
    plt.tight_layout()
    plt.savefig(FIGDIR / "utterance_length_hist.png")
    plt.clf()

    # (Optional) Token length by language
    df.boxplot(column="curr_len", by="curr_lang")
    plt.title("Token Length by Language")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(FIGDIR / "token_len_by_language.png")
    plt.clf()

    print(f"Saved figures to {FIGDIR}")

if __name__ == "__main__":
    main()
