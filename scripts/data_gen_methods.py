import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Config
# =========================
SENTENCE_LEVEL_PATH = "data/processed/final_dataset_v01.csv"
TOKEN_LEVEL_PATH = "data/processed/tokens_labeled.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_sentence_length_distribution(df):
    """
    Figure 1: Histogram of sentence lengths in tokens.
    """
    plt.figure()
    df["num_tokens"].hist(bins=20)
    plt.title("Sentence Length Distribution")
    plt.xlabel("Number of Tokens per Utterance")
    plt.ylabel("Count of Utterances")
    plt.tight_layout()
    out_path = FIG_DIR / "fig_sentence_length_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_language_token_distribution(df):
    """
    Figure 2: Bar chart of total English vs Hindi tokens.
    """
    en_total = df["num_english_tokens"].sum()
    hi_total = df["num_hindi_tokens"].sum()

    plt.figure()
    plt.bar(["English", "Hindi"], [en_total, hi_total])
    plt.title("Token Distribution Across Languages")
    plt.xlabel("Language")
    plt.ylabel("Total Token Count")
    plt.tight_layout()
    out_path = FIG_DIR / "fig_language_token_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_sentence_level_codeswitch_rate(df):
    """
    Figure 3: Bar chart of code-switched vs non-code-switched utterances.
    """
    counts = df["is_code_switched"].value_counts().sort_index()
    labels = ["Non-Code-Switched", "Code-Switched"]

    plt.figure()
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Sentence-Level Code-Switching Frequency")
    plt.ylabel("Number of Utterances")
    plt.tight_layout()
    out_path = FIG_DIR / "fig_sentence_level_codeswitch_rate.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_token_switch_vs_position(token_df):
    """
    Figure 4: Switch probability as a function of token position in the utterance.

    Uses 'position_frac' (0 = first token, 1 = last token) and computes
    average y_switch in bins along the sentence.
    """
    # Bin positions into deciles
    bins = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    token_df["pos_bin"] = pd.cut(
        token_df["position_frac"],
        bins=bins,
        include_lowest=True,
        labels=[f"{int(b*100)}–{int((b+0.1)*100)}%" for b in bins[:-1]]
    )

    switch_rates = (
        token_df.groupby("pos_bin")["y_switch"]
        .mean()
        .reset_index()
        .dropna()
    )

    plt.figure()
    plt.plot(switch_rates["pos_bin"].astype(str), switch_rates["y_switch"], marker="o")
    plt.title("Token-Level Switch Probability vs Position in Utterance")
    plt.xlabel("Relative Token Position (Percent of Utterance)")
    plt.ylabel("P(Switch at Token)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = FIG_DIR / "fig_token_switch_vs_position.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main():
    print("Loading sentence-level dataset...")
    sent_df = pd.read_csv(SENTENCE_LEVEL_PATH)

    print("Loading token-level dataset...")
    tok_df = pd.read_csv(TOKEN_LEVEL_PATH)

    # 1) Sentence length distribution
    plot_sentence_length_distribution(sent_df)

    # 2) Language token distribution
    plot_language_token_distribution(sent_df)

    # 3) Sentence-level code-switching rate
    plot_sentence_level_codeswitch_rate(sent_df)

    # 4) Token-level switch vs position
    plot_token_switch_vs_position(tok_df)

    print("\nAll figures generated and saved in:", FIG_DIR)


if __name__ == "__main__":
    main()
