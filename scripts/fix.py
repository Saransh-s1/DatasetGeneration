import pandas as pd

# --- Hardcoded settings ---
INPUT_CSV = "../data/processed/sentences_with_tags_v02.csv"
OUTPUT_CSV = "../data/processed/final_dataset_v01.csv"
SPEAKER_COL = "speaker"
UTTERANCE_COL = "utterance"
# --------------------------

def main():
    df = pd.read_csv(INPUT_CSV)

    for col in (SPEAKER_COL, UTTERANCE_COL):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("*", "", regex=False)
        else:
            print(f"Warning: column '{col}' not found; skipping.")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Cleaned file saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()