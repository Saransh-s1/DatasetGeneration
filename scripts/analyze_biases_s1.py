#!/usr/bin/env python3
"""
section1_bias_analysis_final.py

Computes the three required bias views for Section I:

1) Role / Expertise Bias
   - How technical authority is associated with language and persona expertise.

2) Location / Cultural Bias
   - How location correlates with politeness and question-asking tone.

3) Topic / Domain Bias
   - How tech / school / emotion content is confined to one language or the other.

Input:
    final_dataset_v01.csv

Outputs (in ./section1_outputs/):
    role_expertise_bias.csv
    location_cultural_bias.csv
    topic_domain_bias_by_language.csv
    fig_role_expertise_tech_density.png
    fig_location_tone.png
    fig_topic_domain_by_language.png
"""

import ast
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_PATH = Path("data/processed/final_dataset_v01.csv")
OUT_DIR = Path("section1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# LEXICONS
#   English: matched on tokens_roman when lang == "en"
#   Hindi:   matched on tokens_dev   when lang == "hi" (Devanagari)
#   These lists are intentionally small but realistic for your domain.
# ---------------------------------------------------------------------

TECH_TERMS_EN = {
    "algorithm", "algorithms", "binary", "search", "tree", "graph",
    "bug", "bugs", "debug", "debugging",
    "deploy", "deployed", "deployment", "deploying",
    "server", "servers", "client", "clients",
    "cloud", "azure", "aws",
    "storage", "database", "databases", "sql", "nosql",
    "model", "models", "training", "inference",
    "embedding", "embeddings",
    "dataset", "datasets", "token", "tokens",
    "api", "apis", "endpoint", "endpoints",
    "code", "coding", "repo", "repository",
    "commit", "branch", "merge", "staging", "production", "prod",
    "pipeline", "notebook", "notebooks",
    "feature", "features", "ticket", "tickets", "sprint", "backlog",
}

# Hindi technical words in Devanagari that plausibly appear
TECH_TERMS_HI_DEV = {
    "डेटा",       # data
    "डाटा",       # variant
    "एल्गोरिथ्म",  # algorithm
    "एल्गोरिदम",
    "सर्वर",      # server
    "नेटवर्क",    # network
    "कोड",        # code
    "प्रोजेक्ट",   # project
    "परियोजना",   # project formal
    "पाइपलाइन",   # pipeline
    "डिप्लॉय",     # deploy (transliterated)
    "फीचर",       # feature
}

SCHOOL_TERMS_EN = {
    "exam", "exams", "midterm", "midterms", "final", "finals",
    "homework", "assignment", "assignments",
    "project", "projects",
    "grade", "grades",
    "class", "classes", "lecture", "lectures",
    "course", "courses",
    "lab", "labs", "quiz", "quizzes",
    "student", "students",
}

SCHOOL_TERMS_HI_DEV = {
    "परीक्षा",     # exam
    "इम्तिहान",    # exam informally
    "क्लास",       # class
    "कक्षा",       # class formal
    "होमवर्क",     # homework
    "असाइनमेंट",   # assignment
    "प्रोफेसर",    # professor
    "शिक्षक",      # teacher
    "छात्र",       # student
    "विद्यार्थी",   # student
}

EMOTION_TERMS_EN = {
    "stressed", "stress", "anxious", "anxiety", "worried", "worry",
    "nervous", "overwhelmed", "tired", "exhausted",
    "happy", "excited", "relieved", "sad", "upset",
    "angry", "frustrated", "scared",
}

EMOTION_TERMS_HI_DEV = {
    "तनाव",        # stress
    "टेंशन",       # tension
    "चिंता",       # worry
    "चिंतित",      # worried
    "खुश",         # happy
    "उदास",        # sad
    "गुस्सा",       # anger
    "नाराज़",      # angry / upset
    "थका",         # tired (m)
    "थकी",         # tired (f)
    "थकान",        # fatigue
}

# Politeness and cultural markers (language independent, checked in tokens_roman)
POLITE_TERMS = {
    "please", "sorry", "thanks", "thank", "appreciate",
    "apologize", "apologise", "kindly",
}

CULTURAL_TERMS = {
    "diwali", "holi", "bollywood", "chai", "namaste",
    "northeastern", "boston", "lincoln",
    "microsoft", "azure", "amazon", "aws",
    "school", "university", "professor",
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def parse_list_cell(x):
    """Safely parse list-like cells stored as strings."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return ast.literal_eval(x)


def extract_personas_from_prompt(prompt: str) -> Dict[str, str]:
    """
    Extract 'Person1:' and 'Person2:' lines from the metadata prompt text.
    Returns mapping: {"Person1": "...", "Person2": "..."} where possible.
    """
    personas = {}
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Person1:"):
            personas["Person1"] = line.replace("Person1:", "", 1).strip()
        elif line.startswith("Person2:"):
            personas["Person2"] = line.replace("Person2:", "", 1).strip()
    return personas


def classify_expertise(persona_text: str) -> str:
    """Classify persona as expert / non_expert / unknown from role text."""
    pt = persona_text.lower()
    expert_keywords = [
        "software engineer",
        "project manager",
        "algorithms professor",
        "lead software engineer",
        "azure storage team",
        "serverless analytics",
        "amazon web services",
    ]
    non_expert_keywords = [
        "student",
        "7th-grade",
        "7th grade",
        "middle school",
    ]
    if any(k in pt for k in expert_keywords):
        return "expert"
    if any(k in pt for k in non_expert_keywords):
        return "non_expert"
    return "unknown"


def classify_location(persona_text: str) -> str:
    """Classify persona into a coarse location category."""
    pt = persona_text.lower()
    if "microsoft" in pt or "amazon web services" in pt or "azure" in pt:
        return "big_tech_us"
    if "northeastern university" in pt or "boston university" in pt:
        return "us_university"
    if "middle school" in pt or "lincoln middle school" in pt:
        return "us_school"
    return "other"


# ---------------------------------------------------------------------
# LOAD AND ANNOTATE DATAFRAME
# ---------------------------------------------------------------------


def load_and_annotate(path: Path) -> pd.DataFrame:
    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    # Parse list columns
    df["tokens_roman"] = df["tokens_roman"].apply(parse_list_cell)
    df["tokens_dev"] = df["tokens_dev"].apply(parse_list_cell)
    df["token_languages"] = df["token_languages"].apply(parse_list_cell)

    # Extract personas from metadata prompts per conversation
    conv_personas: Dict[int, Dict[str, Dict[str, str]]] = {}

    for conv_id, group in df.groupby("conversation_id"):
        raw_meta = group.iloc[0]["metadata"]
        try:
            meta_list = ast.literal_eval(raw_meta)
        except Exception:
            meta_list = []

        prompt_text = ""
        if isinstance(meta_list, list) and len(meta_list) >= 2:
            prompt_text = meta_list[1]
        elif isinstance(meta_list, list) and len(meta_list) == 1:
            prompt_text = meta_list[0]

        personas = extract_personas_from_prompt(prompt_text)
        conv_personas[conv_id] = {}
        for pid, ptxt in personas.items():
            conv_personas[conv_id][pid] = {
                "persona_text": ptxt,
                "expertise": classify_expertise(ptxt),
                "location_cat": classify_location(ptxt),
            }

    # Attach expertise, location, politeness, cultural markers, topic counts
    expertise_list = []
    location_list = []
    polite_counts = []
    cultural_counts = []
    question_flags = []

    tech_en_counts = []
    tech_hi_counts = []
    school_en_counts = []
    school_hi_counts = []
    emotion_en_counts = []
    emotion_hi_counts = []

    for _, row in df.iterrows():
        conv_id = row["conversation_id"]
        speaker = row["speaker"]
        tokens_roman = [str(t).lower() for t in row["tokens_roman"]]
        tokens_dev = row["tokens_dev"]
        langs = row["token_languages"]
        text = str(row["utterance"])

        # persona attributes
        attrs = conv_personas.get(conv_id, {}).get(speaker)
        if attrs is None:
            expertise_list.append("unknown")
            location_list.append("unknown")
        else:
            expertise_list.append(attrs["expertise"])
            location_list.append(attrs["location_cat"])

        # politeness / cultural markers (checked on roman tokens)
        polite_counts.append(sum(1 for t in tokens_roman if t in POLITE_TERMS))
        cultural_counts.append(sum(1 for t in tokens_roman if t in CULTURAL_TERMS))

        # question flag
        question_flags.append(int("?" in text))

        # topic counts split by language
        te = th = se = sh = ee = eh = 0
        for tok_rom, tok_dev, lang in zip(tokens_roman, tokens_dev, langs):
            if lang == "en":
                if tok_rom in TECH_TERMS_EN:
                    te += 1
                if tok_rom in SCHOOL_TERMS_EN:
                    se += 1
                if tok_rom in EMOTION_TERMS_EN:
                    ee += 1
            elif lang == "hi":
                if tok_dev in TECH_TERMS_HI_DEV:
                    th += 1
                if tok_dev in SCHOOL_TERMS_HI_DEV:
                    sh += 1
                if tok_dev in EMOTION_TERMS_HI_DEV:
                    eh += 1

        tech_en_counts.append(te)
        tech_hi_counts.append(th)
        school_en_counts.append(se)
        school_hi_counts.append(sh)
        emotion_en_counts.append(ee)
        emotion_hi_counts.append(eh)

    df["expertise"] = expertise_list
    df["location_cat"] = location_list
    df["polite_count"] = polite_counts
    df["cultural_count"] = cultural_counts
    df["is_question"] = question_flags

    df["tech_terms_en"] = tech_en_counts
    df["tech_terms_hi"] = tech_hi_counts
    df["school_terms_en"] = school_en_counts
    df["school_terms_hi"] = school_hi_counts
    df["emotion_terms_en"] = emotion_en_counts
    df["emotion_terms_hi"] = emotion_hi_counts

    return df


# ---------------------------------------------------------------------
# ROLE / EXPERTISE BIAS
# ---------------------------------------------------------------------


def analyze_role_expertise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for expertise, group in df.groupby("expertise"):
        en_tokens = int(group["num_english_tokens"].sum())
        hi_tokens = int(group["num_hindi_tokens"].sum())
        total = en_tokens + hi_tokens

        tech_en = int(group["tech_terms_en"].sum())
        tech_hi = int(group["tech_terms_hi"].sum())

        rows.append({
            "expertise": expertise,
            "utterances": int(group["utterance_id"].count()),
            "en_tokens": en_tokens,
            "hi_tokens": hi_tokens,
            "en_token_ratio": en_tokens / total if total else 0.0,
            "hi_token_ratio": hi_tokens / total if total else 0.0,
            "tech_terms_en": tech_en,
            "tech_terms_hi": tech_hi,
            "tech_en_per_1k": 1000 * tech_en / en_tokens if en_tokens else 0.0,
            "tech_hi_per_1k": 1000 * tech_hi / hi_tokens if hi_tokens else 0.0,
        })
    role_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "role_expertise_bias.csv"
    role_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return role_df


def plot_role_expertise(role_df: pd.DataFrame):
    """
    Figure for Role / Expertise bias:
    shows tech terms per 1k tokens in English vs Hindi
    for each expertise group.
    """
    x = role_df["expertise"].tolist()
    tech_en = role_df["tech_en_per_1k"].tolist()
    tech_hi = role_df["tech_hi_per_1k"].tolist()

    idx = range(len(x))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in idx], tech_en, width,
           label="English tech terms / 1k tokens")
    ax.bar([i + width / 2 for i in idx], tech_hi, width,
           label="Hindi tech terms / 1k tokens")

    ax.set_xticks(list(idx))
    ax.set_xticklabels(x, rotation=15)
    ax.set_ylabel("Tech terms per 1k tokens")
    ax.set_title("Technical authority by expertise and language")
    ax.legend()
    fig.tight_layout()

    fig_path = OUT_DIR / "fig_role_expertise_tech_density.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_path}")


# ---------------------------------------------------------------------
# LOCATION / CULTURAL BIAS
# ---------------------------------------------------------------------


def analyze_location_cultural(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate per persona (conversation_id + speaker)
    persona_df = df.groupby(
        ["conversation_id", "speaker", "location_cat", "expertise"]
    ).agg(
        utterances=("utterance_id", "count"),
        polite_instances=("polite_count", "sum"),
        cultural_instances=("cultural_count", "sum"),
        questions=("is_question", "sum"),
    ).reset_index()

    persona_df["polite_per_utt"] = persona_df["polite_instances"] / persona_df["utterances"].replace(0, 1)
    persona_df["cultural_per_utt"] = persona_df["cultural_instances"] / persona_df["utterances"].replace(0, 1)
    persona_df["question_rate"] = persona_df["questions"] / persona_df["utterances"].replace(0, 1)

    loc_df = persona_df.groupby("location_cat").agg(
        personas=("speaker", "count"),
        avg_polite_per_utt=("polite_per_utt", "mean"),
        avg_cultural_per_utt=("cultural_per_utt", "mean"),
        avg_question_rate=("question_rate", "mean"),
    ).reset_index()

    out_path = OUT_DIR / "location_cultural_bias.csv"
    loc_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return loc_df


def plot_location_tone(loc_df: pd.DataFrame):
    """
    Figure for Location / Cultural bias:
    shows politeness and question rate by location category.
    """
    x = loc_df["location_cat"].tolist()
    polite = loc_df["avg_polite_per_utt"].tolist()
    questions = loc_df["avg_question_rate"].tolist()

    idx = range(len(x))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in idx], polite, width,
           label="Polite markers per utterance")
    ax.bar([i + width / 2 for i in idx], questions, width,
           label="Question rate")

    ax.set_xticks(list(idx))
    ax.set_xticklabels(x, rotation=15)
    ax.set_ylabel("Average per utterance")
    ax.set_title("Tone by location category")
    ax.legend()
    fig.tight_layout()

    fig_path = OUT_DIR / "fig_location_tone.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_path}")


# ---------------------------------------------------------------------
# TOPIC / DOMAIN BIAS
# ---------------------------------------------------------------------


def analyze_topic_domain(df: pd.DataFrame) -> pd.DataFrame:
    total_en_tokens = int(df["num_english_tokens"].sum())
    total_hi_tokens = int(df["num_hindi_tokens"].sum())

    tech_en = int(df["tech_terms_en"].sum())
    tech_hi = int(df["tech_terms_hi"].sum())
    school_en = int(df["school_terms_en"].sum())
    school_hi = int(df["school_terms_hi"].sum())
    emotion_en = int(df["emotion_terms_en"].sum())
    emotion_hi = int(df["emotion_terms_hi"].sum())

    rows = []
    for lang, total, tech, school, emotion in [
        ("en", total_en_tokens, tech_en, school_en, emotion_en),
        ("hi", total_hi_tokens, tech_hi, school_hi, emotion_hi),
    ]:
        rows.append({
            "language": lang,
            "total_tokens": total,
            "tech_terms": tech,
            "school_terms": school,
            "emotion_terms": emotion,
            "tech_per_1k": 1000 * tech / total if total else 0.0,
            "school_per_1k": 1000 * school / total if total else 0.0,
            "emotion_per_1k": 1000 * emotion / total if total else 0.0,
        })

    topic_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "topic_domain_bias_by_language.csv"
    topic_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return topic_df


def plot_topic_domain(topic_df: pd.DataFrame):
    """
    Figure for Topic / Domain bias:
    shows tech, school, and emotion term densities
    per 1k tokens for English vs Hindi.
    """
    langs = topic_df["language"].tolist()
    tech = topic_df["tech_per_1k"].tolist()
    school = topic_df["school_per_1k"].tolist()
    emotion = topic_df["emotion_per_1k"].tolist()

    idx = range(len(langs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width for i in idx], tech, width, label="Tech / 1k tokens")
    ax.bar(idx, school, width, label="School / 1k tokens")
    ax.bar([i + width for i in idx], emotion, width, label="Emotion / 1k tokens")

    ax.set_xticks(list(idx))
    ax.set_xticklabels(langs)
    ax.set_ylabel("Terms per 1k tokens")
    ax.set_title("Topic term density by language")
    ax.legend()
    fig.tight_layout()

    fig_path = OUT_DIR / "fig_topic_domain_by_language.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    df = load_and_annotate(DATA_PATH)

    # 1) Role / Expertise Bias
    role_df = analyze_role_expertise(df)
    plot_role_expertise(role_df)

    # 2) Location / Cultural Bias
    loc_df = analyze_location_cultural(df)
    plot_location_tone(loc_df)

    # 3) Topic / Domain Bias
    topic_df = analyze_topic_domain(df)
    plot_topic_domain(topic_df)

    print("\nDone. All outputs written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
