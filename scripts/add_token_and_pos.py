import pandas as pd
import json
from pathlib import Path
import spacy
from collections import Counter

# Load spaCy models
print("Loading spaCy models...")
nlp_en = spacy.load('en_core_web_sm')  # English model
nlp_hi = spacy.load('hi_core_news_sm')  # Hindi model
print("Models loaded successfully!\n")

def detect_word_language_spacy(word):
    """
    Detect language using spaCy models.
    Returns 'hi' for Hindi, 'en' for English.
    """
    # Try Hindi model first
    doc_hi = nlp_hi(word)
    # Try English model
    doc_en = nlp_en(word)
    
    # If Hindi model recognizes it (not all tokens are 'X' POS), likely Hindi
    hi_confidence = sum(1 for token in doc_hi if token.pos_ != 'X')
    en_confidence = sum(1 for token in doc_en if token.pos_ != 'X')
    
    # Return language with higher confidence
    if hi_confidence > en_confidence:
        return 'hi'
    else:
        return 'en'

def tokenize_and_tag_spacy(text):
    """
    Tokenize, lemmatize, and POS tag using spaCy.
    Returns: tokens, lemmas, pos_tags, languages
    """
    # Process with English model first (assuming mixed text)
    doc = nlp_en(text)
    
    # Filter out punctuation and whitespace
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    pos_tags = [token.pos_ for token in doc if not token.is_punct and not token.is_space]
    
    if not tokens:
        return [], [], [], []
    
    # Detect language for each token
    languages = []
    for token_text in tokens:
        lang = detect_word_language_spacy(token_text)
        languages.append(lang)
    
    return tokens, lemmas, pos_tags, languages

def process_utterances(input_path, output_path):
    """
    Add tokenization, lemmatization, and POS tagging to utterances using spaCy.
    """
    print(f"Loading data from {input_path}...")
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path)
    
    print(f"Loaded {len(df)} rows")
    print("\nProcessing utterances with spaCy...")
    
    tokens_list = []
    lemmas_list = []
    pos_list = []
    languages_list = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)}...")
        
        utterance = str(row['utterance'])
        
        # Use spaCy for processing
        tokens, lemmas, pos_tags, langs = tokenize_and_tag_spacy(utterance)
        
        tokens_list.append(tokens)
        lemmas_list.append(lemmas)
        pos_list.append(pos_tags)
        languages_list.append(langs)
    
    # Add new columns (rest remains the same)
    df['tokens'] = tokens_list
    df['lemmas'] = lemmas_list
    df['pos_tags'] = pos_list
    df['token_languages'] = languages_list
    
    # Add statistics
    df['num_tokens'] = df['tokens'].apply(len)
    df['num_english_tokens'] = df['token_languages'].apply(lambda x: x.count('en'))
    df['num_hindi_tokens'] = df['token_languages'].apply(lambda x: x.count('hi'))
    
    df['is_code_switched'] = df['num_hindi_tokens'] > 0
    df['hindi_ratio'] = df.apply(lambda row: row['num_hindi_tokens'] / row['num_tokens'] if row['num_tokens'] > 0 else 0, axis=1)
    
    # Save
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Print statistics (same as before)
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total utterances: {len(df)}")
    print(f"Code-switched utterances: {df['is_code_switched'].sum()} ({df['is_code_switched'].sum()/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Use your actual paths
    input_path = "data/processed/codeswitch_v01.csv"
    output_path = "data/processed/codeswitch_v01_with_pos.csv"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    df = process_utterances(input_path, output_path)
    
    print("\n✓ Processing complete!")
    print(f"Output saved to: {output_path}")
    hindi_count = len(df[df["hindi_ratio"] > 0])
    total_count = len(df)

    print(total_count)
    print(hindi_count)