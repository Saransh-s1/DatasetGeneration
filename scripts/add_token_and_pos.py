import pandas as pd
import json
from pathlib import Path
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK data automatically
print("Downloading required NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded successfully!\n")
except:
    print("Note: Some NLTK downloads may have failed, but will continue...\n")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Common English words (must check BEFORE Hindi)
ENGLISH_WORDS = {
    # Articles & Determiners
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her',
    'its', 'our', 'their', 'some', 'any', 'each', 'every', 'no', 'none', 'much', 'many',
    
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'who', 'whom', 'whose', 'which', 'what', 'myself', 'yourself', 'himself', 'herself',
    'itself', 'ourselves', 'themselves',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over',
    'out', 'off', 'down', 'near', 'across', 'along', 'around', 'behind', 'beside',
    
    # Conjunctions
    'and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'although', 'though',
    'unless', 'since', 'until', 'than', 'as', 'nor', 'yet',
    
    # Common verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'can', 'shall', 'go', 'get', 'make', 'know', 'think', 'take', 'see', 'come',
    'want', 'use', 'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need',
    'feel', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin', 'seem',
    'help', 'talk', 'turn', 'start', 'show', 'hear', 'play', 'run', 'move', 'like',
    'live', 'believe', 'bring', 'happen', 'write', 'sit', 'stand', 'lose', 'pay',
    'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
    
    # Common adjectives
    'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old',
    'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young',
    'important', 'few', 'public', 'bad', 'same', 'able', 'best', 'better', 'sure',
    'clear', 'major', 'sorry', 'real', 'happy', 'free', 'ready', 'simple', 'late',
    'hard', 'full', 'easy', 'strong', 'possible', 'whole', 'difficult', 'busy',
    
    # Common adverbs
    'not', 'now', 'just', 'very', 'still', 'also', 'here', 'there', 'then', 'well',
    'only', 'even', 'back', 'more', 'how', 'too', 'again', 'really', 'most', 'already',
    'always', 'never', 'often', 'sometimes', 'today', 'tomorrow', 'yesterday', 'away',
    'enough', 'quite', 'almost', 'soon', 'probably', 'maybe', 'perhaps', 'together',
    
    # Common nouns
    'time', 'year', 'people', 'way', 'day', 'man', 'thing', 'woman', 'life', 'child',
    'world', 'school', 'state', 'family', 'student', 'group', 'country', 'problem',
    'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question',
    'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother',
    'area', 'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'eye',
    'job', 'word', 'business', 'issue', 'side', 'kind', 'head', 'house', 'service',
    'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car',
    'city', 'community', 'name', 'president', 'team', 'minute', 'idea', 'kid', 'body',
    'information', 'project', 'class', 'meeting', 'test', 'assignment', 'exam',
    
    # Question words
    'where', 'why', 'how',
    
    # Other common words
    'yes', 'no', 'ok', 'okay', 'thanks', 'thank', 'please', 'hello', 'hi', 'bye',
    'sorry', 'excuse', 'welcome',
}

# Comprehensive list of Romanized Hindi words
HINDI_WORDS = {
    # Verbs
    'hai', 'hain', 'tha', 'the', 'thi', 'ho', 'hoon', 'hoga', 'hogi', 'honge',
    'kar', 'karna', 'karte', 'kiya', 'kiye', 'karenge', 'karunga', 'karegi',
    'rahe', 'raha', 'rahi', 'sakta', 'sakti', 'sakte', 'chahiye', 'padega',
    'aana', 'aaye', 'gaya', 'gayi', 'gaye', 'lena', 'dena', 'dekha', 'dekhi',
    'dete', 'lete', 'diya', 'liya', 'karo', 'karke',
    
    # Pronouns
    'main', 'mujhe', 'mere', 'mera', 'meri', 'tumhe', 'tumhara', 'tumhari',
    'usse', 'usne', 'uska', 'uski', 'maine', 'humne', 'tumne', 'unhone',
    'iske', 'uske', 'jinke', 'jiske', 'hum', 'tum', 'woh', 'yeh',
    
    # Postpositions
    'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 'pe', 'tak', 'liye',
    
    # Conjunctions
    'aur', 'ya', 'lekin', 'par', 'kyunki', 'ki', 'agar', 'toh', 'to',
    
    # Question words
    'kya', 'kaise', 'kyun', 'kab', 'kahan', 'kaun', 'kaunsa', 'kitna',
    'kitne', 'kitni', 'kaunse',
    
    # Adverbs of place/time
    'yahan', 'wahan', 'yahaan', 'wahaan', 'kal', 'aaj', 'abhi', 'phir',
    'kabhi', 'jab', 'tab', 'subah', 'raat', 'shaam', 'din', 'dopahar',
    
    # Quantifiers/Adjectives
    'bahut', 'thoda', 'zyada', 'kam', 'kuch', 'sab', 'sabhi', 'koi',
    'bilkul', 'ekdum', 'itna', 'utna', 'jitna', 'itne', 'utne',
    
    # Common nouns
    'log', 'dost', 'ghar', 'shahar', 'desh', 'duniya', 'samay', 'waqt',
    'kaam', 'baat', 'cheez', 'jagah', 'tarah', 'tarike', 'baar',
    
    # Adjectives
    'accha', 'achha', 'acha', 'bura', 'khush', 'udaas', 'pyaar', 'naraz',
    'sahi', 'galat', 'naya', 'purana', 'bada', 'chota', 'lambi', 'chhoti',
    
    # Common words
    'nahi', 'nahin', 'haan', 'ji', 'zaroorat', 'pasand', 'matlab',
    'samajh', 'pata', 'yaad', 'bhi', 'hi', 'na', 'bola', 'kaha',
    
    # Additional common words
    'baad', 'pehle', 'saath', 'sath', 'dhyan', 'pareshaan',
    'padhna', 'likhna', 'bolna', 'sunna', 'dekhna', 'banana',
    'jaana', 'aana', 'milna', 'khana', 'peena', 'sona', 'uthna',
    'beta', 'ma', 'bhai', 'didi', 'sir', 'madam',
}

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tag to WordNet POS tag for lemmatization"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default

def detect_word_language(word):
    """
    Detect language for Romanized Hindi/English code-switching.
    Returns 'hi' for Hindi, 'en' for English.
    IMPORTANT: Check English FIRST to avoid false Hindi matches.
    """
    # Convert to lowercase
    clean_word = word.lower()
    
    # Check English words FIRST (priority)
    if clean_word in ENGLISH_WORDS:
        return 'en'
    
    # Then check if it's in Hindi word list
    if clean_word in HINDI_WORDS:
        return 'hi'
    
    # Default to English for unknown words
    return 'en'

def tokenize_and_tag(text):
    """
    Tokenize, lemmatize, and POS tag text (excluding punctuation).
    Returns: tokens, lemmas, pos_tags, languages
    """
    # Tokenize using NLTK
    all_tokens = word_tokenize(text)
    
    # Filter out punctuation
    tokens = [token for token in all_tokens if token.isalnum()]
    
    if not tokens:
        return [], [], [], []
    
    # POS tag using NLTK (need original tags for lemmatization)
    pos_tags_detailed = pos_tag(tokens)
    
    # Get universal POS tags
    pos_tags_universal = pos_tag(tokens, tagset='universal')
    pos_tags = [pos for word, pos in pos_tags_universal]
    
    # Lemmatize each token
    lemmas = []
    for token, (word, pos_detail) in zip(tokens, pos_tags_detailed):
        wordnet_pos = get_wordnet_pos(pos_detail)
        lemma = lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
        lemmas.append(lemma)
    
    # Detect language for each token (using lemmas for better accuracy)
    languages = []
    for lemma in lemmas:
        lang = detect_word_language(lemma)
        languages.append(lang)
    
    return tokens, lemmas, pos_tags, languages

def process_utterances(input_path, output_path):
    """
    Add tokenization, lemmatization, and POS tagging to utterances.
    """
    print(f"Loading data from {input_path}...")
    
    # Handle both CSV and JSON
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path)
    
    print(f"Loaded {len(df)} rows")
    
    print("\nProcessing utterances (tokenizing, lemmatizing, POS tagging)...")
    
    tokens_list = []
    lemmas_list = []
    pos_list = []
    languages_list = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)}...")
        
        utterance = str(row['utterance'])
        
        # Tokenize, lemmatize, and tag
        tokens, lemmas, pos_tags, langs = tokenize_and_tag(utterance)
        
        tokens_list.append(tokens)
        lemmas_list.append(lemmas)
        pos_list.append(pos_tags)
        languages_list.append(langs)
    
    # Add new columns
    df['tokens'] = tokens_list
    df['lemmas'] = lemmas_list
    df['pos_tags'] = pos_list
    df['token_languages'] = languages_list
    
    # Add statistics
    df['num_tokens'] = df['tokens'].apply(len)
    df['num_english_tokens'] = df['token_languages'].apply(lambda x: x.count('en'))
    df['num_hindi_tokens'] = df['token_languages'].apply(lambda x: x.count('hi'))
    
    # Calculate code-switching metrics
    df['is_code_switched'] = df['num_hindi_tokens'] > 0
    df['hindi_ratio'] = df.apply(lambda row: row['num_hindi_tokens'] / row['num_tokens'] if row['num_tokens'] > 0 else 0, axis=1)
    
    # Save as CSV only
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total utterances: {len(df)}")
    print(f"Code-switched utterances: {df['is_code_switched'].sum()} ({df['is_code_switched'].sum()/len(df)*100:.1f}%)")
    print(f"Average tokens per utterance: {df['num_tokens'].mean():.2f}")
    print(f"Average English tokens: {df['num_english_tokens'].mean():.2f}")
    print(f"Average Hindi tokens: {df['num_hindi_tokens'].mean():.2f}")
    print(f"Average Hindi ratio: {df['hindi_ratio'].mean():.2%}")
    
    # Print sample
    print("\n" + "="*60)
    print("SAMPLE OUTPUT")
    print("="*60)
    for i in range(min(5, len(df))):
        if df.iloc[i]['num_tokens'] > 0:
            print(f"\nUtterance {i + 1}:")
            print(f"Text: {df.iloc[i]['utterance'][:80]}...")
            print(f"Tokens: {df.iloc[i]['tokens'][:10]}")
            print(f"Lemmas: {df.iloc[i]['lemmas'][:10]}")
            print(f"POS Tags: {df.iloc[i]['pos_tags'][:10]}")
            print(f"Languages: {df.iloc[i]['token_languages'][:10]}")
            print(f"Hindi: {df.iloc[i]['num_hindi_tokens']}, English: {df.iloc[i]['num_english_tokens']}")
    
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