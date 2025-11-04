import pandas as pd
import json
import re
from pathlib import Path

def split_utterances(conversation_text):
    """
    Split conversation into individual utterances (sentences per speaker).
    Returns list of dicts with speaker and utterance.
    """
    utterances = []
    
    # Split by newlines to get each speaker turn
    lines = conversation_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line has speaker format (Speaker: text or person1: text)
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                
                # Split the text into sentences (by period)
                # Handle multiple periods, question marks, exclamation marks
                sentences = re.split(r'[.!?]+', text)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:  # Only add non-empty sentences
                        utterances.append({
                            'speaker': speaker,
                            'utterance': sentence
                        })
    
    return utterances

def process_codeswitch_data(input_json_path, output_json_path):
    """
    Process code-switched conversation data and create one row per utterance.
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to save output JSON file
    """
    # Load the data
    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} conversations")
    
    # Process each conversation
    all_rows = []
    
    for conv_idx, conversation in enumerate(data):
        print(f"Processing conversation {conv_idx + 1}/{len(data)}...")
        
        # Extract metadata
        generation_strategy = conversation.get('generation_strategy', '')
        source = conversation.get('source', '')
        metadata = conversation.get('metadata', [])
        conv_id = conversation.get('id', conv_idx)
        original_utterance = conversation.get('utterance', '')
        
        # Split into individual utterances
        utterances = split_utterances(original_utterance)
        
        # Create a row for each utterance
        for utt_idx, utt in enumerate(utterances):
            row = {
                'conversation_id': conv_id,
                'utterance_id': f"{conv_id}_{utt_idx}",
                'speaker': utt['speaker'],
                'utterance': utt['utterance'],
                'generation_strategy': generation_strategy,
                'source': source,
                'metadata': metadata
            }
            all_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)

    
    print(f"\nCreated {len(df)} utterance rows from {len(data)} conversations")
    print(f"Average utterances per conversation: {len(df)/len(data):.2f}")
    
    # Save to JSON
    print(f"\nSaving to {output_json_path}...")
    df.to_json(output_json_path, orient='records', indent=2, force_ascii=False)
    
    # Also save as CSV for easier viewing
    csv_path = output_json_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Also saved as CSV: {csv_path}")
    
    # Print sample
    print("\n=== Sample of processed data ===")
    print(df.head(10).to_string())
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total conversations: {len(data)}")
    print(f"Total utterances: {len(df)}")
    print(f"Utterances per conversation: {df.groupby('conversation_id').size().describe()}")
    print(f"\nGeneration strategies:")
    print(df['generation_strategy'].value_counts())
    print(f"\nSpeakers:")
    print(df['speaker'].value_counts())
    
    return df

# Example usage
if __name__ == "__main__":
    # Set your paths here
    input_path = "data/raw/codeswitch_v02.json"
    output_path = "data/processed/codeswitch_by_sentences_v02.csv"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Process the data
    df = process_codeswitch_data(input_path, output_path)
    
    print("\n✓ Processing complete!")