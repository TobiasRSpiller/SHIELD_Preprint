#!/usr/bin/env python3

import json
import os
import csv
from pathlib import Path

def load_conversation_data(input_dir):
    """Load all JSON conversation files from the input directory."""
    conversations = []
    input_path = Path(input_dir)
    
    for json_file in input_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return conversations

def format_conversation(conversation_history):
    """Format conversation with clear user/assistant indicators in single column."""
    formatted_parts = []
    
    for turn in conversation_history:
        role = turn.get('role', '').lower()
        text = turn.get('text', '').strip()
        
        if role == 'user' and text:
            formatted_parts.append(f"[USER]: {text}")
        elif role == 'assistant' and text:
            formatted_parts.append(f"[ASSISTANT]: {text}")
    
    return "\n\n".join(formatted_parts)

def prepare_annotation_data(input_dir, output_file):
    """Prepare conversation data for annotation."""
    # Load all conversations
    conversations = load_conversation_data(input_dir)
    
    # Prepare output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If file exists, add timestamp identifier
    if output_path.exists():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        suffix = output_path.suffix
        new_filename = f"{stem}_{timestamp}{suffix}"
        output_file = output_path.parent / new_filename
        print(f"Output file exists. Saving as: {output_file}")
    
    # Create CSV file for annotation
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with all relevant columns
        writer.writerow(['conversation_id', 'generation_model', 'temperature', 'max_tokens', 'prompt_template_id', 'appropriateness', 'conversation'])
        
        # Process each conversation
        for conv in conversations:
            conversation_id = conv.get('conversation_id', '')
            generation_model = conv.get('generation_model', '')
            conversation_history = conv.get('conversation_history', [])
            
            # Extract generation parameters
            metadata = conv.get('metadata', {})
            gen_params = metadata.get('generation_parameters', {})
            temperature = gen_params.get('temperature', '')
            max_tokens = gen_params.get('max_tokens', '')
            
            # **FIXED LINE:** Use 'prompt_template_id' from JSON
            prompt_id = conv.get('prompt_template_id', '') 
            
            # This field does not seem to exist in the example JSON, but leaving the logic in
            appropriateness = conv.get('appropriateness', '')
            
            # Format conversation with clear indicators
            formatted_conversation = format_conversation(conversation_history)
            
            # Write to CSV
            writer.writerow([conversation_id, generation_model, temperature, max_tokens, prompt_id, appropriateness, formatted_conversation])
    
    print(f"Prepared {len(conversations)} conversations for annotation")
    print(f"Output saved to: {output_file}")

def main():
    # Define paths
    input_dir = "data/01_raw_generations"
    output_file = "data/02_for_annotation/for_annotation.csv"
    
    # Prepare annotation data
    prepare_annotation_data(input_dir, output_file)

if __name__ == "__main__":
    main()