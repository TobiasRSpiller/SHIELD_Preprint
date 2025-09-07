import os
import json
import uuid
import time
import logging
import argparse
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import litellm

# --- Configuration Section ---

# Set up basic logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scripts/generation.log"),
        logging.StreamHandler()
    ]
)

# 1. List of LLM models to use
MODELS_TO_USE = [
    "gpt-4.1-2025-04-14", 
    "gemini/gemma-3-1b-it",
    "groq/moonshotai/kimi-k2-instruct",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "claude-sonnet-4-20250514" 
]

# 2. Generation parameters
GENERATION_CONFIG = {
    "temperature": 0.5,
    "max_tokens": 500,
    "timeout": 30
}

# 3. File paths
PROMPT_TEMPLATES_PATH = "scripts/prompt_templates.csv"
OUTPUT_DIR = "data/01_raw_generations"
# --- End of Configuration Section ---


def setup_environment():
    """Load environment variables and create output directory."""
    load_dotenv()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    logging.info("Environment setup complete. API keys loaded.")

def load_prompts(file_path):
    """Load prompt templates from a CSV file."""
    try:
        prompts_df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {len(prompts_df)} prompts from {file_path}.")
        return prompts_df.to_dict('records')
    except FileNotFoundError:
        logging.error(f"Prompt file not found at {file_path}. Exiting.")
        exit()

def get_output_filepath(model, prompt_id):
    """Generate consistent filepath for a given model and prompt combination."""
    safe_model_name = model.replace('/', '-')
    filename = f"{safe_model_name}_{prompt_id}.json"
    return os.path.join(OUTPUT_DIR, filename)

def conversation_exists(model, prompt_id):
    """Check if conversation already exists for this model/prompt combination."""
    filepath = get_output_filepath(model, prompt_id)
    return os.path.exists(filepath)

def get_checkpoint_status(prompt_templates):
    """Analyze current checkpoint status and return statistics."""
    total_combinations = len(prompt_templates) * len(MODELS_TO_USE)
    completed = 0
    
    for model in MODELS_TO_USE:
        for prompt_info in prompt_templates:
            if conversation_exists(model, prompt_info['tag']):
                completed += 1
    
    return completed, total_combinations

def generate_conversation(model, prompt_text):
    """Generates a single-turn conversation with robust error handling and retries."""
    messages = [{"role": "user", "content": prompt_text}]
    try:
        logging.info(f"  Sending prompt to {model}...")
        response = litellm.completion(
            model=model,
            messages=messages,
            **GENERATION_CONFIG,
            num_retries=3
        )
        return response
    except litellm.AuthenticationError as e:
        logging.error(f"  !! AUTHENTICATION ERROR for {model}. Check your API key. Details: {e}")
    except litellm.InvalidRequestError as e:
        logging.error(f"  !! INVALID REQUEST ERROR for {model}. Check prompt or parameters. Details: {e}")
    except litellm.ContentPolicyViolationError as e:
        logging.warning(f"  !! CONTENT POLICY VIOLATION for {model}. The model refused to respond. Details: {e}")
        return e
    except Exception as e:
        logging.error(f"  !! UNEXPECTED ERROR for {model}: {type(e).__name__}. Details: {e}")
    
    return None

def save_conversation(response, model, prompt_id, prompt_text, prompt_info):
    """Saves the generated conversation or error state to a JSON file."""
    conversation_id = str(uuid.uuid4())
    
    if response is None:
        assistant_response_text = "ERROR: No response from model due to API failure."
        metadata = {"error": "Generation function returned None, check logs for details."}
    elif isinstance(response, litellm.ContentPolicyViolationError):
        assistant_response_text = "ERROR: Content Policy Violation"
        metadata = {"base_model_safety_filter_triggered": True, "error_details": str(response)}
    else:
        assistant_response_text = response.choices[0].message.content or ""
        metadata = {
            "base_model_safety_filter_triggered": response.choices[0].finish_reason != 'stop',
            "generation_parameters": GENERATION_CONFIG,
            "raw_response_object": response.model_dump()
        }

    output_data = {
        "conversation_id": conversation_id,
        "generation_model": model,
        "prompt_template_id": prompt_id,
        "tag": prompt_info.get('tag', ''),
        "appropriateness": prompt_info.get('appropriateness', ''),
        "generation_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata,
        "conversation_history": [
            {"turn": 1, "role": "user", "text": prompt_text},
            {"turn": 2, "role": "assistant", "text": assistant_response_text}
        ]
    }

    filepath = get_output_filepath(model, prompt_id)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"  -> Successfully saved to {filepath}")
    except Exception as e:
        logging.error(f"  !! ERROR: Could not write to file {filepath}: {e}")

def main():
    """Main execution loop for data generation."""
    parser = argparse.ArgumentParser(description='Generate SHIELD benchmark conversations')
    parser.add_argument('--force', action='store_true', 
                       help='Force regeneration of all conversations (ignore checkpoint)')
    parser.add_argument('--status', action='store_true',
                       help='Show checkpoint status and exit')
    args = parser.parse_args()
    
    logging.info("--- Starting SHIELD Benchmark Data Generation ---")
    setup_environment()
    prompt_templates = load_prompts(PROMPT_TEMPLATES_PATH)
    
    # Show checkpoint status
    completed, total = get_checkpoint_status(prompt_templates)
    logging.info(f"Checkpoint Status: {completed}/{total} conversations completed ({completed/total*100:.1f}%)")
    
    if args.status:
        print(f"Checkpoint Status: {completed}/{total} conversations completed ({completed/total*100:.1f}%)")
        return
    
    if completed == total and not args.force:
        logging.info("All conversations already generated. Use --force to regenerate.")
        return
    
    if args.force:
        logging.info("Force mode enabled - will regenerate existing conversations")
    
    current_generation = 0
    skipped = 0

    for model in MODELS_TO_USE:
        logging.info(f"\nProcessing model: {model}")
        for prompt_info in prompt_templates:
            current_generation += 1
            prompt_id = prompt_info['tag']
            prompt_text = prompt_info['query']
            
            # Check checkpoint unless force mode is enabled
            if not args.force and conversation_exists(model, prompt_id):
                skipped += 1
                logging.info(f" ({current_generation}/{total}) SKIPPED (exists): {model} x {prompt_id}")
                continue
            
            logging.info(f" ({current_generation}/{total}) Generating for: {model} x {prompt_id}")
            api_response = generate_conversation(model, prompt_text)
            save_conversation(api_response, model, prompt_id, prompt_text, prompt_info)
            
            # Apply model-specific rate limiting to avoid API errors.
            # Use a 1s delay for Claude and a 0.5s delay for other models.
            if 'claude' in model:
                time.sleep(1)
            else:
                time.sleep(0.5)

    if skipped > 0:
        logging.info(f"\n--- Generation Complete: {skipped} conversations skipped (already existed) ---")
    else:
        logging.info("\n--- Data Generation Complete ---")

if __name__ == "__main__":
    main()