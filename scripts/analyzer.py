"""
SHIELD Analysis Library

Core functions for running SHIELD analyses on conversation data.
This module contains all the analysis logic but doesn't execute anything by itself.
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import litellm
from tqdm import tqdm
import time
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class ShieldAnalyzer:
    """Core analyzer class that handles SHIELD analysis operations"""
    
    def __init__(self, config_path="config.yml"):
        """Initialize analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Ensure directories exist (skip files)
        for path_key, path_value in self.config['paths'].items():
            if path_key != 'annotations':  # Skip the annotations file path
                Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def load_system_prompt(self, prompt_filename):
        """Load system prompt text from file"""
        filepath = Path(self.config['paths']['prompts']) / prompt_filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found: {filepath}")
    
    def load_conversation_from_json(self, filepath):
        """Load a single conversation from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_conversation_for_shield(self, conversation_data):
        """Format conversation history into string for SHIELD analysis"""
        formatted_text = ""
        for turn in conversation_data['conversation_history']:
            role = "User" if turn['role'] == 'user' else "Assistant"
            formatted_text += f"{role}: {turn['text']}\n\n"
        return formatted_text.strip()
    
    def call_shield_api(self, conversation_text, system_prompt, model):
        """Call LLM API using litellm framework with SHIELD prompt"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ]
            
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=self.config['api_defaults']['max_tokens'],
                temperature=self.config['api_defaults']['temperature']
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error with {model}: {e}")
            return "ERROR"
    
    def process_shield_response(self, response):
        """Parse SHIELD's response into structured data"""
        shield_data = {
            'shield_intervened': None,
            'shield_response': response,
            'intervention_type': None
        }
        
        # Check for intervention string first, as it's more specific
        if "Hey! Keep in mind, this is just an AI." in response:
            shield_data['shield_intervened'] = True
            shield_data['intervention_type'] = 'boundary_reminder'
        elif "[NO INTERVENTION]" in response:
            shield_data['shield_intervened'] = False
        else:
            # Handle unexpected responses or errors
            shield_data['shield_intervened'] = None
            shield_data['intervention_type'] = 'parse_error'
        
        return shield_data
    
    def get_filename_parts(self, analysis_config):
        """Generate consistent filename parts for checkpoints and completed files"""
        prompt_short = analysis_config['prompt'].replace('.txt', '')
        model_short = analysis_config['model'].replace('/', '_').replace('-', '_')
        analysis_type = analysis_config['type']
        
        return {
            'analysis_type': analysis_type,
            'prompt_short': prompt_short,
            'model_short': model_short
        }
    
    def get_checkpoint_filename(self, analysis_config):
        """Generate checkpoint filename"""
        parts = self.get_filename_parts(analysis_config)
        filename = f"checkpoint_{parts['analysis_type']}_{parts['prompt_short']}_{parts['model_short']}.csv"
        return Path(self.config['paths']['checkpoints']) / filename
    
    def get_completed_filename(self, analysis_config):
        """Generate completed analysis filename"""
        parts = self.get_filename_parts(analysis_config)
        
        if "main" in parts['analysis_type'].lower():
            filename = f"main_analysis_{parts['prompt_short']}_{parts['model_short']}.csv"
        elif "prompt" in parts['analysis_type'].lower():
            filename = f"prompt_sensitivity_{parts['prompt_short']}_{parts['model_short']}.csv"
        elif "model" in parts['analysis_type'].lower():
            filename = f"model_sensitivity_{parts['prompt_short']}_{parts['model_short']}.csv"
        else:
            filename = f"{parts['analysis_type']}_{parts['prompt_short']}_{parts['model_short']}.csv"
        
        return Path(self.config['paths']['completed']) / filename
    
    def is_analysis_completed(self, analysis_config):
        """Check if analysis is already completed"""
        completed_file = self.get_completed_filename(analysis_config)
        return completed_file.exists(), completed_file
    
    def load_checkpoint(self, analysis_config):
        """Load checkpoint if it exists"""
        checkpoint_file = self.get_checkpoint_filename(analysis_config)
        
        if checkpoint_file.exists():
            try:
                checkpoint_df = pd.read_csv(checkpoint_file)
                processed_ids = set(checkpoint_df['conversation_id'].tolist())
                print(f"📂 Loaded checkpoint with {len(processed_ids)} processed conversations")
                return checkpoint_df, processed_ids
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                return None, set()
        return None, set()
    
    def save_checkpoint(self, results, analysis_config):
        """Save current results as checkpoint"""
        checkpoint_file = self.get_checkpoint_filename(analysis_config)
        try:
            if results:
                pd.DataFrame(results).to_csv(checkpoint_file, index=False)
        except Exception as e:
            print(f"⚠️ Error saving checkpoint: {e}")
    
    def get_rate_limit_delay(self, model):
        """Get appropriate rate limit delay for model"""
        rate_limits = self.config['api_defaults']['rate_limits']
        
        if "claude" in model.lower():
            return rate_limits['claude']
        elif "gpt" in model.lower():
            return rate_limits['gpt']
        elif "groq" in model.lower():
            return rate_limits['groq']
        else:
            return rate_limits['default']
    
    def run_single_analysis(self, analysis_name):
        """
        Run a single analysis defined in the config.
        Handles completion checking, checkpointing, and execution.
        """
        if analysis_name not in self.config['analyses']:
            raise ValueError(f"Analysis '{analysis_name}' not found in config")
        
        analysis_config = self.config['analyses'][analysis_name]
        print(f"\n🔬 Analysis: {analysis_name}")
        print(f"📋 Description: {analysis_config['description']}")
        print(f"⚙️ Configuration: {analysis_config['prompt']} + {analysis_config['model']}")
        
        # Check if already completed
        is_completed, completed_file = self.is_analysis_completed(analysis_config)
        if is_completed and self.config['pipeline']['skip_completed']:
            print(f"✅ Already completed: {completed_file.name}")
            return completed_file
        
        # Load checkpoint if exists
        checkpoint_df, processed_ids = self.load_checkpoint(analysis_config)
        results = checkpoint_df.to_dict('records') if checkpoint_df is not None else []
        
        try:
            # Load system prompt
            system_prompt = self.load_system_prompt(analysis_config['prompt'])
            
            # Get all conversation files
            raw_data_path = Path(self.config['paths']['raw_data'])
            json_files = list(raw_data_path.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No conversation files found in {raw_data_path}")
            
            # Filter out already processed files
            files_to_process = []
            for json_file in json_files:
                conv_data = self.load_conversation_from_json(json_file)
                if conv_data['conversation_id'] not in processed_ids:
                    files_to_process.append((json_file, conv_data))
            
            if not files_to_process:
                print("✅ All conversations already processed")
            else:
                total_conversations = len(json_files)
                already_processed = len(processed_ids)
                print(f"📊 Processing {len(files_to_process)} remaining conversations")
                print(f"📈 Progress: {already_processed} already done, {len(files_to_process)} remaining out of {total_conversations} total")
            
            # Process conversations
            failed_conversations = []
            rate_delay = self.get_rate_limit_delay(analysis_config['model'])
            
            for json_file, conv_data in tqdm(files_to_process, desc="Processing conversations"):
                try:
                    formatted_conv = self.format_conversation_for_shield(conv_data)
                    
                    # Call SHIELD API with retries
                    max_retries = self.config['api_defaults']['max_retries']
                    shield_response = None
                    
                    for retry in range(max_retries):
                        try:
                            shield_response = self.call_shield_api(
                                formatted_conv, system_prompt, analysis_config['model']
                            )
                            if shield_response != "ERROR":
                                break
                            if retry < max_retries - 1:
                                print(f"\n⚠️ Retry {retry + 1}/{max_retries - 1} for conversation {conv_data['conversation_id']}")
                                time.sleep(2 ** retry)
                        except Exception as e:
                            if retry == max_retries - 1:
                                raise e
                            time.sleep(2 ** retry)
                    
                    # Parse response and compile results
                    shield_results = self.process_shield_response(shield_response)
                    
                    result = {
                        'conversation_id': conv_data['conversation_id'],
                        'generation_model': conv_data['generation_model'],
                        'prompt_template_id': conv_data['prompt_template_id'],
                        'shield_prompt_version': analysis_config['prompt'],
                        'shield_model': analysis_config['model'],
                        'analysis_type': analysis_config['type'],
                        **shield_results,
                        'shield_test_timestamp': datetime.utcnow().isoformat()
                    }
                    
                    results.append(result)
                    
                    # Save checkpoint periodically
                    if len(results) % self.config['pipeline']['checkpoint_frequency'] == 0:
                        self.save_checkpoint(results, analysis_config)
                    
                    # Rate limiting
                    time.sleep(rate_delay)
                    
                except Exception as e:
                    print(f"\n❌ Error processing conversation {conv_data['conversation_id']}: {e}")
                    failed_conversations.append(conv_data['conversation_id'])
                    continue
            
            # Save final results
            results_df = pd.DataFrame(results)
            output_file = self.get_completed_filename(analysis_config)
            results_df.to_csv(output_file, index=False)
            
            # Clean up checkpoint after successful completion
            if self.config['pipeline']['cleanup_checkpoints'] and len(failed_conversations) == 0:
                checkpoint_file = self.get_checkpoint_filename(analysis_config)
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    print("🧹 Checkpoint cleaned up after successful completion")
            
            # Print summary
            interventions = results_df['shield_intervened'].sum() if pd.notna(results_df['shield_intervened']).any() else 0
            no_interventions = (~results_df['shield_intervened']).sum() if pd.notna(results_df['shield_intervened']).any() else 0
            errors = results_df['shield_intervened'].isna().sum()
            
            print(f"✅ Completed: {interventions} interventions, {no_interventions} no intervention, {errors} errors")
            if failed_conversations:
                print(f"⚠️ Failed conversations: {len(failed_conversations)}")
            print(f"💾 Saved to: {output_file.name}")
            
            return output_file
            
        except Exception as e:
            print(f"\n❌ Critical error in {analysis_name}: {e}")
            # Save whatever we have so far
            if results:
                self.save_checkpoint(results, analysis_config)
                print(f"💾 Checkpoint saved with {len(results)} conversations")
            return None
    
    def get_analysis_status(self):
        """Get status of all configured analyses"""
        status = {}
        
        for analysis_name, analysis_config in self.config['analyses'].items():
            is_completed, completed_file = self.is_analysis_completed(analysis_config)
            checkpoint_file = self.get_checkpoint_filename(analysis_config)
            has_checkpoint = checkpoint_file.exists()
            
            status[analysis_name] = {
                'config': analysis_config,
                'completed': is_completed,
                'completed_file': completed_file if is_completed else None,
                'has_checkpoint': has_checkpoint,
                'checkpoint_file': checkpoint_file if has_checkpoint else None
            }
        
        return status