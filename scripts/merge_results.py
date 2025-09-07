#!/usr/bin/env python3
"""
SHIELD Results Merger

Merges completed SHIELD analyses with human annotations to create master datasets.
Uses the same configuration system as the main pipeline.
"""

import pandas as pd
from pathlib import Path
import json
import yaml
import argparse
from datetime import datetime

class ResultsMerger:
    """Handles merging of SHIELD results with human annotations"""
    
    def __init__(self, config_path="config.yml"):
        """Initialize merger with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_conversation_metadata(self, conversation_id):
        """Load additional metadata from original JSON files"""
        raw_data_path = Path(self.config['paths']['raw_data'])
        json_files = list(raw_data_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('conversation_id') == conversation_id:
                        return {
                            'conversation_length': len(data['conversation_history']),
                            'total_tokens': sum(len(turn['text'].split()) for turn in data['conversation_history']),
                            'generation_timestamp': data.get('generation_timestamp_utc')
                        }
            except Exception:
                continue
        
        return {
            'conversation_length': None,
            'total_tokens': None,
            'generation_timestamp': None
        }
    
    def merge_single_analysis(self, analysis_file, output_suffix=""):
        """Merge a single SHIELD analysis file with human annotations"""
        annotations_path = Path(self.config['paths']['annotations'])
        output_path = Path(self.config['paths']['master_dataset'])
        
        # Load data
        annotations_df = pd.read_csv(annotations_path)
        shield_df = pd.read_csv(analysis_file)
        
        print(f"📊 Merging {analysis_file.name}")
        print(f"   📝 Annotations: {len(annotations_df)} rows")
        print(f"   🛡️  SHIELD results: {len(shield_df)} rows")
        
        # Merge on conversation_id
        master_df = pd.merge(
            annotations_df,
            shield_df,
            on='conversation_id',
            how='outer',
            suffixes=('_human', '_shield')
        )
        
        # Add metadata
        print("   🔄 Adding conversation metadata...")
        metadata_list = []
        for conv_id in master_df['conversation_id']:
            metadata = self.load_conversation_metadata(conv_id)
            metadata['conversation_id'] = conv_id
            metadata_list.append(metadata)
        
        metadata_df = pd.DataFrame(metadata_list)
        master_df = pd.merge(master_df, metadata_df, on='conversation_id', how='left')
        
        # Save result
        if output_suffix:
            output_file = output_path / f"master_dataset_{output_suffix}.csv"
        else:
            output_file = output_path / "master_dataset.csv"
        
        master_df.to_csv(output_file, index=False)
        print(f"   💾 Saved: {output_file.name} ({len(master_df)} rows)")
        
        return output_file
    
    def merge_all_analyses(self):
        """Merge all completed analyses into individual and combined datasets"""
        completed_path = Path(self.config['paths']['completed'])
        completed_files = list(completed_path.glob("*.csv"))
        
        if not completed_files:
            print("❌ No completed analysis files found")
            return
        
        print(f"🔄 Found {len(completed_files)} completed analysis files")
        
        # Merge each analysis individually
        master_files = []
        for analysis_file in completed_files:
            suffix = analysis_file.stem
            master_file = self.merge_single_analysis(analysis_file, suffix)
            master_files.append(master_file)
        
        # Create combined dataset
        print("\\n🔗 Creating combined dataset...")
        all_dfs = []
        
        for master_file in master_files:
            df = pd.read_csv(master_file)
            # Extract analysis variant from filename
            variant = master_file.stem.replace('master_dataset_', '')
            df['analysis_variant'] = variant
            all_dfs.append(df)
            print(f"   📊 Added {variant}: {len(df)} rows")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_file = Path(self.config['paths']['master_dataset']) / "master_dataset_all_analyses.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"\\n💾 Combined dataset: {combined_file.name} ({len(combined_df)} rows)")
            return combined_file
        
        return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Merge SHIELD results with human annotations')
    parser.add_argument('--config', default='config.yml',
                        help='Path to configuration file (default: config.yml)')
    parser.add_argument('--analysis', 
                        help='Specific analysis file to merge (relative to completed data folder)')
    parser.add_argument('--all', action='store_true',
                        help='Merge all completed analyses (default behavior)')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    
    print("🔗 SHIELD Results Merger")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        merger = ResultsMerger(args.config)
        
        if args.analysis:
            # Merge specific analysis
            completed_path = Path(merger.config['paths']['completed'])
            analysis_file = completed_path / args.analysis
            
            if not analysis_file.exists():
                print(f"❌ Analysis file not found: {analysis_file}")
                return 1
            
            suffix = analysis_file.stem
            merger.merge_single_analysis(analysis_file, suffix)
        else:
            # Merge all analyses (default)
            merger.merge_all_analyses()
        
        print("\\n✅ Merging completed successfully!")
        print("📊 Master datasets are ready for R analysis!")
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Error during merging: {e}")
        return 1

if __name__ == "__main__":
    exit(main())