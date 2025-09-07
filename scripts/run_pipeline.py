"""
SHIELD Analysis Pipeline Runner

This script orchestrates the execution of SHIELD analyses based on configuration.
It reads the config file, determines what analyses need to be run, and executes them.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml

# Import our analyzer library
from .analyzer import ShieldAnalyzer

def print_banner():
    """Print startup banner"""
    print("🛡️  SHIELD Analysis Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def print_analysis_status(analyzer):
    """Print current status of all analyses"""
    status = analyzer.get_analysis_status()
    
    print("\n📊 Analysis Status:")
    for analysis_name, info in status.items():
        config = info['config']
        status_icon = "✅" if info['completed'] else ("🔄" if info['has_checkpoint'] else "⏳")
        status_text = "Completed" if info['completed'] else ("In Progress" if info['has_checkpoint'] else "Pending")
        
        print(f"   {status_icon} {analysis_name}: {status_text}")
        print(f"      📝 {config['description']}")
        print(f"      ⚙️ {config['prompt']} + {config['model']}")
        
        if info['completed']:
            print(f"      💾 File: {info['completed_file'].name}")
        elif info['has_checkpoint']:
            print(f"      📂 Checkpoint: {info['checkpoint_file'].name}")

def get_analyses_to_run(analyzer, args):
    """Determine which analyses need to be run"""
    config = analyzer.config
    status = analyzer.get_analysis_status()
    
    # If specific analyses requested via command line
    if args.analyses:
        requested = args.analyses
        # Validate all requested analyses exist
        for analysis in requested:
            if analysis not in config['analyses']:
                print(f"❌ Error: Analysis '{analysis}' not found in config")
                print(f"Available analyses: {list(config['analyses'].keys())}")
                sys.exit(1)
        analyses_to_check = requested
    # If specified in config
    elif config['pipeline']['run_analyses']:
        analyses_to_check = config['pipeline']['run_analyses']
    # Otherwise run all defined analyses
    else:
        analyses_to_check = list(config['analyses'].keys())
    
    # Filter based on completion status and force flag
    analyses_to_run = []
    
    for analysis_name in analyses_to_check:
        info = status[analysis_name]
        
        # Skip if completed (unless forcing)
        if info['completed'] and not args.force:
            if args.verbose:
                print(f"⏭️  Skipping {analysis_name}: already completed")
            continue
        
        # Skip main analysis if requested
        if args.skip_main and info['config']['type'] == 'main':
            if args.verbose:
                print(f"⏭️  Skipping {analysis_name}: main analysis skip requested")
            continue
        
        analyses_to_run.append(analysis_name)
    
    return analyses_to_run

def run_analyses(analyzer, analyses_to_run, args):
    """Execute the analyses"""
    if not analyses_to_run:
        print("\\n✅ No analyses need to be run!")
        return True
    
    print(f"\\n🎯 Running {len(analyses_to_run)} analyses:")
    for analysis in analyses_to_run:
        config = analyzer.config['analyses'][analysis]
        print(f"   📋 {analysis}: {config['description']}")
    
    print("\\n" + "=" * 60)
    
    success_count = 0
    failed_analyses = []
    
    # Run each analysis
    for i, analysis_name in enumerate(analyses_to_run, 1):
        print(f"\\n{'=' * 20} Analysis {i}/{len(analyses_to_run)} {'=' * 20}")
        print(f"🔬 {analysis_name}")
        print("=" * 60)
        
        try:
            result = analyzer.run_single_analysis(analysis_name)
            if result:
                success_count += 1
                print(f"\\n✅ {analysis_name} completed successfully")
            else:
                failed_analyses.append(analysis_name)
                print(f"\\n❌ {analysis_name} failed")
        except KeyboardInterrupt:
            print(f"\\n\\n⚠️ User interrupted during: {analysis_name}")
            print("Progress has been saved in checkpoint files.")
            sys.exit(1)
        except Exception as e:
            print(f"\\n❌ Unexpected error in {analysis_name}: {e}")
            failed_analyses.append(analysis_name)
    
    # Print final summary
    print("\\n" + "=" * 60)
    if failed_analyses:
        print("⚠️  PIPELINE COMPLETED WITH SOME FAILURES")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {len(failed_analyses)}")
        print(f"Failed analyses: {', '.join(failed_analyses)}")
        return False
    else:
        print("🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print(f"✅ Completed: {success_count} analyses")
        return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run SHIELD analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run all analyses defined in config
  %(prog)s --analyses main_analysis           # Run only main analysis
  %(prog)s --analyses prompt_sensitivity_v2 prompt_sensitivity_v3  # Run specific analyses
  %(prog)s --skip-main                        # Run all except main analysis  
  %(prog)s --force                            # Re-run even completed analyses
  %(prog)s --status                           # Show status without running anything
        """
    )
    
    parser.add_argument('--config', default='config.yml',
                        help='Path to configuration file (default: config.yml)')
    parser.add_argument('--analyses', nargs='+',
                        help='Specific analyses to run (overrides config)')
    parser.add_argument('--skip-main', action='store_true',
                        help='Skip main analysis (focus on sensitivity analyses)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run analyses even if already completed')
    parser.add_argument('--status', action='store_true',
                        help='Show analysis status and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    print_banner()
    
    try:
        # Initialize analyzer
        analyzer = ShieldAnalyzer(args.config)
        
        # Show current status
        print_analysis_status(analyzer)
        
        # If just showing status, exit
        if args.status:
            print("\\n📊 Status check complete.")
            sys.exit(0)
        
        # Determine what to run
        analyses_to_run = get_analyses_to_run(analyzer, args)
        
        # Confirm before starting (unless forced)
        if analyses_to_run and not args.force:
            print(f"\\n🚀 Ready to run {len(analyses_to_run)} analyses.")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                sys.exit(0)
        
        # Run the analyses
        start_time = datetime.now()
        success = run_analyses(analyzer, analyses_to_run, args)
        end_time = datetime.now()
        
        # Final summary
        duration = (end_time - start_time).total_seconds() / 60
        print(f"\\nTotal runtime: {duration:.1f} minutes")
        
        if success:
            print("\\n📊 You can now run the R Markdown analysis!")
        else:
            print("\\n⚠️  Fix the issues above before running the R Markdown analysis")
        
        print("=" * 60)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\\n\\n⚠️ Pipeline interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()