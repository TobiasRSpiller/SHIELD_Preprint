# SHIELD Benchmark Pipeline

A comprehensive pipeline for testing and analyzing AI safety shield systems across different models and prompt configurations. This pipeline evaluates how well AI safety measures block inappropriate content while allowing appropriate responses.

## Overview

The SHIELD benchmark evaluates AI safety systems through a complete pipeline that:
1. Generates conversations across multiple AI models using various prompt types
2. Prepares data for human annotation (performed with external annotation software)
3. Analyzes shield effectiveness using different configurations
4. Merges results with human annotations
5. Performs statistical analysis of shield performance

**Note**: Human annotation is performed using external annotation software and is not included in this repository.

## Folder Structure

```
shield_benchmark/
├── README.md                         # This file
├── config.yml                        # Pipeline configuration
│
├── scripts/                          # Core pipeline scripts
│   ├── 01_generate_conversations.py  # Generate LLM conversations
│   ├── 02_prepare_for_annotation.py  # Format data for annotation
│   ├── analyzer.py                   # Core SHIELD analysis library
│   ├── run_pipeline.py               # Main pipeline orchestrator
│   ├── merge_results.py              # Results merger with annotations
│   └── prompt_templates.csv          # Conversation prompt templates
│
├── system_prompts/                   # SHIELD prompt variants
│   ├── shield_v1.txt                # Base SHIELD prompt
│   ├── shield_v2.txt                # Sensitivity test variant 1
│   └── shield_v3.txt                # Sensitivity test variant 2
│
├── data/                             # All pipeline data
│   ├── 01_raw_generations/           # Raw LLM conversation files
│   ├── 02_for_annotation/            # Data prepared for annotation
│   ├── 03_annotated_results/         # Human annotation results
│   ├── 04_shield_checkpoint_data/    # Analysis checkpoints
│   ├── 05_shield_completed_data/     # Completed SHIELD analyses
│   └── 06_master_dataset/            # Final merged datasets
│
└── R_Analysis/                       # Statistical analysis
    └── SHIELD_Performance_analysis.Rmd  # R Markdown analysis
```

## Pipeline Scripts

### Core Pipeline Components

#### `config.yml`
Central configuration file defining:
- File paths for all pipeline stages
- API settings and rate limits
- Analysis definitions and parameters
- Pipeline behavior settings

#### `scripts/analyzer.py`
Core library containing:
- `ShieldAnalyzer` class with all analysis functions
- System prompt loading and conversation formatting
- LLM API calls for SHIELD evaluation
- Progress tracking and checkpoint management
- Results saving and loading

#### `scripts/run_pipeline.py`
Main orchestrator script that:
- Reads configuration and determines required analyses
- Executes SHIELD analyses with progress tracking
- Handles checkpointing and resumption
- Provides status reporting and command-line interface

#### `scripts/merge_results.py`
Results merger that:
- Combines SHIELD analysis results with human annotations
- Creates master datasets for statistical analysis
- Maintains data consistency across pipeline stages

### Data Generation Scripts

#### `scripts/01_generate_conversations.py`
Generates test conversations by:
- Loading prompt templates from `prompt_templates.csv`
- Creating conversations across multiple LLM models:
  - GPT-4.1 Nano
  - Gemini Gemma-3n
  - Groq Llama 3
  - Claude 3.5 Haiku
- Categorizing outputs into conversation types:
  - `appropriate_emotional` - Appropriate responses to emotional topics
  - `control` - Standard control conversations
  - `inappropriate_blocked` - Inappropriate content that should be blocked
  - `inappropriate_not_blocked` - Inappropriate content that wasn't blocked
- Saving structured JSON files for each conversation

#### `scripts/02_prepare_for_annotation.py`
Prepares data for human annotation by:
- Loading all generated conversation JSON files
- Formatting conversations for annotation interface
- Creating CSV files with conversation metadata
- Organizing data for external annotation software

### Analysis Pipeline

The pipeline runs multiple types of analyses:

#### Main Analysis
- **Purpose**: Primary SHIELD effectiveness evaluation
- **Configuration**: Default prompt (shield_v1.txt) with Groq Llama model
- **Output**: Baseline performance metrics

#### Prompt Sensitivity Analysis
- **Purpose**: Test how prompt variations affect SHIELD performance
- **Variants**: shield_v2.txt and shield_v3.txt
- **Output**: Comparative analysis of prompt effectiveness

#### Model Sensitivity Analysis
- **Purpose**: Test SHIELD performance across different LLM models
- **Models**: Claude 3 Haiku vs. default Groq Llama
- **Output**: Cross-model performance comparison

## Usage

### Run Complete Pipeline
```bash
# Generate conversations (if needed)
python3 scripts/01_generate_conversations.py

# Prepare for annotation (if needed)
python3 scripts/02_prepare_for_annotation.py

# Run all SHIELD analyses
python3 scripts/run_pipeline.py

# Merge with annotations
python3 scripts/merge_results.py
```

### Run Specific Analyses
```bash
# Run only sensitivity analyses
python3 scripts/run_pipeline.py --skip-main

# Run specific named analyses
python3 scripts/run_pipeline.py --analyses prompt_sensitivity_v2 model_sensitivity_claude

# Force re-run completed analyses
python3 scripts/run_pipeline.py --force
```

### Statistical Analysis
```bash
# Open R/RStudio and run the R Markdown file
# R_Analysis/SHIELD_Performance_analysis.Rmd
```

## Data Flow

1. **Generation**: `01_generate_conversations.py` creates raw conversation data
2. **Preparation**: `02_prepare_for_annotation.py` formats data for annotation
3. **Annotation**: External software used for human labeling (not in repository)
4. **Analysis**: `run_pipeline.py` runs SHIELD evaluations on conversations
5. **Merging**: `merge_results.py` combines SHIELD results with human annotations
6. **Statistics**: R Markdown analysis provides final performance metrics

## Requirements

- Python 3.7+ with packages: `litellm`, `pandas`, `pyyaml`, `tqdm`, `python-dotenv`
- R with packages: `tidyverse`, `plotly`, `caret`, `irr` (for statistical analysis)
- API keys for LLM providers (OpenAI, Anthropic, Google, Groq)

## Configuration

All pipeline behavior is controlled through `config.yml`. Key settings include:

- **File Paths**: Where to find/save data at each stage
- **API Settings**: Rate limits, retry counts, model parameters
- **Analysis Definitions**: Which analyses to run with what parameters
- **Pipeline Behavior**: Checkpointing, progress display, cleanup options

To add new analyses, simply add entries to the `analyses` section in the config file - no code changes required.