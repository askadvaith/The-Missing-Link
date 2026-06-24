# Downstream Testing Pipeline

A comparative testing framework for benchmarking synthesized prompts against standard baseline techniques (Zero-Shot, CoT, ReAct, Expert-Persona) using DeepEval benchmarks.

## Key Features
- **Log-Probability Scoring**: Accurate MCQ evaluation using 1-token generation (`logprobs=True`), removing the need for a subjective LLM judge.
- **Universal Model Compatibility**: Uses `litellm` to support OpenAI, Anthropic, and local Ollama instances (requires Ollama >= 0.4.0 for `logprobs=True`).
- **Prompt Benchmarking**: Dynamically injects graph components into prompts to compare against standard baselines.

---

## 1. Installation & Setup

We recommend using `uv` for fast, reproducible dependency installations.

```powershell
# 1. Create a virtual environment
uv venv

# 2. Activate it (Windows)
.\.venv\Scripts\Activate.ps1

# 3. Install packages
uv pip install -r requirements.txt
```

### API Keys & Environment Variables

If you are using external APIs like OpenAI, Anthropic, or Google Gemini, you need to set up your environment variables.

```powershell
copy .env.example .env
```

Open the `.env` file and fill in your API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
*Note: If you are only using local models via Ollama, API keys are not required.*

---

## 2. Preparing the Input Data

Before running the pipeline, you **must** provide the Graph component prediction JSON files. 

These files need to be placed inside the `link_pred_output/` directory in a specific folder structure so the CLI can discover them automatically.

### Expected Directory Structure
Place your JSON files (the filename must contain `novel_techniques`) into `link_pred_output/` following either a flat or hypergraph structure:

```text
link_pred_output/
├── flat/                     <-- Graph Type (flat or hyper)
│   └── GTN/                  <-- Model Name
│       └── run_2026_01_01/   <-- Run Name
│           └── novel_techniques_gtn.json
└── hyper/
    └── HGNN/
        └── run_2026_01_02/
            └── novel_techniques_hgnn.json
```

---

## 3. How to Run

### Interactive Runner (Recommended)

The easiest way to run the pipeline is using the interactive CLI.

```powershell
python cli.py
```

This tool will walk you through:
1. **Source Selection**: Auto-detects the graph models and runs inside `link_pred_output/`.
2. **Task Selection**: Pick which tasks/use-cases to evaluate.
3. **Execution Parameters**: Sample size, Top-K components, concurrency limit.
4. **Model Selection**: Choose your synthesizer and evaluator models (from `config/models.yaml`).

### Non-Interactive Orchestrator

If you want to script the evaluation or run it in the background:

```powershell
python run_eval.py --source-json "link_pred_output/flat/GTN/run_20260318_152942/novel_techniques_gtn.json" \
                   --tasks "Truthfulness / Factual Accuracy" \
                   --samples 10 \
                   --concurrency 5
```

**Optional Arguments:**
- `--source-json` *(Required)*: Path to the predicted links JSON.
- `--tasks`: Comma separated task list (evaluates all if omitted).
- `--samples`: Maximum number of benchmark problems per task (Default: 5).
- `--top-k`: Number of graph components to include (Default: 5).
- `--eval-model`: LiteLLM identifier (e.g., `openai/gpt-4o-mini`).
- `--synth-model`: Model used for prompt synthesis.

---

## 4. Results and Output

All evaluation outputs are saved under the `results/` directory organized by task and model:

`results/<task_name>/<eval_model>_run_<timestamp>/`

Inside each run directory, you will find:
- **`results.json`**: The benchmark accuracy scores per prompt method (Synthesized, Zero-Shot, CoT, ReAct).
- **`metadata.json`**: Run config info (source model, parameters, models used).
- **`prompts/`**: `.txt` files containing the exact text of the prompts sent to the evaluator.
- **`logs/`**: Detailed prediction logs for every evaluated question.

---

## 5. Configuration

You can customize models and benchmarks in the `config/` directory:

1. **`config/models.yaml`**: Define default synthesizer and evaluator models, and create easy-to-use aliases (e.g., `ollama_local`).
2. **`config/datasets.yaml`**: Maps semantic graph tasks to HuggingFace datasets. You can disable specific benchmarks by setting `enabled: false`.
