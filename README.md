# The Missing Link: Knowledge Graph-Guided Discovery of Novel Prompt Compositions

This repository contains the camera-ready code, datasets, and evaluation pipeline for the research work:
**"The Missing Link: Knowledge Graph-Guided Discovery of Novel Prompt Compositions"** (A.S. Kumar et al.), published at the 35th International Conference on Artificial Neural Networks (ICANN), 2026.

The KG itself can be found in standard format at [https://huggingface.co/datasets/AdvaithMagic/PromptForge](https://huggingface.co/datasets/AdvaithMagic/PromptForge).

This repo contains:
1. **Modelling & Link Prediction**: GNN-based and Hypergraph-based pipelines to train embedding models and discover novel composition candidates on a Prompt Engineering Knowledge Graph.
2. **Downstream Testing Pipeline**: A comprehensive testing and benchmarking framework that evaluates synthesized prompt compositions against standard baselines (Zero-Shot, CoT, ReAct, Expert-Persona).

---

## Directory Structure

```text
/
├── README.md                          # Repository overview and guide (this file)
│
├── modelling/                         # 1. GNN Link Prediction & Candidate Discovery
│   ├── requirements.txt               # Dependencies for modelling (commented out PyTorch/PyG)
│   ├── link_prediction/               # Model implementations and utility functions
│   │   ├── gcn_link_prediction.py     # Standard GCN model training & candidate prediction
│   │   ├── hgnn_prediction.py         # Hypergraph HGNN model training & candidate prediction
│   │   ├── utils.py                   # Graph splitting, negative sampling & discovery helpers
│   │   ├── edge_logreg_prediction.py  # Logistic Regression baseline
│   │   ├── random_link_prediction.py  # Random prediction baseline
│   │   └── ...                        # GAT, GTN, RGCN, Metapath2Vec, etc.
│   ├── utils/                         # Figure generation and metric utilities
│   │   └── metrics.py                 # Mathematical metrics scoring functions
│   ├── data/                          # Input Graph dataset representations
│   │   └── hypergraph_output/         # Hypergraph dataset pickle file (.pkl)
│   └── graph_output/                  # Saved graph structure artifacts (.pkl)
│
└── testing_pipeline/                  # 2. Downstream Prompt Benchmarking & Evaluation
    ├── README.md                      # Detailed setup and instructions for the testing pipeline
    ├── requirements.txt               # Dependencies (DeepEval, LiteLLM, datasets, etc.)
    ├── .env.example                   # Template for API keys and environment variables
    ├── cli.py                         # Interactive CLI tool for prompt benchmarking
    ├── run_eval.py                    # Scriptable non-interactive orchestrator
    ├── config/                        # Configuration YAMLs for datasets and LLMs
    │   ├── datasets.yaml              # HuggingFace datasets mappings
    │   └── models.yaml                # Model alias definition mapping
    ├── link_pred_output/              # Predictions to evaluate (copied from modelling outputs)
    │   ├── flat/                      # GNN flat predictions directory
    │   └── hyper/                     # Hypergraph predictions directory
    ├── results/                       # Outputs of benchmarking runs
    └── src/                           # Backend code (loader, prompt builder, evaluator, etc.)
```

---

## 1. Modelling & Link Prediction Setup

The modelling directory handles training the Graph Neural Networks and predicting novel prompt techniques.

### Installation
Since PyTorch and PyG (PyTorch Geometric) are system-dependent, they are commented out in `modelling/requirements.txt` to prevent automated installation failures. They must be installed separately.

1. **Navigate to the modelling directory**:
   ```bash
   cd modelling
   ```

2. **Install PyTorch**:
   Choose the installation matching your hardware on the [PyTorch official page](https://pytorch.org/).
   *Example for CPU:*
   ```bash
   pip install torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install PyTorch Geometric (PyG)**:
   *Example for CPU:*
   ```bash
   pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
   pip install torch-geometric
   ```

4. **Install Remaining Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Execution & Training
To train the link prediction models and export candidate compositions:

- **Run Standard Graph GNNs (e.g. GCN)**:
  ```bash
  python -m link_prediction.gcn_link_prediction
  ```
  *(Select your graph run interactively when prompted. Outputs are saved in `link_pred_output/GCN/run_<timestamp>/`)*

- **Run Hypergraph GNN (HGNN)**:
  ```bash
  python -m link_prediction.hgnn_prediction
  ```
  *(Outputs are saved in `link_pred_output/HGNN/run_<timestamp>/`)*

Outputs will include:
- `model_*.pt`: The trained model state weights dictionary.
- `hyperparameters.json`: The execution environment and hyperparameters config.
- `novel_techniques_*.json`: A generated list of top candidate prompt compositions.

---

## 2. Downstream Testing Pipeline Setup

The downstream pipeline benchmarks candidate prompt compositions against traditional baselines.

### Installation
1. **Navigate to the testing_pipeline directory**:
   ```bash
   cd testing_pipeline
   ```

2. **Create and Activate a Virtual Environment** (Recommended: using `uv`):
   ```bash
   uv venv
   # Windows:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   # or with pip:
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Copy `.env.example` to `.env` and fill in any required model API keys:
   ```bash
   copy .env.example .env
   ```

### Preparing Prediction Input Data
To test candidates generated from the **Modelling** step:
1. Locate the `novel_techniques_*.json` output from your modelling run.
2. Place the JSON file inside `testing_pipeline/link_pred_output/` under the appropriate model type (`flat/` or `hyper/`) in the following structure:
   ```text
   link_pred_output/
   ├── flat/                     <-- Graph Type
   │   └── GCN/                  <-- Model Name (e.g., GCN, GAT, GTN)
   │       └── run_20260624/     <-- Run Name
   │           └── novel_techniques_gcn.json
   └── hyper/
       └── HGNN/
           └── run_20260624/
               └── novel_techniques_hgnn.json
   ```

### Running Benchmark Evaluations
From the `testing_pipeline` directory:

- **Interactive CLI (Recommended)**:
  ```bash
  python cli.py
  ```
  This interactive tool guides you through selecting candidate JSON sources, benchmark tasks, sample sizes, models, and concurrency limits.

- **Non-Interactive Command**:
  ```bash
  python run_eval.py --source-json "link_pred_output/flat/GCN/run_20260624/novel_techniques_gcn.json" \
                     --tasks "Truthfulness / Factual Accuracy" \
                     --samples 10 \
                     --concurrency 5
  ```

Benchmark results and generated prompts are detailed under `results/`. Refer to `testing_pipeline/README.md` for a full breakdown.
