import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

def _sanitize_path_token(text: str) -> str:
    """Sanitize a user-provided token so it is safe as a filename on Windows/Linux."""
    token = str(text or "").strip()
    token = re.sub(r'[<>:"/\\|?*]', '_', token)
    token = re.sub(r'\s+', '_', token)
    token = token.strip(' .')
    return token or "unnamed"

def make_run_dir(base_out_dir: str = "results", run_prefix: str | None = None) -> str:
    """Creates a timestamped output directory for the run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = _sanitize_path_token(run_prefix) if run_prefix else None
    run_token = f"{prefix}_run_{timestamp}" if prefix else f"run_{timestamp}"
    run_dir = os.path.join(base_out_dir, run_token)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_prompts(run_dir: str, task_name: str, prompts: Dict[str, str]):
    """Saves generated prompts to text files for inspection."""
    safe_task = _sanitize_path_token(task_name)
    prompt_dir = os.path.join(run_dir, "prompts", safe_task)
    os.makedirs(prompt_dir, exist_ok=True)
    
    for baseline, text in prompts.items():
        safe_baseline = _sanitize_path_token(baseline)
        file_path = os.path.join(prompt_dir, f"{safe_baseline}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

def save_results(run_dir: str, results_payload: Dict[str, Any]):
    """Saves the final evaluation results JSON."""
    results_file = os.path.join(run_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, default=str)

def save_metadata(run_dir: str, metadata: Dict[str, Any]):
    """Saves run metadata JSON for provenance and reproducibility."""
    metadata_file = os.path.join(run_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

def save_detailed_logs(run_dir: str, task_name: str, method_name: str, logs: List[Dict[str, Any]]):
    """Saves raw predictions, targets and scores for a specific method/task."""
    safe_task = _sanitize_path_token(task_name)
    safe_method = _sanitize_path_token(method_name)
    log_dir = os.path.join(run_dir, "logs", safe_task)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{safe_method}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, default=str)
