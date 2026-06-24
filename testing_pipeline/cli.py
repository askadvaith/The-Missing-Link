"""Interactive CLI for testing-pipeline-v3 (Log-Prob evaluation).

Usage
─────
  python cli.py

Provides a menu-driven interface to:
  1. Select a predicted_links JSON from link_pred_output/
  2. Select which tasks / use cases to evaluate
  3. Configure parameters (Sample Size, Components, Concurrency)
  4. Select LLM models (Synthesizer and Evaluator)
  5. Launch comparative evaluation using log-probability scoring
"""

import os
import sys
import json
import asyncio
from pathlib import Path
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from run_eval import load_yaml, _resolve_model_config
from src.evaluator import evaluate_task
from src.logger import make_run_dir, save_results, save_prompts, save_metadata, _sanitize_path_token

console = Console()

def _ask_or_exit(prompt):
    try:
        res = prompt.ask()
        if res is None:
            console.print("\n[yellow]Cancelled by user. Exiting...[/yellow]")
            sys.exit(0)
        return res
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user. Exiting...[/yellow]")
        sys.exit(0)

LINK_PRED_OUTPUT_DIR = "link_pred_output"
DEFAULT_TOP_K = 15
DEFAULT_MAX_SAMPLES = 10000
DEFAULT_CONCURRENCY = 5

def _safe_task_dir_name(task_name: str) -> str:
    return str(task_name or "unknown_task").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _resolve_dataset_task_key(task_name: str, datasets_cfg: dict) -> str | None:
    """Resolve raw task labels to a datasets.yaml key using optional task_aliases."""
    datasets = datasets_cfg.get("datasets", {}) if isinstance(datasets_cfg, dict) else {}
    aliases = datasets_cfg.get("task_aliases", {}) if isinstance(datasets_cfg, dict) else {}
    mapped = aliases.get(task_name, task_name)
    return mapped if mapped in datasets else None

def _source_origin_info(source_json_path: str) -> dict:
    p = Path(source_json_path)
    parts = list(p.parts)
    graph_type = None
    model_name = None
    run_name = None
    if "link_pred_output" in parts:
        i = parts.index("link_pred_output")
        if i + 1 < len(parts) and parts[i + 1] in ("flat", "hyper"):
            graph_type = parts[i + 1]
            if i + 2 < len(parts):
                model_name = parts[i + 2]
            if i + 3 < len(parts):
                run_name = parts[i + 3]
        else:
            if i + 1 < len(parts):
                model_name = parts[i + 1]
            if i + 2 < len(parts):
                run_name = parts[i + 2]
    
    # Use the filename stem as the model name if it's a specific seed JSON
    if p.name and "seed" in p.name.lower():
        model_name = p.stem

    import re
    seed = None
    if p.name:
        match = re.search(r'seed[_\-]?(\d+)', p.name, re.IGNORECASE)
        if match:
            seed = match.group(1)

    return {
        "source_json": source_json_path,
        "source_graph_type": graph_type,
        "source_model": model_name,
        "source_run": run_name,
        "source_filename": p.name,
        "seed": seed,
    }

def _parse_positive_int(raw: str, default: int, field_name: str) -> int:
    if not raw or not raw.strip():
        return default
    try:
        val = int(raw.strip())
        if val < 1:
            raise ValueError
        return val
    except:
        console.print(f"[red]Invalid integer for {field_name}. Using default {default}.[/red]")
        return default


def _list_model_dirs(base_dir: str) -> list[str]:
    """Return a list of selectable model paths, e.g. 'flat/GTN', 'hyper/HGNN'."""
    curr = Path(base_dir)
    if not curr.exists() and Path("../link_pred_output").exists():
         curr = Path("../link_pred_output")
    if not curr.exists():
         return []
    graph_types = ["flat", "hyper"]
    results = []
    for gt in graph_types:
        gt_path = curr / gt
        if gt_path.exists():
            for model_dir in sorted(d for d in gt_path.iterdir() if d.is_dir()):
                results.append(f"{gt}/{model_dir.name}")
    # Fallback: legacy top-level model dirs (not flat/hyper)
    if not results:
        results = sorted(d.name for d in curr.iterdir() if d.is_dir() and d.name not in ("flat", "hyper"))
    return results


def _collect_source_jsons(run_dir: Path) -> list[Path]:
    """Collect ensembled and per-seed source JSONs for a single run."""
    candidates = []

    for candidate in sorted(run_dir.glob("*novel_techniques*.json"), key=lambda p: p.name):
        if candidate.is_file():
            candidates.append(candidate)

    for seed_dir in sorted((p for p in run_dir.glob("seed_discovery__*") if p.is_dir()), key=lambda p: p.name):
        for candidate in sorted(seed_dir.glob("*.json"), key=lambda p: p.name):
            if candidate.is_file() and "novel_techniques" in candidate.name:
                candidates.append(candidate)

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        unique_candidates.append(candidate)

    return unique_candidates


def _list_runs_with_novel_techniques(model_dir_path: str) -> list[dict]:
    """model_dir_path may be 'flat/GTN' or legacy 'GTN'."""
    curr = Path(LINK_PRED_OUTPUT_DIR)
    if not curr.exists(): curr = Path("../link_pred_output")
    m_dir = curr / model_dir_path
    results = []
    if m_dir.exists():
        for r_dir in sorted([p for p in m_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            source_jsons = _collect_source_jsons(r_dir)
            if source_jsons:
                results.append({
                    "run": r_dir.name,
                    "json_files": [p.name for p in source_jsons],
                    "paths": [str(p) for p in source_jsons],
                })
    return results

def step_select_source() -> list[str] | None:
    model_dirs = _list_model_dirs(LINK_PRED_OUTPUT_DIR)
    if not model_dirs:
        console.print(f"[red]No subdirectories found in link_pred_output/[/red]")
        return None

    model = _ask_or_exit(questionary.select(
        "Select model type:",
        choices=model_dirs,
        style=questionary.Style([("highlighted", "fg:cyan bold"), ("selected", "fg:green")])
    ))
    console.print(f"  → [green]{model}[/green]")

    runs = _list_runs_with_novel_techniques(model)
    if not runs:
        console.print(f"[red]No runs with novel_techniques JSON found in {model}/[/red]")
        return None

    selected_run = _ask_or_exit(questionary.select(
        f"Select a run from {model}/:",
        choices=[
            questionary.Choice(
                title=f"{r['run']}  ({len(r['paths'])} source JSONs)",
                value=r,
            )
            for r in runs
        ],
        style=questionary.Style([("highlighted", "fg:cyan bold"), ("selected", "fg:green")])
    ))

    json_choices = [
        questionary.Choice(title=name, value=path)
        for name, path in zip(selected_run["json_files"], selected_run["paths"])
    ]
    selected_jsons = _ask_or_exit(questionary.checkbox(
        f"Select source JSONs to evaluate for {model}/{selected_run['run']}:",
        choices=json_choices,
    ))

    if not selected_jsons:
        console.print("[red]No source JSONs selected.[/red]")
        return None

    return selected_jsons


def _evaluate_source_json(
    source_path: str,
    task_filter: list[str] | None,
    params: dict,
    model_cfg: dict,
    datasets_cfg: dict,
    synth_cfg: dict,
    eval_cfg: dict,
) -> None:
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            source_data = json.load(f)
    except Exception as e:
         console.print(f"[red]Failed to load source JSON: {e}[/red]")
         return

    if isinstance(source_data, dict) and "shortlists_by_use_case" in source_data:
        source_data = source_data["shortlists_by_use_case"]

    source_origin = _source_origin_info(source_path)
    source_model_prefix = source_origin.get("source_model")

    def _get_task_id(item: dict) -> str:
        task = item.get("task")
        if isinstance(task, dict):
            return task.get("id", "")
        return task or ""

    selected_tasks = []
    if task_filter:
        target_tasks = set(task_filter)
        selected_tasks = [t for t in source_data if _get_task_id(t) in target_tasks]
    else:
        selected_tasks = source_data

    for item in selected_tasks:
        task_name = _get_task_id(item)
        dataset_task_key = _resolve_dataset_task_key(task_name, datasets_cfg)
        if not dataset_task_key:
             console.print(f"[yellow]Skipping task {task_name} (Not found in datasets.yaml mapping)[/yellow]")
             continue
        if dataset_task_key != task_name:
            console.print(f"[dim]Mapped task label '{task_name}' -> '{dataset_task_key}'[/dim]")
        
        # Standardize task_name to the canonical key for consistent baseline prompt formats and directories
        task_name = dataset_task_key

        task_ds_cfg = datasets_cfg["datasets"][dataset_task_key]
        benchmark_cfgs = task_ds_cfg.get("benchmarks") if isinstance(task_ds_cfg, dict) else None
        if benchmark_cfgs is None:
            benchmark_cfgs = [task_ds_cfg]

        task_dir_name = _safe_task_dir_name(task_name)
        evaluator_model_dir = _sanitize_path_token(model_cfg["eval_model"])
        task_base_out_dir = os.path.join("results", task_dir_name, evaluator_model_dir)
        task_run_dir = make_run_dir(task_base_out_dir, run_prefix=source_model_prefix)

        task_payload = {
            "source": source_path,
            "canonical_task": task_name,
            "synthesizer_model": model_cfg["synth_model"],
            "evaluator_model": model_cfg["eval_model"],
            "benchmarks": []
        }

        blueprints = item.get("shortlisted_technique_blueprints")
        if blueprints:
            candidates = []
            for bp in blueprints:
                candidates.extend(bp.get("shortlisted_components") or [])
        else:
            candidates = item.get("suggested_components") or item.get("components") or item.get("top_components") or item.get("novel_components") or []
        components = []
        for c in candidates:
             if isinstance(c, str): components.append(c)
             elif isinstance(c, dict): components.append(c.get("name") or c.get("component") or "")
        components = [c for c in components if c][:params["top_k"]]

        console.print(f"\n[bold green]Running {task_name}...[/bold green]")
        console.print(f"[dim]Output directory: {task_run_dir}[/dim]")

        for idx, benchmark_cfg in enumerate(benchmark_cfgs, start=1):
            if isinstance(benchmark_cfg, dict) and benchmark_cfg.get("enabled") is False:
                benchmark_name = benchmark_cfg.get("benchmark_name", f"benchmark_{idx}")
                console.print(f"[yellow]Skipping disabled benchmark: {benchmark_name}[/yellow]")
                continue

            benchmark_name = benchmark_cfg.get("benchmark_name", f"benchmark_{idx}")
            console.print(f"[cyan]  • Benchmark [{idx}/{len(benchmark_cfgs)}]: {benchmark_name}[/cyan]")

            task_results = asyncio.run(evaluate_task(
                task_name=f"{task_name} :: {benchmark_name}",
                components=components,
                dataset_cfg=benchmark_cfg,
                eval_model_cfg=eval_cfg,
                synth_model_cfg=synth_cfg,
                synthesis_task_name=task_name,
                run_dir=task_run_dir,
                max_samples=params["max_samples"],
                concurrency=params["concurrency"],
                skip_baselines=params["skip_baselines"],
                source_info=source_origin,
            ))

            if "error" not in task_results:
                task_results["canonical_task"] = task_name
                task_results["benchmark_name"] = benchmark_name
                save_prompts(task_run_dir, f"{task_name}__{benchmark_name}", task_results.pop("prompts_used", {}))
                task_payload["benchmarks"].append(task_results)

        save_results(task_run_dir, task_payload)
        save_metadata(task_run_dir, {
            **source_origin,
            "canonical_task": task_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "models": {
                "synthesizer": model_cfg["synth_model"],
                "evaluator": model_cfg["eval_model"],
            },
            "params": {
                "top_k": params["top_k"],
                "samples": params["max_samples"],
                "concurrency": params["concurrency"],
            },
            "benchmarks_requested": len(benchmark_cfgs),
            "benchmarks_completed": len(task_payload["benchmarks"]),
        })
        console.print(f"[green]Saved task results → {task_run_dir}[/green]")


def step_select_tasks(json_paths: list[str]) -> list[str] | None:
    task_presence = {}
    task_names = {}
    
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "shortlists_by_use_case" in data:
                task_items = data["shortlists_by_use_case"]
            else:
                task_items = data
            for entry in task_items:
                t = entry.get("task")
                t_id = ""
                t_name = ""
                if isinstance(t, dict):
                    t_id = t.get("id", "")
                    t_name = t.get("name", t_id)
                elif t:
                    t_id = t
                    t_name = t
                
                if t_id:
                    if t_id not in task_presence:
                        task_presence[t_id] = set()
                    task_presence[t_id].add(path)
                    task_names[t_id] = t_name
        except Exception as exc:
            console.print(f"[red]Failed to load tasks from {Path(path).name}: {exc}[/red]")

    if not task_presence:
        console.print("[red]No tasks found in the selected JSON(s).[/red]")
        return None

    sorted_tasks = sorted(list(task_presence.keys()))
    use_all = _ask_or_exit(questionary.confirm(f"Evaluate all {len(sorted_tasks)} unique tasks across all selected JSONs?", default=True))
    
    selected_tasks = sorted_tasks
    if not use_all:
        selected_tasks = _ask_or_exit(questionary.checkbox(
            "Select tasks to evaluate:",
            choices=[questionary.Choice(title=f"{t_id} ({task_names[t_id]})", value=t_id) for t_id in sorted_tasks]
        ))
        if not selected_tasks:
            console.print("[red]No tasks selected.[/red]")
            return None

    # Check for warnings
    warnings = []
    for t_id in selected_tasks:
        present_in = task_presence[t_id]
        missing_from = [Path(p).name for p in json_paths if p not in present_in]
        if missing_from:
            warnings.append((t_id, missing_from))

    if warnings:
        console.print("\n[bold orange3]⚠ WARNING: Some selected tasks are not present in all selected JSONs:[/bold orange3]")
        for t_id, missing_files in warnings:
            files_str = ", ".join(missing_files)
            console.print(f"  • Task [yellow]'{t_id}'[/yellow] is missing from: {files_str}")
        console.print()
        proceed = _ask_or_exit(questionary.confirm("Do you want to proceed with the evaluation anyway?", default=True))
        if not proceed:
            console.print("[yellow]Evaluation cancelled by user.[/yellow]")
            sys.exit(0)

    return None if use_all else selected_tasks


def step_configure_params() -> dict:
    top_k = _ask_or_exit(questionary.text(f"Top-K components per task (default: {DEFAULT_TOP_K}):", default=str(DEFAULT_TOP_K)))
    max_samples = _ask_or_exit(questionary.text(f"Max benchmark samples per task (default: {DEFAULT_MAX_SAMPLES}):", default=str(DEFAULT_MAX_SAMPLES)))
    concurrency = _ask_or_exit(questionary.text(f"Max concurrent litellm requests (default: {DEFAULT_CONCURRENCY}):", default=str(DEFAULT_CONCURRENCY)))
    skip_baselines = _ask_or_exit(questionary.confirm("Skip evaluating baseline prompts?", default=False))

    return {
        "top_k": _parse_positive_int(top_k, DEFAULT_TOP_K, "Top-K"),
        "max_samples": _parse_positive_int(max_samples, DEFAULT_MAX_SAMPLES, "Max samples"),
        "concurrency": _parse_positive_int(concurrency, DEFAULT_CONCURRENCY, "Concurrency"),
        "skip_baselines": skip_baselines
    }


def step_select_models() -> dict:
    models_cfg = load_yaml("config/models.yaml")
    synth_default = models_cfg["defaults"]["synthesizer"]
    eval_default = models_cfg["defaults"]["evaluator"]

    choices = [questionary.Choice(f"{key}", value=key) for key in models_cfg.get("model_aliases", {}).keys()]

    synth_choices = [questionary.Choice(f"Use default ({synth_default})", value="__default_synth__")] + choices + [questionary.Choice("Custom Model ID...", value="custom")]
    eval_choices = [questionary.Choice(f"Use default ({eval_default})", value="__default_eval__")] + choices + [questionary.Choice("Custom Model ID...", value="custom")]

    synth_key = _ask_or_exit(questionary.select(f"Synthesizer Model (default: {synth_default}):", choices=synth_choices))
    synth_model = synth_default if synth_key == "__default_synth__" else _ask_or_exit(questionary.text("Synthesizer litellm string:")) if synth_key == "custom" else synth_key

    eval_key = _ask_or_exit(questionary.select(f"Evaluator Model (default: {eval_default}):", choices=eval_choices))
    eval_model = eval_default if eval_key == "__default_eval__" else _ask_or_exit(questionary.text("Evaluator litellm string:")) if eval_key == "custom" else eval_key

    return {"synth_model": synth_model, "eval_model": eval_model}


def run_gui():
    console.print(Panel.fit(
        "[bold cyan]🧪 Testing Pipeline V3 — Interactive CLI[/bold cyan]\n"
        "[dim]Log-probability prompt benchmarking[/dim]",
        border_style="cyan",
    ))

    console.print("\n[bold]Step 1:[/bold] Select source file\n")
    source_paths = step_select_source()
    if not source_paths: sys.exit(1)
    console.print(f"  → [green]{len(source_paths)} source JSON(s) selected[/green]\n")

    console.print("[bold]Step 2:[/bold] Select tasks / use cases\n")
    task_filter = step_select_tasks(source_paths)
    if task_filter: console.print(f"  → [green]{len(task_filter)} tasks selected[/green]\n")
    else: console.print(f"  → [green]All tasks[/green]\n")

    console.print("[bold]Step 3:[/bold] Configure execution parameters\n")
    params = step_configure_params()
    console.print(f"  → top_k={params['top_k']}, max_samples={params['max_samples']}, concurrency={params['concurrency']}, skip_baselines={params['skip_baselines']}\n")

    console.print("[bold]Step 4:[/bold] Select LLM endpoints\n")
    model_cfg = step_select_models()
    console.print(f"  → synthesizer={model_cfg['synth_model']}\n  → evaluator={model_cfg['eval_model']}\n")

    console.print()
    table = Table(title="Evaluation Configuration", show_lines=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Backend", "Log-Prob Scoring (logprobs=True)")
    table.add_row(
        "Source JSON",
        source_paths[0] if len(source_paths) == 1 else ", ".join(Path(p).name for p in source_paths),
    )
    table.add_row("Tasks", str(task_filter or "All"))
    table.add_row("Top-K Components", str(params["top_k"]))
    table.add_row("Max Samples", str(params["max_samples"]))
    table.add_row("Concurrency", str(params["concurrency"]))
    table.add_row("Skip Baselines", str(params["skip_baselines"]))
    table.add_row("Synthesizer Model", str(model_cfg["synth_model"]))
    table.add_row("Evaluator Model", str(model_cfg["eval_model"]))
    console.print(table)
    console.print()

    proceed = _ask_or_exit(questionary.confirm("Launch evaluation?", default=True))
    if not proceed:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)

    console.print("\n[bold cyan]Starting Log-Prob evaluation…[/bold cyan]\n")

    # 1. Load Configurations
    try:
        datasets_cfg = load_yaml("config/datasets.yaml")
        models_cfg = load_yaml("config/models.yaml")
    except Exception as e:
        console.print(f"[red]Failed to load configs: {e}[/red]")
        return

    synth_cfg = _resolve_model_config(model_cfg["synth_model"], models_cfg)
    eval_cfg = _resolve_model_config(model_cfg["eval_model"], models_cfg)

    for source_path in source_paths:
        console.print(f"[bold cyan]Source:[/bold cyan] {source_path}")
        _evaluate_source_json(
            source_path=source_path,
            task_filter=task_filter,
            params=params,
            model_cfg=model_cfg,
            datasets_cfg=datasets_cfg,
            synth_cfg=synth_cfg,
            eval_cfg=eval_cfg,
        )

    console.print("\n[bold green]✓ Run complete[/bold green]")

if __name__ == "__main__":
    run_gui()
