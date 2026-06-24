import argparse
import sys
import yaml
import json
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from src.evaluator import evaluate_task
from src.logger import make_run_dir, save_results, save_prompts, save_metadata, _sanitize_path_token

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
         return yaml.safe_load(f)

def _resolve_model_config(choice: str, models_yaml: dict) -> dict:
    if choice in models_yaml.get("model_aliases", {}):
        cfg = dict(models_yaml["model_aliases"][choice] or {})
    else:
        # Default all custom models to temperature: 0 and seed: 42 to guarantee determinism
        is_openai_like = any(x in choice.lower() for x in ["ollama", "llama", "gemma", "granite", "local"])
        cfg = {
            "model": choice,
            "provider": "openai_compatible" if is_openai_like else "unknown",
            "temperature": 0,
            "seed": 42
        }

    if not cfg.get("api_base") and cfg.get("base_url"):
        cfg["api_base"] = cfg["base_url"]

    if not cfg.get("api_base") and os.getenv("OLLAMA_BASE_URL"):
        cfg["api_base"] = os.getenv("OLLAMA_BASE_URL")

    api_key_env = cfg.get("api_key_env")
    if api_key_env and not cfg.get("api_key"):
        env_key = os.getenv(str(api_key_env))
        if env_key:
           cfg["api_key"] = env_key

    return cfg

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

async def main_async(args):
    # 1. Load Configurations
    try:
        datasets_cfg = load_yaml("config/datasets.yaml")
        models_cfg = load_yaml("config/models.yaml")
    except Exception as e:
        print(f"Failed to load configs: {e}")
        return

    # Load source JSON
    try:
        with open(args.source_json, "r", encoding="utf-8") as f:
            source_data = json.load(f)
    except Exception as e:
         print(f"Failed to load source JSON: {e}")
         return

    # Normalize V3 hypergraph format
    if isinstance(source_data, dict) and "shortlists_by_use_case" in source_data:
        source_data = source_data["shortlists_by_use_case"]

    # Resolve Models
    synth_model_choice = args.synth_model or models_cfg["defaults"]["synthesizer"]
    eval_model_choice = args.eval_model or models_cfg["defaults"]["evaluator"]

    synth_cfg = _resolve_model_config(synth_model_choice, models_cfg)
    eval_cfg = _resolve_model_config(eval_model_choice, models_cfg)

    source_origin = _source_origin_info(args.source_json)
    source_model_prefix = source_origin.get("source_model")
    print("Starting Log-Prob Pipeline — per-task results → results/<task>/run_<timestamp>")

    def _get_task_id(item: dict) -> str:
        task = item.get("task")
        if isinstance(task, dict):
            return task.get("id", "")
        return task or ""

    selected_tasks = []
    if args.tasks:
        target_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
        selected_tasks = [t for t in source_data if _get_task_id(t) in target_tasks]
    else:
        selected_tasks = source_data

    if not selected_tasks:
        print("No matched tasks found in source JSON.")
        return

    for item in selected_tasks:
        task_name = _get_task_id(item)

        dataset_task_key = _resolve_dataset_task_key(task_name, datasets_cfg)
        if not dataset_task_key:
             print(f"Skipping task {task_name} (Not found in datasets.yaml mapping)")
             continue
        if dataset_task_key != task_name:
            print(f"Mapped task label '{task_name}' -> '{dataset_task_key}'")

        task_ds_cfg = datasets_cfg["datasets"][dataset_task_key]
        benchmark_cfgs = task_ds_cfg.get("benchmarks") if isinstance(task_ds_cfg, dict) else None
        if benchmark_cfgs is None:
            benchmark_cfgs = [task_ds_cfg]

        task_dir_name = _safe_task_dir_name(task_name)
        evaluator_model_dir = _sanitize_path_token(eval_model_choice)
        task_base_out_dir = os.path.join("results", task_dir_name, evaluator_model_dir)
        task_run_dir = make_run_dir(task_base_out_dir, run_prefix=source_model_prefix)
        print(f"Task output directory: {task_run_dir}")

        task_payload = {
            "source": args.source_json,
            "canonical_task": task_name,
            "synthesizer_model": synth_model_choice,
            "evaluator_model": eval_model_choice,
            "benchmarks": []
        }

        # Extract components
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
        components = [c for c in components if c][:args.top_k]

        print(f"\n{'='*50}\nTask: {task_name}\nComponents: {components}\n{'='*50}")

        for idx, benchmark_cfg in enumerate(benchmark_cfgs, start=1):
            if isinstance(benchmark_cfg, dict) and benchmark_cfg.get("enabled") is False:
                benchmark_name = benchmark_cfg.get("benchmark_name", f"benchmark_{idx}")
                print(f"Skipping disabled benchmark: {benchmark_name}")
                continue

            benchmark_name = benchmark_cfg.get("benchmark_name", f"benchmark_{idx}")
            print(f"Running benchmark [{idx}/{len(benchmark_cfgs)}]: {benchmark_name}")

            task_results = await evaluate_task(
                task_name=f"{task_name} :: {benchmark_name}",
                components=components,
                dataset_cfg=benchmark_cfg,
                eval_model_cfg=eval_cfg,
                synth_model_cfg=synth_cfg,
                synthesis_task_name=task_name,
                run_dir=task_run_dir,
                max_samples=args.samples,
                concurrency=args.concurrency,
                skip_baselines=args.skip_baselines,
                source_info=source_origin,
            )

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
                "synthesizer": synth_model_choice,
                "evaluator": eval_model_choice,
            },
            "params": {
                "top_k": args.top_k,
                "samples": args.samples,
                "concurrency": args.concurrency,
            },
            "benchmarks_requested": len(benchmark_cfgs),
            "benchmarks_completed": len(task_payload["benchmarks"]),
        })
        print(f"Saved task results: {os.path.join(task_run_dir, 'results.json')}")

    print("\nDone! Results saved under results/<task>/run_<timestamp>/results.json")

def main():
    parser = argparse.ArgumentParser(description="Testing Pipeline V3 — Log-Prob Evaluation")
    parser.add_argument("--source-json", type=str, required=True, help="Path to Predicted Links Output JSON")
    parser.add_argument("--samples", type=int, default=10000, help="Max benchmark samples per task")
    parser.add_argument("--top-k", type=int, default=15, help="Top K components per task")
    parser.add_argument("--synth-model", type=str, default=None, help="Alias or litellm model string for Prompt Synthesis")
    parser.add_argument("--eval-model", type=str, default=None, help="Alias or litellm model string for evaluation")
    parser.add_argument("--tasks", type=str, default=None, help="Comma separated list of tasks to evaluate, default=all")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent litellm requests")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip evaluating baseline prompts")

    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
