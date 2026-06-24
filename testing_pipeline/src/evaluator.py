"""Evaluation engine using log-probability scoring.

For MCQ tasks:
    Uses first-token log-prob ranking — generates only 1 token per sample,
    then picks the answer choice with the highest probability.

For other tasks:
    Generates a full answer with logprobs=True, scores with text-based
    metrics, and also captures mean log-prob as a confidence signal.
"""

import asyncio
import re
import time
from typing import Dict, Any, List

from src.prompt_builder import build_baseline_prompts, async_build_synthesized_prompt
from src.models import (
    async_predict_with_logprobs,
    extract_choice_logprobs,
)
from src.dataset_loader import load_hf_dataset
from src.metrics import score_single, _extract_mcq_choice, _normalize_text
from src.logger import save_detailed_logs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_prompt(template: str, question: str) -> str:
    """Insert *question* into a prompt template."""
    if not template:
        return question
    if "{INPUT}" in template:
        return template.replace("{INPUT}", question)
    return f"{template}\n\n{question}"


def _infer_valid_choices_from_input(question_text: str) -> List[str] | None:
    """Infer option letters from MCQ-formatted text when choice metadata is unavailable."""
    if not question_text:
        return None

    matches = re.findall(r"(?:^|\n)\s*[\(\[]?([A-Z])[\)\].:]\s+", str(question_text).upper())
    ordered: List[str] = []
    seen = set()
    for m in matches:
        if m not in seen:
            ordered.append(m)
            seen.add(m)

    return ordered if len(ordered) >= 2 else None


# ---------------------------------------------------------------------------
# MCQ evaluation via log-prob ranking
# ---------------------------------------------------------------------------

async def _evaluate_mcq_logprob(
    template: str,
    data: List[Dict],
    model_config: dict,
    concurrency: int,
) -> Dict[str, Any]:
    """Score MCQ samples by comparing first-token log-probabilities."""
    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(idx: int, item: dict) -> dict:
        async with semaphore:
            prompt = _wrap_prompt(template, item["generated_input"])
            result = await async_predict_with_logprobs(
                prompt, model_config, max_tokens=1, top_logprobs=20,
            )

            valid_choices = item.get("valid_choices")
            if isinstance(valid_choices, list) and valid_choices:
                valid_choices = [str(c).strip().upper() for c in valid_choices if str(c).strip()]
            else:
                valid_choices = None

            if not valid_choices:
                valid_choices = _infer_valid_choices_from_input(item.get("generated_input", ""))

            raw_target = str(item.get("target", "")).strip()
            # Normalize target to bare letter — handles "(A)", "A", "1" → "A" etc.
            target = _extract_mcq_choice(raw_target) if raw_target else ""

            # Try log-prob ranking first
            choice_lps = extract_choice_logprobs(result["logprobs"], valid_choices=valid_choices)

            if choice_lps:
                best_choice = max(choice_lps, key=choice_lps.get)
                correct = 1.0 if best_choice == target else 0.0
            else:
                # Fallback: parse the generated text
                pred_choice = _extract_mcq_choice(result["text"])
                if valid_choices and pred_choice not in valid_choices:
                    pred_choice = ""
                best_choice = pred_choice
                correct = 1.0 if pred_choice == target else 0.0

            return {
                "index": idx,
                "correct": correct,
                "predicted": best_choice,
                "target": target,
                "raw_target": raw_target,
                "valid_choices": valid_choices,
                "choice_logprobs": choice_lps,
                "generated_text": result["text"],
                "mean_logprob": result["mean_logprob"],
            }

    tasks = [score_one(i, d) for i, d in enumerate(data)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x["index"])

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0

    return {
        "score": round(accuracy, 4),
        "metric": "logprob_mcq_accuracy",
        "num_samples": len(results),
        "per_sample": results,
    }


# ---------------------------------------------------------------------------
# Generative evaluation (exact-match, token-F1, classification, etc.)
# ---------------------------------------------------------------------------

async def _evaluate_generative(
    template: str,
    data: List[Dict],
    model_config: dict,
    metric_type: str,
    concurrency: int,
) -> Dict[str, Any]:
    """Generate full answers with logprobs, then score with text metrics."""
    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(idx: int, item: dict) -> dict:
        async with semaphore:
            prompt = _wrap_prompt(template, item["generated_input"])
            result = await async_predict_with_logprobs(
                prompt, model_config, max_tokens=256, top_logprobs=5,
            )

            generated = result["text"]
            target = item.get("target", "")

            score = score_single(metric_type, generated, target)

            return {
                "index": idx,
                "score": score,
                "generated": generated[:500],
                "target": str(target)[:200],
                "mean_logprob": result["mean_logprob"],
            }

    tasks = [score_one(i, d) for i, d in enumerate(data)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x["index"])

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    avg_logprob = sum(r["mean_logprob"] for r in results) / len(results) if results else 0.0

    return {
        "score": round(avg_score, 4),
        "mean_logprob": round(avg_logprob, 4),
        "metric": metric_type,
        "num_samples": len(results),
        "per_sample": results,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def evaluate_task(
    task_name: str,
    components: List[str],
    dataset_cfg: dict,
    eval_model_cfg: dict,
    synth_model_cfg: dict,
    synthesis_task_name: str | None = None,
    run_dir: str = None,
    max_samples: int = 50,
    concurrency: int = 5,
    skip_baselines: bool = False,
    source_info: dict = None,
) -> Dict[str, Any]:
    """Run comparative evaluation for one task using log-prob scoring.

    Builds a synthesized prompt + baselines, loads the dataset, and
    evaluates each prompt method side-by-side.
    """
    benchmark_name = dataset_cfg.get("benchmark_name", "unknown")
    metric_type = dataset_cfg.get("metric_type", "exact_match_normalized")

    print(f"\n[Eval] Task: {task_name} | Benchmark: {benchmark_name} | Metric: {metric_type}")

    # ── Build prompt templates ────────────────────────────────────────
    prompt_task = synthesis_task_name or task_name
    print(f"[Eval] Synthesizing prompt for: {prompt_task}")
    synth_template = await async_build_synthesized_prompt(
        prompt_task, components, synth_model_cfg, source_info=source_info
    )

    methods = {"Synthesized": synth_template}
    if not skip_baselines:
        baseline_templates = build_baseline_prompts(prompt_task)
        methods.update(baseline_templates)

    # ── Load dataset ──────────────────────────────────────────────────
    hf_repo = dataset_cfg.get("hf_repo", "")
    subset = dataset_cfg.get("subset", "")
    print(f"[Eval] Loading dataset: {hf_repo} / {subset}")
    try:
        data = load_hf_dataset(dataset_cfg, max_samples)
    except Exception as e:
        print(f"[Eval] ERROR loading dataset: {e}")
        return {"error": str(e), "task": task_name}

    print(f"[Eval] Loaded {len(data)} samples")

    results: Dict[str, Any] = {
        "task": task_name,
        "benchmark_name": benchmark_name,
        "hf_repo": hf_repo,
        "subset": subset,
        "metric_type": metric_type,
        "max_samples": max_samples,
        "num_samples": len(data),
        "prompts_used": methods,
        "scores": {},
    }

    # ── Evaluate each prompt method ───────────────────────────────────
    for method_name, template in methods.items():
        print(f"\n[Eval] === {method_name} === ({benchmark_name})")
        t0 = time.time()

        if metric_type == "mcq_generative":
            method_result = await _evaluate_mcq_logprob(
                template, data, eval_model_cfg, concurrency,
            )
        else:
            method_result = await _evaluate_generative(
                template, data, eval_model_cfg, metric_type, concurrency,
            )

        elapsed = time.time() - t0
        method_result["elapsed_s"] = round(elapsed, 2)

        # Save detailed per-sample logs
        if run_dir:
            try:
                save_detailed_logs(
                    run_dir, task_name, method_name,
                    method_result.get("per_sample", []),
                )
            except Exception as e:
                print(f"[Eval] Warning: could not save detailed logs: {e}")

        # Keep results.json compact — remove per_sample from top-level
        method_result.pop("per_sample", None)
        results["scores"][method_name] = method_result

        print(f"[Eval] {method_name} -> Score: {method_result['score']:.4f} ({elapsed:.1f}s)")

    return results
