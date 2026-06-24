from typing import List, Dict

def build_baseline_prompts(task: str) -> Dict[str, str]:
    """Returns a dictionary of standard baseline prompts for the given task."""
    return {
        "Zero-Shot": (
            f"You are solving the following task: {task}.\n\n"
            "Input:\n{INPUT}\n\n"
            "Answer:"
        ),
        "CoT": (
            f"You are solving the following task: {task}.\n\n"
            "Input:\n{INPUT}\n\n"
            "Let's think step by step to derive the correct answer.\nAnswer:"
        ),
        "ReAct": (
            f"You are solving the following task: {task}.\n\n"
            "Input:\n{INPUT}\n\n"
            "Use Thought, Action, and Observation interleaving if required, but output ONLY the final answer.\nAnswer:"
        ),
        "Expert-Persona": (
            f"You are a leading domain expert solving the following task: {task}.\n\n"
            "Input:\n{INPUT}\n\n"
            "Provide a precise, verified, and strictly correct answer.\nAnswer:"
        )
    }

import textwrap
from src.models import async_predict_single

def _build_meta_prompt(task: str, components: List[str]) -> str:
    lines = [f"  • {comp}" for comp in components]
    component_block = "\n".join(lines)
    
    return textwrap.dedent(f"""
        You are an expert prompt engineer. Your task is to write a single, high-quality
        prompt template that will be used to evaluate a language model on the task below.

        ══════════════════════════════════════════
        TASK: {task}
        ══════════════════════════════════════════

        The prompt MUST naturally incorporate ALL of the following prompt-engineering
        components:

        {component_block}

        ══════════════════════════════════════════
        STRICT REQUIREMENTS FOR YOUR OUTPUT:
        ══════════════════════════════════════════
        1. Output ONLY the prompt text — no preamble, no commentary, no markdown fences.
        2. Use the exact token  {{INPUT}}  (with curly braces) as the placeholder where
           the actual task question / input will be inserted at runtime.
        3. The prompt must be self-contained: a model receiving it has everything it needs.
        4. Guide the model through each component in a natural, flowing, step-by-step way.
        5. End with a clear OUTPUT FORMAT specification (e.g. "Final Answer: ...").
        6. Target length: 150–300 words. Be thorough but not padded.
        7. Ensure that you keep the prompt as minimal as possible while integrating all the components. DO NOT BE overly verbose.
        8. The generated prompt must require the evaluated model to output ONLY the final answer and must not request explanations, reasoning steps, or any additional text.
        9. You must explicitly instruct the model to not use any formatting like **, _, etc.
    """).strip()

import json
import os
import hashlib

CACHE_DIR = "config/prompt_cache"

_accessed_cache_files = set()

def start_tracking_cache():
    global _accessed_cache_files
    _accessed_cache_files.clear()

def get_tracked_cache_files() -> list:
    return list(_accessed_cache_files)

def _get_cache_key(task: str, components: List[str], model_name: str) -> str:
    # Use task, sorted components list, and model identifier to generate a unique key
    stable_components = sorted(components)
    data = {
        "task": task,
        "components": stable_components,
        "model": model_name
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _load_cached_prompt(cache_key: str) -> str | None:
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        _accessed_cache_files.add(os.path.abspath(cache_file))
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("prompt")
                elif isinstance(data, str):
                    return data
        except Exception:
            return None
    return None

def _save_cached_prompt(cache_key: str, task: str, components: List[str], model_name: str, prompt: str, source_info: dict = None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    _accessed_cache_files.add(os.path.abspath(cache_file))
    data = {
        "task": task,
        "components": sorted(components),
        "model": model_name,
        "prompt": prompt
    }
    if source_info:
        data["source_info"] = source_info
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save prompt cache for key {cache_key}: {e}")

async def async_build_synthesized_prompt(task: str, components: List[str], synth_model_cfg: dict, source_info: dict = None) -> str:
    """Builds a novel synthesized prompt combining shortlisted graph attributes via an LLM."""
    if not components:
        return build_baseline_prompts(task)["Zero-Shot"]

    model_name = synth_model_cfg.get("model", "unknown")
    cache_key = _get_cache_key(task, components, model_name)

    cached_prompt = _load_cached_prompt(cache_key)
    if cached_prompt is not None:
        print(f"Loading cached prompt template for task '{task}'...")
        return cached_prompt
        
    meta_prompt = _build_meta_prompt(task, components)
    print(f"Synthesizing prompt template using {model_name}...")
    generated = await async_predict_single(meta_prompt, synth_model_cfg)
    
    if not generated:
        print(f"Warning: Synthesized prompt from {model_name} is empty! Using fallback.")
        return build_baseline_prompts(task)["Zero-Shot"]
        
    # Normalize double-braced {{INPUT}} that the LLM may produce
    generated = generated.replace("{{INPUT}}", "{INPUT}")
    generated = generated.replace("{{input}}", "{INPUT}")

    if "{INPUT}" not in generated and "{input}" not in generated:
        print(f"Warning: Synthesized prompt from {model_name} missing {{INPUT}} placeholder. Automatically appending it.")
        generated += "\n\nInput:\n{INPUT}\n\nAnswer:"
        
    # Standardize placeholder if model used lowercase
    final_prompt = generated.replace("{input}", "{INPUT}")

    # Save to cache
    _save_cached_prompt(cache_key, task, components, model_name, final_prompt, source_info)

    return final_prompt

