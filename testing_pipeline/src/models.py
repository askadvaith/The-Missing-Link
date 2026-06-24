"""Model interface for litellm-based inference with log-probability support.

Provides:
  - prepare_litellm_request()         – normalize model config to litellm args
  - async_predict_single()            – generate a single completion
  - async_predict_with_logprobs()     – generate with per-token logprobs returned
  - extract_choice_logprobs()         – pick MCQ answer from top log-probs
  - async_batch_predict()             – batched generation with concurrency control
"""

import asyncio
import os
import re
import time
from typing import List, Any, Optional, Dict, Tuple

import litellm

litellm.suppress_debug_info = True
litellm.drop_params = True


def prepare_litellm_request(model_config: dict) -> tuple[str, str | None, str | None, float, dict, dict]:
    """Normalizes model/provider fields into a litellm-compatible request tuple.

    Returns (model_name, api_base, api_key, temperature, extra_body, extra_kwargs).
    """
    model_name = model_config.get("model", "ollama/llama3.2:1b")
    provider = str(model_config.get("provider", "")).lower()
    
    # Resolve local Ollama host if configured in environment
    env_ollama_host = os.getenv("OLLAMA_HOST")
    if env_ollama_host:
        if not (env_ollama_host.startswith("http://") or env_ollama_host.startswith("https://")):
            env_ollama_host = "http://" + env_ollama_host

    api_base = (
        model_config.get("api_base") 
        or model_config.get("base_url")
    )
    api_key = model_config.get("api_key", None)

    api_key_env = model_config.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env))

    # Route Ollama models through the OpenAI-compatible endpoint.
    is_ollama_like = (
        str(model_name).startswith("ollama/")
        or provider == "ollama"
        or (provider == "openai_compatible" and "/" not in str(model_name))
    )
    if is_ollama_like:
        bare_model = str(model_name)[7:] if str(model_name).startswith("ollama/") else str(model_name)
        if not bare_model.startswith("openai/"):
            model_name = "openai/" + bare_model
        else:
            model_name = bare_model
        if not api_base:
            api_base = os.getenv("OLLAMA_BASE_URL") or env_ollama_host or "http://localhost:11434/v1"
        
        if api_base and not str(api_base).rstrip("/").endswith("/v1"):
            api_base = str(api_base).rstrip("/") + "/v1"
            
        if not api_key:
            api_key = "ollama"

    temperature = float(model_config.get("temperature", 0.0))
    
    extra_body = {}
    if "think" in model_config:
        extra_body["think"] = model_config["think"]
        
    extra_kwargs = {}
    if "thinking" in model_config:
        extra_kwargs["thinking"] = model_config["thinking"]
    if "reasoning_effort" in model_config:
        extra_kwargs["reasoning_effort"] = model_config["reasoning_effort"]
        
    return str(model_name), api_base, api_key, temperature, extra_body, extra_kwargs


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

async def async_predict_single(prompt: str, model_config: dict) -> str:
    """Generate a single completion (no logprobs)."""
    model_name, api_base, api_key, temperature, extra_body, extra_kwargs = prepare_litellm_request(model_config)
    seed = model_config.get("seed")
    max_tokens = model_config.get("max_tokens", 4096)

    try:
        response = await litellm.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            extra_body=extra_body if extra_body else None,
            **extra_kwargs
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling model {model_name}: {e}")
        return ""


async def async_predict_with_logprobs(
    prompt: str,
    model_config: dict,
    max_tokens: int = 1,
    top_logprobs: int = 20,
) -> Dict[str, Any]:
    """Generate a completion with per-token log-probabilities.

    Returns::

        {
            "text": str,              # generated text
            "logprobs": [             # list of per-token info
                {
                    "token": str,
                    "logprob": float,
                    "top_logprobs": [{"token": str, "logprob": float}, ...]
                }, ...
            ],
            "mean_logprob": float,    # mean logprob across generated tokens
        }
    """
    model_name, api_base, api_key, _, extra_body, extra_kwargs = prepare_litellm_request(model_config)
    seed = model_config.get("seed")

    try:
        response = await litellm.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            api_base=api_base,
            api_key=api_key,
            temperature=0.0,  # deterministic for evaluation
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=min(top_logprobs, 20),
            seed=seed,
            extra_body=extra_body if extra_body else None,
            **extra_kwargs
        )

        text = (response.choices[0].message.content or "").strip()

        lp_data: List[Dict[str, Any]] = []
        choice_logprobs = getattr(response.choices[0], "logprobs", None)

        if choice_logprobs and hasattr(choice_logprobs, "content") and choice_logprobs.content:
            for token_lp in choice_logprobs.content:
                entry: Dict[str, Any] = {
                    "token": token_lp.token,
                    "logprob": token_lp.logprob,
                    "top_logprobs": [],
                }
                if hasattr(token_lp, "top_logprobs") and token_lp.top_logprobs:
                    entry["top_logprobs"] = [
                        {"token": tlp.token, "logprob": tlp.logprob}
                        for tlp in token_lp.top_logprobs
                    ]
                lp_data.append(entry)

        mean_lp = 0.0
        if lp_data:
            mean_lp = sum(e["logprob"] for e in lp_data) / len(lp_data)

        return {"text": text, "logprobs": lp_data, "mean_logprob": mean_lp}

    except Exception as e:
        print(f"  [Model] Error in logprob completion: {e}")
        return {"text": "", "logprobs": [], "mean_logprob": float("-inf")}


# ---------------------------------------------------------------------------
# Log-prob extraction for MCQ
# ---------------------------------------------------------------------------

def extract_choice_logprobs(
    logprobs_data: List[Dict],
    valid_choices: List[str] | None = None,
) -> Dict[str, float]:
    """Extract log-probabilities for MCQ choice tokens from the first generated token.

    Searches through ``top_logprobs`` at position 0 for tokens that match
    *valid_choices* (default ``["A","B","C","D","E"]``).

    Returns a dict mapping choice letter -> log-probability.
    """
    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D", "E"]

    valid_choices = [str(ch).strip().upper() for ch in valid_choices if str(ch).strip()]
    valid_set = set(valid_choices)
    if not valid_choices or not logprobs_data:
        return {}

    def _parse_choice_token(token: Any) -> str | None:
        raw = str(token or "").strip().upper()
        if not raw:
            return None

        if raw in valid_set:
            return raw

        # Handle single-letter wrappers like "(A)", "A.", "[B]", etc.
        wrapped = re.match(r"^[\(\[]?([A-Z])[\)\].:]?$", raw)
        if wrapped and wrapped.group(1) in valid_set:
            return wrapped.group(1)

        # Handle explicit forms like "Option A" or "Answer: B".
        named = re.match(r"^(?:OPTION|ANSWER)\s*[:\-]?\s*([A-Z])$", raw)
        if named and named.group(1) in valid_set:
            return named.group(1)

        return None

    first_token_info = logprobs_data[0]
    top_lps = first_token_info.get("top_logprobs", [])

    choice_probs: Dict[str, float] = {}
    for tlp in top_lps:
        choice = _parse_choice_token(tlp.get("token", ""))
        if choice and choice not in choice_probs:
            choice_probs[choice] = tlp.get("logprob", float("-inf"))

    # Also check the actual generated token.
    gen_choice = _parse_choice_token(first_token_info.get("token", ""))
    if gen_choice and gen_choice not in choice_probs:
        choice_probs[gen_choice] = first_token_info.get("logprob", float("-inf"))

    return choice_probs


# ---------------------------------------------------------------------------
# Batched generation with concurrency control
# ---------------------------------------------------------------------------

async def async_batch_predict(
    prompts: List[str],
    model_config: dict,
    concurrency: int = 5,
    progress_label: str = "",
) -> List[str]:
    """Run a batch of prompts through the model with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    if not prompts:
        return []

    model_name = model_config.get("model", "unknown-model")
    label = progress_label or "batch"
    total = len(prompts)
    print(f"  [Model] Starting {label}: {total} requests on {model_name} (concurrency={concurrency})")

    async def bounded_predict(index: int, prompt: str) -> tuple[int, str]:
        async with semaphore:
            response = await async_predict_single(prompt, model_config)
            return index, response

    pending = {
        asyncio.create_task(bounded_predict(i, p))
        for i, p in enumerate(prompts)
    }
    outputs = [""] * total
    completed = 0
    t0 = time.time()
    last_progress_ts = t0

    while pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=10.0,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            waited = time.time() - last_progress_ts
            elapsed = time.time() - t0
            print(
                f"  [Model] Waiting on {label}: {completed}/{total} complete, "
                f"{len(pending)} in-flight, +{waited:.1f}s since last completion (elapsed {elapsed:.1f}s)"
            )
            continue

        for task in done:
            idx, pred = await task
            outputs[idx] = pred
            completed += 1
            last_progress_ts = time.time()
            if completed % max(1, total // 5) == 0 or completed == total:
                print(f"  [Model] Completed {label}: {completed}/{total}")

    elapsed = time.time() - t0
    print(f"  [Model] Finished {label}: {total}/{total} in {elapsed:.1f}s")
    return outputs
