# pyrefly: ignore [missing-import]
from datasets import load_dataset
from typing import Dict, Any, List

def _get_case_insensitive(item: Dict[str, Any], key: str) -> Any:
    """Retrieves a value from a dictionary by key name case-insensitively, supporting dot notation for nested keys."""
    if not isinstance(item, dict):
        return None
    if "." in key:
        parts = key.split(".", 1)
        first_val = _get_case_insensitive(item, parts[0])
        if isinstance(first_val, dict):
            return _get_case_insensitive(first_val, parts[1])
        return None
    val = item.get(key)
    if val is not None:
        return val
    key_lower = key.lower()
    for k, v in item.items():
        if k.lower() == key_lower:
            return v
    return None

def load_hf_dataset(config: Dict[str, Any], max_samples: int = None) -> List[Dict[str, Any]]:
    """Loads a HuggingFace dataset and returns a list of dictionaries based on the configuration."""
    repo = config.get("hf_repo")
    subset = config.get("subset")
    split = config.get("split", "test")
    input_keys = config.get("input_keys", ["question"])
    choice_keys = config.get("choice_keys", [])
    answer_key = config.get("answer_key")
    metric_type = config.get("metric_type")
    label_map_text = config.get("label_map_text") or config.get("prompt_hint")
    revision = config.get("revision")

    if not repo:
        raise ValueError(
            "Dataset configuration error: missing 'hf_repo'. "
            "If using multi-benchmark mappings, pass an individual benchmark config to load_hf_dataset()."
        )

    answer_index_base = config.get("answer_index_base", 0)

    try:
        kwargs = {}
        if revision:
            kwargs["revision"] = revision
        
        # Support both a single subset or a list of subsets
        subsets = subset if isinstance(subset, list) else [subset]
        
        datasets_list = []
        for sub in subsets:
            if sub and sub != "default" and sub != "all":
                 dataset = load_dataset(repo, sub, split=split, **kwargs)
            else:
                 dataset = load_dataset(repo, split=split, **kwargs)
            datasets_list.append(dataset)
    except Exception as e:
        raise ValueError(f"Failed to load dataset {repo} ({subset}): {e}")

    filter_cfg = config.get("filter")

    # Convert to list and slice
    data_list = []
    if max_samples and max_samples > 0:
        # Apportion max_samples across subsets to ensure representativeness
        per_subset_limit = max(1, (max_samples + len(datasets_list) - 1) // len(datasets_list))
        for d in datasets_list:
            sub_list = list(d)
            if filter_cfg:
                sub_list = [
                    item for item in sub_list
                    if all(
                        _get_case_insensitive(item, fk) in fv if isinstance(fv, list)
                        else _get_case_insensitive(item, fk) == fv
                        for fk, fv in filter_cfg.items()
                    )
                ]
            data_list.extend(sub_list[:per_subset_limit])
        data_list = data_list[:max_samples]
    else:
        for d in datasets_list:
            sub_list = list(d)
            if filter_cfg:
                sub_list = [
                    item for item in sub_list
                    if all(
                        _get_case_insensitive(item, fk) in fv if isinstance(fv, list)
                        else _get_case_insensitive(item, fk) == fv
                        for fk, fv in filter_cfg.items()
                    )
                ]
            data_list.extend(sub_list)

    structured_data = []
    for item in data_list:
        # Build the final input text string
        input_text = ""
        valid_choices = None
        # Handle special MMLU / HellaSwag MCQ combinations
        if metric_type == "mcq_generative":
            # Build the main prompt body from all provided input keys.
            prompt_parts = []
            for k in input_keys:
                val = _get_case_insensitive(item, k)
                if val is not None:
                    prompt_parts.append(str(val))
            input_text += "\n\n".join([p for p in prompt_parts if p]).strip() + "\n\n"

            # Resolve choices from explicit choice_keys first, then legacy second input key.
            choices = None
            if isinstance(choice_keys, list) and choice_keys:
                first_choice_val = _get_case_insensitive(item, choice_keys[0])
                if len(choice_keys) == 1 and isinstance(first_choice_val, list):
                    choices = first_choice_val
                else:
                    choices = []
                    for k in choice_keys:
                        val = _get_case_insensitive(item, k)
                        choices.append(val if val is not None else k)
            elif len(input_keys) > 1:
                choices_key = input_keys[1]
                choices = _get_case_insensitive(item, choices_key)

            # If dict style choices (truthfulQA, ARC, CommonsenseQA, etc.)
            if isinstance(choices, dict):
                if "choices" in choices:
                    choices = choices["choices"]
                elif "text" in choices:
                    choices = choices["text"]

            if isinstance(choices, list):
                options = []
                for i, c in enumerate(choices):
                    letter = chr(ord('A') + i)
                    options.append(f"{letter}. {c}")
                valid_choices = [chr(ord('A') + i) for i in range(len(choices))]
                input_text += "Options:\n" + "\n".join(options) + "\n\n"
                input_text += "Respond with ONLY the correct option letter (e.g. 'A')."
        elif metric_type == "classification":
            # Classification task - output numeric label (0, 1, 2, etc)
            primary_key = input_keys[0]
            primary_val = _get_case_insensitive(item, primary_key)
            input_text += str(primary_val or "") + "\n\n"
            if label_map_text:
                input_text += str(label_map_text).strip() + "\n"
            input_text += "Classify and respond with ONLY the numeric label (0, 1, 2, etc)."
        else:
            # Simple concatenation for standard inputs
            input_text = "\n\n".join([str(_get_case_insensitive(item, k) or "") for k in input_keys])

        target = None
        if answer_key:
            target = _get_case_insensitive(item, answer_key)
            if metric_type == "mcq_generative" and isinstance(target, int):
                # Convert gold label to letter (adjusting for index base)
                target = chr(ord('A') + (target - answer_index_base))
            elif metric_type == "mcq_generative" and isinstance(target, str) and target.isdigit():
                # Convert string label to letter (adjusting for index base)
                target = chr(ord('A') + (int(target) - answer_index_base))
            elif metric_type == "mcq_generative" and repo in ("truthful_qa", "truthfulqa/truthful_qa"):
                 # Extract the single index from labels dict
                 labels = target.get("labels", []) if isinstance(target, dict) else []
                 for i, l in enumerate(labels):
                     if int(l) == 1:
                         target = chr(ord('A') + i)
                         break
            elif metric_type == "classification" and isinstance(target, int):
                # Keep as string for easy comparison
                target = str(target)

        structured_data.append({
            "generated_input": input_text.strip(),
            "target": target,
            "valid_choices": valid_choices,
            "raw": item
        })

    return structured_data
