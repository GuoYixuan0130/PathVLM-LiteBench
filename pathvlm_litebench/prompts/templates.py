from __future__ import annotations


PATHOLOGY_PROMPT_TEMPLATES: dict[str, list[str]] = {
    "tumor": [
        "a histopathology image of tumor tissue",
        "a pathology patch showing malignant tissue",
        "a microscopic image of cancerous tissue",
        "a H&E stained tissue patch with tumor region",
    ],
    "normal": [
        "a histopathology image of normal tissue",
        "a pathology patch showing non-tumor tissue",
        "a microscopic image of benign tissue",
        "a H&E stained tissue patch without tumor",
    ],
    "necrosis": [
        "a histopathology image showing necrosis",
        "a pathology patch with necrotic tissue",
        "a microscopic image of necrotic region",
        "a H&E stained tissue patch with necrosis",
    ],
    "inflammation": [
        "a histopathology image showing inflammation",
        "a pathology patch with inflammatory cells",
        "a microscopic image with inflammatory infiltration",
        "a H&E stained tissue patch showing inflammation",
    ],
    "stroma": [
        "a histopathology image of stromal tissue",
        "a pathology patch showing fibrous stroma",
        "a microscopic image of connective tissue stroma",
        "a H&E stained tissue patch with stromal region",
    ],
    "lymphocyte": [
        "a histopathology image with lymphocytes",
        "a pathology patch showing lymphocyte infiltration",
        "a microscopic image with dense lymphocytic cells",
        "a H&E stained tissue patch containing lymphocytes",
    ],
    "gland": [
        "a histopathology image showing glandular structures",
        "a pathology patch with glandular tissue",
        "a microscopic image of gland formation",
        "a H&E stained tissue patch with glands",
    ],
}


def list_prompt_concepts() -> list[str]:
    """
    List available built-in pathology prompt concepts.
    """
    return sorted(PATHOLOGY_PROMPT_TEMPLATES.keys())


def get_prompt_variants(concept: str) -> list[str]:
    """
    Get prompt variants for a built-in pathology concept.

    Args:
        concept:
            Concept name, such as "tumor" or "necrosis".

    Returns:
        A copy of the prompt variant list.
    """
    concept_key = concept.lower()

    if concept_key not in PATHOLOGY_PROMPT_TEMPLATES:
        available = ", ".join(list_prompt_concepts())
        raise ValueError(
            f"Unknown prompt concept: '{concept}'. "
            f"Available concepts: {available}."
        )

    return list(PATHOLOGY_PROMPT_TEMPLATES[concept_key])


def build_class_prompts(
    class_names: list[str],
    template: str = "a histopathology image of {class_name}",
) -> list[str]:
    """
    Build one prompt per class name using a prompt template.

    Args:
        class_names:
            Class names such as ["tumor", "normal", "necrosis"].
        template:
            Template string containing "{class_name}".

    Returns:
        A list of formatted prompts.
    """
    if len(class_names) == 0:
        raise ValueError("class_names must not be empty.")

    if "{class_name}" not in template:
        raise ValueError("template must contain '{class_name}'.")

    return [template.format(class_name=class_name) for class_name in class_names]


def build_prompt_groups(
    concepts: list[str] | None = None,
) -> tuple[list[str], list[list[str]]]:
    """
    Build grouped prompt variants for prompt sensitivity analysis.

    Args:
        concepts:
            Optional list of concept names. If None, all built-in concepts are used.

    Returns:
        concept_names:
            List of concept names.
        prompt_texts_by_concept:
            List of prompt variant lists.
    """
    if concepts is None:
        concept_names = list_prompt_concepts()
    else:
        concept_names = [concept.lower() for concept in concepts]

    prompt_texts_by_concept = [
        get_prompt_variants(concept) for concept in concept_names
    ]

    return concept_names, prompt_texts_by_concept
