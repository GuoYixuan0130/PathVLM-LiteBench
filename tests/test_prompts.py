from pathvlm_litebench.prompts import (
    build_class_prompts,
    build_prompt_groups,
    get_prompt_variants,
    list_prompt_concepts,
)


def test_list_prompt_concepts_contains_expected_items():
    concepts = list_prompt_concepts()
    assert "tumor" in concepts
    assert "normal" in concepts
    assert "necrosis" in concepts


def test_get_prompt_variants_case_insensitive():
    prompts = get_prompt_variants("Tumor")
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    assert any("tumor" in prompt.lower() or "malignant" in prompt.lower() for prompt in prompts)


def test_get_prompt_variants_unknown_concept():
    try:
        get_prompt_variants("unknown_concept")
    except ValueError as exc:
        assert "Unknown prompt concept" in str(exc)
        assert "Available concepts" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown prompt concept.")


def test_build_class_prompts():
    prompts = build_class_prompts(["tumor", "normal"])
    assert prompts == [
        "a histopathology image of tumor",
        "a histopathology image of normal",
    ]


def test_build_class_prompts_custom_template():
    prompts = build_class_prompts(
        ["tumor", "normal"],
        template="a H&E stained patch showing {class_name}",
    )
    assert prompts == [
        "a H&E stained patch showing tumor",
        "a H&E stained patch showing normal",
    ]


def test_build_class_prompts_invalid_template():
    try:
        build_class_prompts(["tumor"], template="a histopathology image")
    except ValueError as exc:
        assert "template must contain" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid template.")


def test_build_prompt_groups_subset():
    concept_names, prompt_groups = build_prompt_groups(["tumor", "necrosis"])
    assert concept_names == ["tumor", "necrosis"]
    assert len(prompt_groups) == 2
    assert all(len(group) > 0 for group in prompt_groups)
