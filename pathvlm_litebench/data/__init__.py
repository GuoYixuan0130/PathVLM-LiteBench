__all__ = [
    "load_patch_images",
    "load_patch_images_from_paths",
    "save_embeddings",
    "load_embeddings",
    "save_metadata",
    "load_metadata",
    "PatchRecord",
    "load_patch_manifest",
    "records_to_image_paths",
    "records_to_labels",
    "get_unique_labels",
    "filter_records_by_split",
    "filter_records_by_label",
    "convert_manifest",
    "convert_mhist_manifest",
    "sample_manifest",
    "summarize_manifest",
]


_LAZY_IMPORTS = {
    "load_patch_images": ("patch_loader", "load_patch_images"),
    "load_patch_images_from_paths": ("patch_loader", "load_patch_images_from_paths"),
    "save_embeddings": ("embedding_cache", "save_embeddings"),
    "load_embeddings": ("embedding_cache", "load_embeddings"),
    "save_metadata": ("embedding_cache", "save_metadata"),
    "load_metadata": ("embedding_cache", "load_metadata"),
    "PatchRecord": ("manifest_loader", "PatchRecord"),
    "load_patch_manifest": ("manifest_loader", "load_patch_manifest"),
    "records_to_image_paths": ("manifest_loader", "records_to_image_paths"),
    "records_to_labels": ("manifest_loader", "records_to_labels"),
    "get_unique_labels": ("manifest_loader", "get_unique_labels"),
    "filter_records_by_split": ("manifest_loader", "filter_records_by_split"),
    "filter_records_by_label": ("manifest_loader", "filter_records_by_label"),
    "convert_manifest": ("manifest_converter", "convert_manifest"),
    "convert_mhist_manifest": ("manifest_converter", "convert_mhist_manifest"),
    "sample_manifest": ("manifest_sampler", "sample_manifest"),
    "summarize_manifest": ("manifest_sampler", "summarize_manifest"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = __import__(
        f"{__name__}.{module_name}",
        fromlist=[attribute_name],
    )
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
