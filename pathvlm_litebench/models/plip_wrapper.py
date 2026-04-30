from .clip_wrapper import CLIPWrapper


class PLIPWrapper(CLIPWrapper):
    """
    Wrapper for the PLIP pathology vision-language model.

    The current PLIP Hugging Face checkpoint exposes a CLIP-compatible
    processor/model interface, so this wrapper reuses CLIPWrapper behavior
    while fixing the default model name to vinid/plip.
    """

    def __init__(self, model_name: str = "vinid/plip", device: str | None = None):
        super().__init__(model_name=model_name, device=device)
