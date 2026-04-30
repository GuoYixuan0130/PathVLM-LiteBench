# PLIP Feasibility Check

This document defines a local-only feasibility check for adding PLIP support in a future v0.2.0 milestone.

The goal is to verify model access, dependency compatibility, embedding APIs, and laptop GPU behavior before implementing `PLIPWrapper`.

Do not add these checks to CI. They may download model weights and depend on local hardware, network access, and Hugging Face availability.

## Local Result Summary

Local feasibility checks passed on the development laptop:

- model id: `vinid/plip`
- private: `False`
- revision: `67ade53ddd32195868f422585f72698ef5d15094`
- processor class: `CLIPProcessor`
- model class: `CLIPModel`
- `get_text_features`: available
- `get_image_features`: available
- CPU text embeddings: `torch.Size([2, 512])`
- CPU image embeddings: `torch.Size([2, 512])`
- CUDA image embeddings: `torch.Size([1, 512])`
- CUDA device: `cuda:0`
- peak CUDA memory for batch size 1 image smoke test: about `589 MB`

The model returns `BaseModelOutputWithPooling` from `get_text_features` and `get_image_features` in this environment. The wrapper must use `.pooler_output` when the returned object is not already a tensor. This matches the compatibility logic already used by `CLIPWrapper`.

## Questions to Answer

Before implementing PLIP, confirm:

- Is the intended Hugging Face model identifier correct?
- Does the model require authentication?
- Can the model and processor load with the current `transformers` version?
- Does the model expose CLIP-style text and image embedding APIs?
- Are returned embeddings shaped as expected?
- Can embeddings be L2-normalized in the same way as `CLIPWrapper`?
- Can the model run on CPU for a tiny smoke test?
- Can the model run on CUDA on an RTX 4060 Laptop GPU?
- Is memory usage acceptable for small patch batches?

## Environment Check

Run this first:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Expected local CUDA environment example:

```text
2.11.0+cu126
True
12.6
NVIDIA GeForce RTX 4060 Laptop GPU
```

## Candidate Model Identifier

Initial candidate:

```text
vinid/plip
```

If this model identifier fails, record the exact error before trying alternatives.

## Metadata and Access Check

This checks whether Hugging Face can resolve the model without loading weights:

```bash
python -c "from huggingface_hub import model_info; info = model_info('vinid/plip'); print(info.modelId); print(info.private); print(info.sha)"
```

Record:

- whether the request succeeds
- whether authentication is required
- model revision / SHA
- license information if available

## Processor and Model Load Check

Run a tiny load check on CPU first:

```bash
python -c "from transformers import AutoModel, AutoProcessor; model_id='vinid/plip'; processor=AutoProcessor.from_pretrained(model_id); model=AutoModel.from_pretrained(model_id); print(type(processor)); print(type(model)); print(hasattr(model, 'get_text_features')); print(hasattr(model, 'get_image_features'))"
```

Record:

- processor class
- model class
- whether `get_text_features` exists
- whether `get_image_features` exists
- any warnings or errors

If `AutoProcessor` or `AutoModel` fails, try CLIP-specific classes only as a diagnostic:

```bash
python -c "from transformers import CLIPModel, CLIPProcessor; model_id='vinid/plip'; processor=CLIPProcessor.from_pretrained(model_id); model=CLIPModel.from_pretrained(model_id); print(type(processor)); print(type(model))"
```

Do not implement a wrapper until the correct loading path is clear.

## Text Embedding Smoke Test

Run:

```bash
python -c "import torch; import torch.nn.functional as F; from transformers import AutoModel, AutoProcessor; model_id='vinid/plip'; processor=AutoProcessor.from_pretrained(model_id); model=AutoModel.from_pretrained(model_id); model.eval(); inputs=processor(text=['a histopathology image of hyperplastic polyp','a histopathology image of sessile serrated adenoma'], return_tensors='pt', padding=True, truncation=True); with torch.no_grad(): feats=model.get_text_features(**inputs); feats=F.normalize(feats, p=2, dim=-1); print(feats.shape); print(feats.dtype); print(torch.linalg.norm(feats, dim=-1))"
```

Expected:

- shape should be `[2, embedding_dim]`
- dtype should usually be `torch.float32`
- normalized vector norms should be close to `1.0`

## Image Embedding Smoke Test

Run:

```bash
python -c "import torch; import torch.nn.functional as F; from PIL import Image; from transformers import AutoModel, AutoProcessor; model_id='vinid/plip'; processor=AutoProcessor.from_pretrained(model_id); model=AutoModel.from_pretrained(model_id); model.eval(); images=[Image.new('RGB',(224,224),'white'), Image.new('RGB',(224,224),'purple')]; inputs=processor(images=images, return_tensors='pt', padding=True); with torch.no_grad(): feats=model.get_image_features(**inputs); feats=F.normalize(feats, p=2, dim=-1); print(feats.shape); print(feats.dtype); print(torch.linalg.norm(feats, dim=-1))"
```

Expected:

- shape should be `[2, embedding_dim]`
- image embedding dimension should match text embedding dimension
- normalized vector norms should be close to `1.0`

## CUDA Smoke Test

Only run this after CPU text and image smoke tests pass:

```bash
python -c "import torch; import torch.nn.functional as F; from PIL import Image; from transformers import AutoModel, AutoProcessor; model_id='vinid/plip'; device='cuda'; processor=AutoProcessor.from_pretrained(model_id); model=AutoModel.from_pretrained(model_id).to(device); model.eval(); images=[Image.new('RGB',(224,224),'white')]; inputs=processor(images=images, return_tensors='pt', padding=True).to(device); torch.cuda.reset_peak_memory_stats(); with torch.no_grad(): feats=model.get_image_features(**inputs); feats=F.normalize(feats, p=2, dim=-1); print(feats.shape); print(feats.device); print(torch.cuda.max_memory_allocated() / 1024**2)"
```

Record:

- whether CUDA load succeeds
- peak allocated memory in MB
- whether batch size 1 is stable
- whether CUDA out-of-memory occurs

Do not benchmark large batches at this stage.

## Success Criteria

PLIP is feasible for v0.2.0 implementation if:

- model and processor load without authentication blockers
- text embeddings can be produced
- image embeddings can be produced
- text and image embedding dimensions match
- embeddings can be normalized
- CPU smoke test works
- CUDA smoke test works on small batches
- memory use is acceptable for laptop-scale patch evaluation
- license and usage terms are compatible with research use

## Failure Criteria

Do not implement `PLIPWrapper` yet if:

- model access requires unavailable credentials
- loading requires unsupported dependencies
- the model lacks CLIP-style image/text embedding APIs
- embeddings cannot be aligned through the current wrapper interface
- CUDA memory use is too high even for tiny batches
- license or usage constraints are unclear

If a check fails, record the exact command, error message, package versions, and whether the failure happened on CPU or CUDA.

## Result Log Template

Use this template when reporting local results:

```text
Date:
OS:
Python:
torch:
transformers:
CUDA available:
GPU:
model_id:

Metadata/access:
Processor class:
Model class:
get_text_features:
get_image_features:

CPU text smoke:
CPU image smoke:
CUDA image smoke:
Peak CUDA memory MB:

Conclusion:
Blockers:
Next action:
```

## Next Step After Success

If all checks pass, the next implementation step is:

1. Add `PLIPWrapper` with the same interface as `CLIPWrapper`.
2. Register `plip` as implemented only after local smoke tests pass.
3. Add unit tests that do not download PLIP weights.
4. Document optional local PLIP smoke commands.
5. Run CLIP vs PLIP comparison on a sampled MHIST manifest.
