# CONCH Feasibility Check

This document defines a local-only feasibility check for possible CONCH support in a future v0.4.0 milestone.

The goal is to verify access, license constraints, dependency compatibility, model loading, embedding APIs, and laptop GPU behavior before implementing `CONCHWrapper`.

Do not add these checks to CI. They may require Hugging Face authentication, gated model access, optional package installation, model weight downloads, and local CUDA hardware.

## Current Status

CONCH is currently registered as a placeholder:

```text
key: conch
model_name: MahmoodLab/CONCH
implemented: False
```

Do not mark `conch` as implemented until the local checks below pass and the integration path is clear.

## Public Access Notes

As of May 6, 2026:

- `MahmoodLab/CONCH` is a gated Hugging Face model.
- Access may require accepting usage terms and using an approved account.
- The model card lists the license as CC-BY-NC-ND-4.0.
- The official usage path points to the `mahmoodlab/CONCH` repository and `create_model_from_pretrained`, not a simple `transformers.AutoModel` path.

Primary references:

- Hugging Face model card: <https://huggingface.co/MahmoodLab/CONCH>
- CONCH GitHub repository: <https://github.com/mahmoodlab/CONCH>

Record any changes to access, license, package, or loading behavior before implementing a wrapper.

## Local Result Summary

Local feasibility checks were run on May 6, 2026. The checks installed the official CONCH package, verified gated Hugging Face access, loaded the model on CPU, produced text and image embeddings, and ran a CUDA image smoke test.

- OS: Windows
- Python: `3.13.2`
- torch: `2.11.0+cu126`
- CUDA available: `True`
- CUDA runtime: `12.6`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- `huggingface_hub`: `1.12.0`
- Hugging Face token present in this environment: `True`
- `huggingface_hub.HfFolder`: unavailable in this installed version
- `huggingface_hub.get_token()`: available and returned a token
- `huggingface_hub.model_info("MahmoodLab/CONCH")`: succeeded without downloading weights
- `huggingface_hub.hf_hub_download("MahmoodLab/CONCH", "meta.yaml")`: succeeded
- model id: `MahmoodLab/CONCH`
- `MahmoodLab/conch` also resolves to `MahmoodLab/CONCH`
- private: `False`
- gated: `auto`
- revision: `f9ca9f877171a28ade80228fb195ac5d79003357`
- license: `cc-by-nc-nd-4.0`
- CONCH install command: `python -m pip install git+https://github.com/Mahmoodlab/CONCH.git`
- CONCH repository commit installed by pip: `141cc09c7d4ff33d8eda562bd75169b457f71a62`
- installed package: `conch==0.1.0`
- installed additional dependencies: `timm==1.0.26`, `ftfy==6.3.1`, `h5py==3.16.0`
- `conch` package importable: `True`
- `conch.open_clip_custom.create_model_from_pretrained` importable: `True`
- `conch.open_clip_custom.tokenize` importable: `True`
- import warning observed: `timm.models.layers` import path is deprecated
- CPU model load command attempted with `checkpoint_path="hf_hub:MahmoodLab/CONCH"`
- CPU model load result: passed
- model class: `conch.open_clip_custom.coca_model.CoCa`
- preprocess: resize and center crop to `448 x 448`, then CLIP-style normalization
- text embedding smoke test: passed, `torch.Size([2, 512])`, `torch.float32`, normalized norms `1.0000`
- image embedding smoke test: passed, `torch.Size([2, 512])`, `torch.float32`, normalized norms `1.0000`
- CPU similarity smoke test: passed, output matrix shape `2 x 2`, finite values
- CUDA image smoke test: passed, `torch.Size([1, 512])`, `torch.float32`, device `cuda:0`, normalized norm `1.0000`
- peak CUDA memory for batch size 1 image smoke test: about `1549 MB`
- tokenizer compatibility note: `conch.open_clip_custom.custom_tokenizer.tokenize` expects `batch_encode_plus`, but the installed `transformers==5.6.2` tokenizer backend does not expose that method. Calling the tokenizer directly and padding `input_ids` to length 128 works.

Current conclusion: CONCH is feasible for a local optional wrapper. The likely wrapper should depend on optional CONCH installation, use `create_model_from_pretrained`, use `get_tokenizer()`, avoid the package helper `tokenize()` under current `transformers`, and keep CI fully offline with mocked objects. Keep `conch` as a placeholder until the wrapper and offline tests are implemented.

## Questions to Answer

Before implementing CONCH, confirm:

- Is `MahmoodLab/CONCH` still the intended model identifier?
- Does the local Hugging Face account have access to the gated model?
- Are the model license and terms compatible with the intended research-only use?
- What package installation path is required?
- Can CONCH load on CPU?
- Can CONCH load on CUDA on the local laptop GPU?
- Does the model expose stable text and image embedding APIs?
- Do text and image embeddings have the same dimensionality?
- Can embeddings be L2-normalized like CLIP and PLIP embeddings?
- Can the wrapper return CPU tensors from both `encode_text` and `encode_images`?
- Can failures for missing authentication, weights, or dependencies be made clear?

## Environment Check

Run this first:

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Record:

- OS
- Python version
- PyTorch version
- CUDA availability
- CUDA runtime version
- GPU name

## Hugging Face Authentication Check

Check whether `huggingface_hub` is installed and whether a token is visible to the local environment:

```bash
python -c "from huggingface_hub import get_token; token=get_token(); print('token_present=', token is not None)"
```

If no token is present, authenticate through the standard Hugging Face CLI before access checks:

```bash
huggingface-cli login
```

Do not commit tokens, local cache paths, or credential files.

## Metadata and Access Check

This checks whether Hugging Face can resolve the model metadata without loading weights:

```bash
python -c "from huggingface_hub import model_info; info = model_info('MahmoodLab/CONCH'); print(info.modelId); print(info.private); print(info.gated); print(info.sha); print(info.cardData.get('license') if info.cardData else None)"
```

Record:

- whether the request succeeds
- whether the model is gated
- whether access is approved
- model revision / SHA
- license string
- exact error message if access is denied

If this step fails because access is not approved, stop. Keep `conch` as a placeholder and document the blocker.

## Dependency Check

CONCH may require installing the official repository as an optional local dependency.

Check whether it is already importable:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('conch') is not None)"
```

If it is not installed, use the official repository instructions in a local virtual environment. A likely installation form is:

```bash
pip install git+https://github.com/mahmoodlab/CONCH.git
```

Record:

- exact install command
- package version or commit
- whether installation modifies major dependencies such as `torch`, `torchvision`, `timm`, or `open_clip_torch`
- any dependency conflicts with PathVLM-LiteBench

Do not add CONCH as a core dependency unless the integration is stable and low-risk.

## Import Check

After installation, verify the expected import path:

```bash
python -c "from conch.open_clip_custom import create_model_from_pretrained, tokenize; print(create_model_from_pretrained); print(tokenize)"
```

Record:

- whether import succeeds
- exact module path
- any import-time warnings
- missing dependencies

If the import path differs from the model card, record the working path before continuing.

## CPU Model Load Check

Run a tiny CPU load check using the official loading path:

```bash
python -c "from conch.open_clip_custom import create_model_from_pretrained; model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path='hf_hub:MahmoodLab/CONCH'); model.eval(); print(type(model)); print(preprocess)"
```

Record:

- whether weights download successfully
- model class
- preprocessing object
- cache location if relevant
- exact warnings or errors

If the checkpoint name differs, record the exact model name that works.

## Text Embedding Smoke Test

Run:

```bash
python -c "import torch; import torch.nn.functional as F; from conch.open_clip_custom import create_model_from_pretrained, tokenize; model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path='hf_hub:MahmoodLab/CONCH'); model.eval(); texts=['a histopathology image of hyperplastic polyp','a histopathology image of sessile serrated adenoma']; tokens=tokenize(texts); with torch.no_grad(): feats=model.encode_text(tokens); feats=F.normalize(feats, p=2, dim=-1); print(feats.shape); print(feats.dtype); print(torch.linalg.norm(feats, dim=-1))"
```

Expected:

- shape should be `[2, embedding_dim]`
- dtype should usually be `torch.float32` or a compatible floating dtype
- normalized vector norms should be close to `1.0`

Record whether `encode_text` exists and whether tokenization accepts a list of strings.

## Image Embedding Smoke Test

Run:

```bash
python -c "import torch; import torch.nn.functional as F; from PIL import Image; from conch.open_clip_custom import create_model_from_pretrained; model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path='hf_hub:MahmoodLab/CONCH'); model.eval(); images=[preprocess(Image.new('RGB',(224,224),'white')), preprocess(Image.new('RGB',(224,224),'purple'))]; batch=torch.stack(images, dim=0); with torch.no_grad(): feats=model.encode_image(batch); feats=F.normalize(feats, p=2, dim=-1); print(feats.shape); print(feats.dtype); print(torch.linalg.norm(feats, dim=-1))"
```

Expected:

- shape should be `[2, embedding_dim]`
- image embedding dimension should match text embedding dimension
- normalized vector norms should be close to `1.0`

Record whether `encode_image` exists and whether the official preprocess output can be batched.

## CPU Similarity Smoke Test

After text and image embeddings pass, verify the shared embedding space:

```bash
python -c "import torch; import torch.nn.functional as F; from PIL import Image; from conch.open_clip_custom import create_model_from_pretrained, tokenize; model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path='hf_hub:MahmoodLab/CONCH'); model.eval(); texts=['a white histology patch','a purple histology patch']; tokens=tokenize(texts); images=torch.stack([preprocess(Image.new('RGB',(224,224),'white')), preprocess(Image.new('RGB',(224,224),'purple'))]); with torch.no_grad(): text_feats=F.normalize(model.encode_text(tokens), p=2, dim=-1); image_feats=F.normalize(model.encode_image(images), p=2, dim=-1); print(image_feats @ text_feats.T)"
```

Record:

- output matrix shape
- whether values are finite
- whether dimensions align without projection errors

Do not interpret synthetic image scores as pathology performance.

## CUDA Smoke Test

Only run after CPU checks pass:

```bash
python -c "import torch; import torch.nn.functional as F; from PIL import Image; from conch.open_clip_custom import create_model_from_pretrained; device='cuda'; model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path='hf_hub:MahmoodLab/CONCH'); model=model.to(device); model.eval(); image=preprocess(Image.new('RGB',(224,224),'white')).unsqueeze(0).to(device); torch.cuda.reset_peak_memory_stats(); with torch.no_grad(): feats=model.encode_image(image); feats=F.normalize(feats, p=2, dim=-1); print(feats.shape); print(feats.device); print(torch.cuda.max_memory_allocated() / 1024**2)"
```

Record:

- whether CUDA load succeeds
- peak allocated memory in MB
- whether batch size 1 is stable
- whether CUDA out-of-memory occurs

Do not benchmark large batches at this stage.

## Wrapper Feasibility Criteria

CONCH is feasible for implementation if:

- model access is approved locally
- license and terms are compatible with research-only use
- dependency installation is stable and does not break existing tests
- CPU text embeddings can be produced
- CPU image embeddings can be produced
- CUDA image embeddings can be produced on a small batch when hardware is available
- text and image embedding dimensions match
- embeddings can be normalized
- errors for missing authentication or dependencies can be made clear
- CI can remain offline and lightweight

## Failure Criteria

Do not implement `CONCHWrapper` yet if:

- model access is unavailable
- license or usage constraints are incompatible or unclear
- package installation conflicts with core dependencies
- the loading path requires fragile local hacks
- the model lacks stable image/text embedding APIs
- embeddings cannot be aligned through the current wrapper interface
- CUDA memory use is too high even for tiny batches
- the wrapper would require downloads in CI

If a check fails, record the exact command, error message, package versions, and whether the failure happened on CPU or CUDA.

## Result Log Template

Result log format:

```text
Date:
OS:
Python:
torch:
torchvision:
CUDA available:
GPU:
CONCH repo/package:
CONCH package commit/version:
model_id:
checkpoint path:

Hugging Face metadata/access:
License:
Gated:
Access approved:

Dependency install:
Import path:
Model class:
Preprocess:
Tokenizer:

CPU model load:
CPU text smoke:
CPU image smoke:
CPU similarity smoke:
CUDA image smoke:
Peak CUDA memory MB:

Conclusion:
Blockers:
Recommended action:
```

## Implementation Follow-up

If all checks pass, the implementation follow-up is:

1. Add optional CONCH dependency documentation.
2. Add `CONCHWrapper` with the same public interface as `CLIPWrapper`.
3. Keep `conch` registered as implemented only after local smoke tests pass.
4. Add tests that use fake objects or monkeypatching and do not download weights.
5. Update README model registry status.
6. Update `docs/project_positioning.md`.
7. Run optional local CLIP vs PLIP vs CONCH comparisons on saved manifests.

If checks fail or access is unavailable, keep `conch` as a documented placeholder and record the blocker here.
