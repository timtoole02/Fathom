# Optional SmolLM2 API acceptance flow

This is an opt-in local acceptance flow for the pinned `hf-huggingfacetb-smollm2-135m-instruct` catalog demo. It exercises the full Fathom product path for one larger SmolLM2/Llama-style SafeTensors/HF model while keeping the public `/v1` contract unchanged.

Do not add this flow to default CI. It downloads or reuses about 271 MB of model files and can use significant RAM during local generation. Treat the output as larger-demo evidence only, not as a quality, latency, throughput, production-readiness, legal, broad SmolLM2/Llama-family, arbitrary Hugging Face, or full OpenAI parity claim.

## What this flow should prove

- The catalog entry is still pinned to `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Catalog installation verifies expected file sizes, SHA256 hashes, Apache-2.0 metadata, and manifest fields before the model becomes runnable.
- After install/inspection, `/v1/models` includes the SmolLM2 chat model as a validated local `safetensors-hf`/Candle model.
- Two same-process non-streaming `/v1/chat/completions` calls return real local assistant content and expose `fathom.metrics.runtime_family: llama` plus cold/warm residency evidence.
- `stream: true` remains refused with `501 not_implemented`.

## One-command optional smoke

```bash
FATHOM_SMOLLM2_ACCEPTANCE=1 \
FATHOM_SMOLLM2_ACCEPTANCE_KEEP_ARTIFACTS=1 \
bash scripts/smollm2_optional_api_acceptance_smoke.sh
```

The script starts an isolated server, installs only the pinned SmolLM2 catalog entry, writes health/install/`/v1/models`/cold-chat/warm-chat/stream-refusal artifacts plus `summary.json` and `summary.md`, then runs offline artifact QA. It is intentionally not part of default CI.

Validate preserved artifacts later with:

```bash
python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py /path/to/artifacts
```

With no arguments, the artifact QA runs a dependency-free synthetic self-test.

## Isolated directories

The harness uses isolated directories by default and supports overrides:

- `FATHOM_SMOLLM2_ACCEPTANCE_ROOT`
- `FATHOM_SMOLLM2_ACCEPTANCE_MODELS_DIR`
- `FATHOM_SMOLLM2_ACCEPTANCE_STATE_DIR`
- `FATHOM_SMOLLM2_ACCEPTANCE_LOG_DIR`
- `FATHOM_SMOLLM2_ACCEPTANCE_ARTIFACT_DIR`
- `FATHOM_SMOLLM2_ACCEPTANCE_PORT`

Set `FATHOM_SMOLLM2_ACCEPTANCE_KEEP_ARTIFACTS=1` when you want to review or preserve generated evidence.

## Share-safe artifact review

Before copying generated artifacts into checked-in docs or public issue comments:

- Replace local paths with generic placeholders.
- Remove prompts or completions that should not be public.
- Keep only summarized fields needed to prove install, `/v1/models`, chat metrics, and refusal behavior.
- Run `bash scripts/public_risk_scan.sh` after adding any sanitized evidence.

## Boundaries preserved

This flow does not claim public/runtime GGUF tokenizer execution, GGUF inference, ONNX chat, PyTorch `.bin`, external provider proxying, arbitrary SafeTensors/HF execution, streaming, or full OpenAI API parity. GGUF remains metadata-only and excluded from `/v1/models`.
