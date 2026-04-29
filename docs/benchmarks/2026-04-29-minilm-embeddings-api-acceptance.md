# MiniLM embeddings optional API acceptance evidence

This is optional local API evidence for the pinned catalog entry `hf-sentence-transformers-all-minilm-l6-v2-safetensors`. It is not default CI, not a first-run requirement, not an embedding quality/retrieval quality/performance claim, and not broad Hugging Face, ONNX chat, or arbitrary model support.

## Scope

- Model repo: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
- License metadata: `apache-2.0`
- Fathom lane: existing default-build Candle `BertModel` / MiniLM SafeTensors embeddings lane
- Catalog id: `hf-sentence-transformers-all-minilm-l6-v2-safetensors`
- Installed embedding model id: `sentence-transformers-all-minilm-l6-v2-model-safetensors`
- Public contract: unchanged; `/v1/embeddings` remains float-only for verified local embedding runtimes, and embedding-only models remain excluded from `/v1/models`.
- Evidence commit: `c04a16fe690f3716ad1994f9c90cc0c3df396582`

## Verified catalog files

The optional demo was installed through the catalog path and verified against these expected sizes and SHA256 hashes:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 612 | `953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41` |
| `model.safetensors` | 90,868,376 | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| `modules.json` | 349 | `84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf` |
| `special_tokens_map.json` | 112 | `303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3` |
| `tokenizer.json` | 466,247 | `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037` |
| `tokenizer_config.json` | 350 | `acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b` |
| `1_Pooling/config.json` | 190 | `4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23` |

Total expected catalog bytes: `91,336,236`.

## Optional API acceptance smoke

The opt-in harness was run locally with preserved artifacts:

```bash
FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE=1 \
FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_KEEP_ARTIFACTS=1 \
bash scripts/minilm_embeddings_optional_api_acceptance_smoke.sh
```

Artifact QA was then run against the preserved artifact directory:

```bash
python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py /path/to/artifacts
```

Result:

- Harness result: `passed`
- Artifact QA result: `passed`
- Installed embedding model id: `sentence-transformers-all-minilm-l6-v2-model-safetensors`
- `/api/embedding-models` included the installed MiniLM embedding model after catalog install/verification.
- `/v1/models` excluded the embedding-only model from chat/generation listings.
- `/v1/embeddings` returned two finite 384-dimensional float vectors.
- Fathom embedding metadata included `runtime: candle-bert-embeddings`, `embedding_dimension: 384`, `scope: verified local embedding runtime only`, and timing metrics.
- `encoding_format: "base64"` was refused with `400 invalid_request` and no fake embedding data.
- `/v1/chat/completions` against the embedding model was refused with no fake `choices`.

## Boundaries preserved

- MiniLM remains an embedding-only demo, not a chat/generation model.
- This proves only the pinned package through an existing verified SafeTensors/HF embedding code path.
- `/v1/models` remains chat/generation-only and excludes embedding-only models.
- `/v1/embeddings` remains float-only; base64 embeddings remain unsupported.
- ONNX chat/general execution, external providers, PyTorch `.bin`, unsupported templates, arbitrary SafeTensors/Hugging Face execution, streaming, and full OpenAI API parity remain refused or unclaimed.
- GGUF remains metadata-only; no public/runtime GGUF tokenizer execution, weight loading, generation, or inference is claimed here.
