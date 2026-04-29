# Optional Qwen2.5 API acceptance flow

This is an opt-in local acceptance recipe for the pinned `hf-qwen-qwen2-5-0-5b-instruct` catalog demo. It exercises the full Fathom product path for one larger Qwen2/ChatML SafeTensors/HF model while keeping the public `/v1` contract unchanged.

Do not add this flow to default CI. It downloads or reuses about 1.0 GB of model files and the existing local smoke observed roughly 3.29 GB maximum resident set size. Treat the output as larger-demo evidence only, not as a quality, latency, throughput, production-readiness, legal, broad Qwen, arbitrary Hugging Face, or full OpenAI parity claim.

## What this flow should prove

- The catalog entry is still pinned to `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Catalog installation verifies expected file sizes, SHA256 hashes, Apache-2.0 metadata, and manifest fields before the model becomes runnable.
- After install/inspection, `/v1/models` includes the Qwen2.5 chat model as a validated local `safetensors-hf`/Candle model.
- Two same-process non-streaming `/v1/chat/completions` calls return real local assistant content and expose `fathom.metrics.runtime_family: qwen2` plus cold/warm residency evidence.
- `stream: true` remains refused with `501 not_implemented`.

## Isolated run skeleton

Use isolated state/model directories so the acceptance pass cannot mutate your normal Fathom install:

```bash
export FATHOM_QWEN25_ACCEPTANCE_ROOT="$(mktemp -d /tmp/fathom-qwen25-api.XXXXXX)"
export FATHOM_MODELS_DIR="$FATHOM_QWEN25_ACCEPTANCE_ROOT/models"
export FATHOM_STATE_DIR="$FATHOM_QWEN25_ACCEPTANCE_ROOT/state"
export FATHOM_LOG_DIR="$FATHOM_QWEN25_ACCEPTANCE_ROOT/logs"
export FATHOM_PORT="18185"
mkdir -p "$FATHOM_MODELS_DIR" "$FATHOM_STATE_DIR" "$FATHOM_LOG_DIR"

cargo run -q -p fathom-server >"$FATHOM_LOG_DIR/server.log" 2>&1 &
export FATHOM_SERVER_PID="$!"
```

Wait for health:

```bash
BASE="http://127.0.0.1:${FATHOM_PORT}"
for _ in $(seq 1 180); do
  curl -fsS "$BASE/v1/health" >/dev/null && break
  sleep 1
done
curl -fsS "$BASE/v1/health" | python3 -m json.tool
```

Install the pinned demo:

```bash
curl -fsS "$BASE/api/models/catalog/install" \
  -H 'content-type: application/json' \
  -d '{"repo_id":"Qwen/Qwen2.5-0.5B-Instruct","filename":"model.safetensors"}' \
  | tee "$FATHOM_QWEN25_ACCEPTANCE_ROOT/install.json" \
  | python3 -m json.tool
```

Check `/v1/models`:

```bash
curl -fsS "$BASE/v1/models" \
  | tee "$FATHOM_QWEN25_ACCEPTANCE_ROOT/v1-models.json" \
  | python3 -m json.tool
```

Run two non-streaming chat calls in the same server process:

```bash
for phase in cold warm; do
  curl -fsS "$BASE/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"hf-qwen-qwen2-5-0-5b-instruct","messages":[{"role":"user","content":"Reply with one short sentence about local inference."}],"max_tokens":24,"temperature":0.2}' \
    | tee "$FATHOM_QWEN25_ACCEPTANCE_ROOT/chat-${phase}.json" \
    | python3 -m json.tool
done
```

Confirm streaming remains refused:

```bash
curl -sS -o "$FATHOM_QWEN25_ACCEPTANCE_ROOT/stream-refusal.json" \
  -w '%{http_code}\n' \
  "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"hf-qwen-qwen2-5-0-5b-instruct","messages":[{"role":"user","content":"Hello"}],"stream":true}'
python3 -m json.tool "$FATHOM_QWEN25_ACCEPTANCE_ROOT/stream-refusal.json"
```

Stop the isolated server when done:

```bash
kill "$FATHOM_SERVER_PID"
```

## Share-safe artifact review

Before copying any generated artifact into checked-in docs or public issue comments:

- Replace local paths with generic placeholders.
- Remove prompts or completions that should not be public.
- Keep only summarized fields needed to prove install, `/v1/models`, chat metrics, and refusal behavior.
- Run `bash scripts/public_risk_scan.sh` after adding any sanitized evidence.

## Boundaries preserved

This flow does not claim public/runtime GGUF tokenizer execution, GGUF inference, ONNX chat, PyTorch `.bin`, external provider proxying, arbitrary SafeTensors/HF execution, streaming, or full OpenAI API parity. GGUF remains metadata-only and excluded from `/v1/models`.
