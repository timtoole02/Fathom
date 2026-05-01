#!/usr/bin/env bash
set -euo pipefail

# Fathom has no built-in authentication; keep FATHOM_BASE_URL on loopback unless you have added your own access controls.
BASE="${FATHOM_BASE_URL:-http://127.0.0.1:8180}"
BASE="${BASE%/}"
MODEL_ID="${FATHOM_MODEL_ID:-echarlaix-tiny-random-phiforcausallm-model-safetensors}"
REPO_ID="${FATHOM_REPO_ID:-echarlaix/tiny-random-PhiForCausalLM}"
FILENAME="${FATHOM_FILENAME:-model.safetensors}"
PROMPT="${FATHOM_PROMPT:-Say hello from a local Fathom API smoke test.}"
MAX_TOKENS="${FATHOM_MAX_TOKENS:-24}"
RUN_EMBEDDINGS="${FATHOM_RUN_EMBEDDINGS:-0}"
EMBEDDING_MODEL_ID="${FATHOM_EMBEDDING_MODEL_ID:-sentence-transformers-all-minilm-l6-v2-model-safetensors}"
EMBEDDING_REPO_ID="${FATHOM_EMBEDDING_REPO_ID:-sentence-transformers/all-MiniLM-L6-v2}"
EMBEDDING_FILENAME="${FATHOM_EMBEDDING_FILENAME:-model.safetensors}"
EMBEDDING_INPUT="${FATHOM_EMBEDDING_INPUT:-Rust ownership keeps memory safety explicit.}"

json_pretty() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -m json.tool
  else
    cat
  fi
}

post_json() {
  local path="$1"
  local body="$2"
  curl -fsS "$BASE$path" \
    -H "Content-Type: application/json" \
    -d "$body"
}

echo "== Fathom health =="
curl -fsS "$BASE/v1/health" | json_pretty

echo
echo "== Install pinned tiny Phi SafeTensors fixture, if not already present =="
install_body="$(REPO_ID="$REPO_ID" FILENAME="$FILENAME" python3 - <<'PY'
import json
import os
print(json.dumps({"repo_id": os.environ["REPO_ID"], "filename": os.environ["FILENAME"]}))
PY
)"
post_json "/api/models/catalog/install" "$install_body" | json_pretty

echo
echo "== Runnable chat models =="
curl -fsS "$BASE/v1/models" | json_pretty

echo
echo "== Non-streaming chat completion =="
chat_body="$(MODEL_ID="$MODEL_ID" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 - <<'PY'
import json
import os
print(json.dumps({
    "model": os.environ["MODEL_ID"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}))
PY
)"
post_json "/v1/chat/completions" "$chat_body" | json_pretty

if [[ "$RUN_EMBEDDINGS" == "1" ]]; then
  echo
  echo "== Install pinned MiniLM SafeTensors embedding fixture, if not already present =="
  embedding_install_body="$(REPO_ID="$EMBEDDING_REPO_ID" FILENAME="$EMBEDDING_FILENAME" python3 - <<'PY'
import json
import os
print(json.dumps({"repo_id": os.environ["REPO_ID"], "filename": os.environ["FILENAME"]}))
PY
)"
  post_json "/api/models/catalog/install" "$embedding_install_body" | json_pretty

  echo
  echo "== Float embeddings from verified local MiniLM runtime =="
  embedding_body="$(MODEL_ID="$EMBEDDING_MODEL_ID" INPUT="$EMBEDDING_INPUT" python3 - <<'PY'
import json
import os
print(json.dumps({
    "model": os.environ["MODEL_ID"],
    "input": os.environ["INPUT"],
    "encoding_format": "float",
}))
PY
)"
  post_json "/v1/embeddings" "$embedding_body" | json_pretty
fi
