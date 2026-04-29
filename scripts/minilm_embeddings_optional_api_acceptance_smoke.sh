#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE:-}" != "1" ]]; then
  cat >&2 <<'EOF'
MiniLM embeddings optional API acceptance is opt-in because it downloads or reuses about 91 MB of model files.
Set FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE=1 to run it.
EOF
  exit 2
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 127; }; }
need cargo; need curl; need python3

PORT="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_PORT:-18187}"
BASE="http://127.0.0.1:${PORT}"
WAIT_SECONDS="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_WAIT_SECONDS:-240}"
REQUEST_TIMEOUT="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_REQUEST_TIMEOUT:-900}"
KEEP_ARTIFACTS="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_KEEP_ARTIFACTS:-0}"
TMP_ROOT="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ROOT:-${TMPDIR:-/tmp}/fathom-minilm-embeddings-api-$$}"
MODELS_DIR="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_MODELS_DIR:-$TMP_ROOT/models}"
STATE_DIR="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_STATE_DIR:-$TMP_ROOT/state}"
ARTIFACT_DIR="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ARTIFACT_DIR:-$TMP_ROOT/artifacts}"
LOG_DIR="${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_LOG_DIR:-$TMP_ROOT/logs}"
REPO_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null || echo unknown)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ $status -eq 0 && "$KEEP_ARTIFACTS" != "1" && -z "${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ROOT:-}" && -z "${FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_ARTIFACT_DIR:-}" ]]; then
    rm -rf "$TMP_ROOT"
  else
    echo "MiniLM embeddings optional acceptance artifacts: $ARTIFACT_DIR" >&2
    if [[ $status -ne 0 && -f "$LOG_DIR/server.log" ]]; then
      echo "--- server.log tail ---" >&2
      tail -120 "$LOG_DIR/server.log" >&2 || true
    fi
  fi
}
trap cleanup EXIT

if curl -fsS "$BASE/v1/health" >/dev/null 2>&1; then
  echo "Port $PORT already has a responding Fathom backend. Choose another FATHOM_MINILM_EMBEDDINGS_ACCEPTANCE_PORT." >&2
  exit 1
fi

mkdir -p "$MODELS_DIR" "$STATE_DIR" "$ARTIFACT_DIR" "$LOG_DIR"

echo "== start isolated Fathom server for optional MiniLM embeddings API acceptance =="
FATHOM_PORT="$PORT" \
FATHOM_STATE_DIR="$STATE_DIR" \
FATHOM_MODELS_DIR="$MODELS_DIR" \
FATHOM_LOG_DIR="$LOG_DIR" \
cargo run -q -p fathom-server >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 "$WAIT_SECONDS"); do
  if curl -fsS "$BASE/v1/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "fathom-server exited before /v1/health became ready" >&2; exit 1; fi
  sleep 1
done
curl -fsS "$BASE/v1/health" >/dev/null

python3 - "$BASE" "$ARTIFACT_DIR" "$REQUEST_TIMEOUT" "$REPO_COMMIT" "$STARTED_AT" <<'PY'
import json, math, pathlib, sys, time, urllib.error, urllib.request
base=sys.argv[1].rstrip('/'); artifacts=pathlib.Path(sys.argv[2]); timeout=float(sys.argv[3]); repo_commit=sys.argv[4]; started_at=sys.argv[5]
artifacts.mkdir(parents=True, exist_ok=True)
MODEL_ID='sentence-transformers-all-minilm-l6-v2-model-safetensors'
REPO_ID='sentence-transformers/all-MiniLM-L6-v2'
REVISION='c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
checks=[]

def request(method,path,body=None):
    data=None; headers={'Accept':'application/json'}
    if body is not None:
        data=json.dumps(body).encode(); headers['Content-Type']='application/json'
    req=urllib.request.Request(base+path,data=data,method=method,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read().decode(); return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw=e.read().decode(); return e.code, json.loads(raw) if raw else None

def save(name,status,payload,check,description,expected):
    (artifacts/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    checks.append({'name':check,'artifact':name,'description':description,'expected_http_status':expected,'http_status':status,'status':'passed' if status==expected else 'failed'})

status,health=request('GET','/v1/health'); save('01-v1-health.json',status,health,'health','Backend health responds before optional MiniLM install.',200); assert status==200
status,install=request('POST','/api/models/catalog/install',{'repo_id':REPO_ID,'filename':'model.safetensors'}); save('02-install-minilm.json',status,install,'install_minilm','Pinned MiniLM SafeTensors embedding demo installs through catalog verification.',200)
assert status==200 and install.get('id')==MODEL_ID and install.get('task')=='text_embedding', install
manifest=install.get('download_manifest') or {}
assert manifest.get('repo_id')==REPO_ID and manifest.get('revision')==REVISION and manifest.get('license')=='apache-2.0' and manifest.get('verification_status')=='verified', manifest
status,emb_models=request('GET','/api/embedding-models'); save('03-api-embedding-models.json',status,emb_models,'embedding_models_include_minilm','/api/embedding-models includes the installed MiniLM embedding model.',200)
assert status==200 and MODEL_ID in [i.get('id') for i in emb_models.get('items',[])], emb_models
status,models=request('GET','/v1/models'); save('04-v1-models-exclude-minilm.json',status,models,'v1_models_exclude_minilm','Embedding-only MiniLM stays out of chat-runnable /v1/models.',200)
assert status==200 and MODEL_ID not in [i.get('id') for i in models.get('data',[])], models
body={'model':MODEL_ID,'input':['Rust ownership keeps memory safety explicit.','Local embeddings stay local.'],'encoding_format':'float'}
status,emb=request('POST','/v1/embeddings',body); save('05-v1-embeddings-float.json',status,emb,'v1_embeddings_float','/v1/embeddings returns finite 384-dimensional float vectors for two inputs.',200)
assert status==200 and emb.get('object')=='list' and emb.get('model')==MODEL_ID, emb
data=emb.get('data'); assert isinstance(data,list) and len(data)==2, emb
for idx,item in enumerate(data):
    vec=item.get('embedding'); assert item.get('object')=='embedding' and item.get('index')==idx and isinstance(vec,list) and len(vec)==384 and all(isinstance(v,float) and math.isfinite(v) for v in vec), item
f=emb.get('fathom') or {}; assert f.get('runtime')=='candle-bert-embeddings' and f.get('embedding_dimension')==384 and f.get('scope')=='verified local embedding runtime only', emb
status,base64=request('POST','/v1/embeddings',{'model':MODEL_ID,'input':'Rust ownership keeps memory safety explicit.','encoding_format':'base64'}); save('06-v1-embeddings-base64-refusal.json',status,base64,'base64_refusal','Base64 embeddings remain refused with invalid_request and no fake data.',400)
assert status==400 and (base64.get('error') or {}).get('code')=='invalid_request' and 'data' not in base64, base64
status,chat=request('POST','/v1/chat/completions',{'model':MODEL_ID,'messages':[{'role':'user','content':'Hello'}],'max_tokens':8}); save('07-chat-embedding-model-refusal.json',status,chat,'chat_embedding_refusal','Embedding-only model is refused for chat with no fake choices.',501)
assert status==501 and 'choices' not in chat, chat
summary={'schema':'fathom.minilm_embeddings_optional_api_acceptance.summary.v1','passed':all(c['status']=='passed' for c in checks),'repo_commit':repo_commit,'started_at':started_at,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'base_url':base,'artifact_dir':'.','model_dir':'models/','state_dir':'state/','log_dir':'logs/','model_id':MODEL_ID,'repo_id':REPO_ID,'revision':REVISION,'checks':checks,'caveats':['Optional local embedding evidence only; not default CI.','Does not prove embedding quality, retrieval quality, latency, throughput, production readiness, legal suitability, arbitrary Hugging Face execution, ONNX chat, streaming, external proxying, or full OpenAI API parity.','Does not claim GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference.']}
(artifacts/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
(artifacts/'summary.md').write_text('# MiniLM embeddings optional API acceptance artifacts\n\n'+f"- Result: `{'passed' if summary['passed'] else 'failed'}`\n- Repo commit: `{repo_commit}`\n- Model: `{MODEL_ID}`\n- Upstream: `{REPO_ID}` at `{REVISION}`\n- Scope: optional local embedding API evidence only; not default CI.\n- Artifact directory: `.`\n- State directory: `state/`\n- Model directory: `models/`\n- Server log: `logs/server.log`\n\n## Checks\n\n"+'\n'.join(f"- `{c['name']}`: `{c['status']}` ({c['artifact']})" for c in checks)+'\n\n## What this does not prove\n\n- No embedding quality, retrieval quality, latency, throughput, production readiness, legal suitability, arbitrary Hugging Face execution, ONNX chat/general execution, streaming, external proxying, or full OpenAI API parity claim.\n- No public/runtime GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference claim.\n')
assert summary['passed'], summary
PY

python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py "$ARTIFACT_DIR"
echo "MiniLM embeddings optional API acceptance passed. Artifacts: $ARTIFACT_DIR"
