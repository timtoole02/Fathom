# Public launch checklist

Use this checklist when validating a clean public Fathom checkout or preparing a launch-facing handoff. It is scoped to the current local, no-download public contract plus optional networked backend acceptance evidence.

## 1. Clean clone and install

```bash
git clone https://github.com/timtoole02/Fathom/ fathom
cd fathom
npm --prefix frontend install
cargo test -q
```

This confirms the ordinary Rust and frontend toolchains are present. It does not download catalog model fixtures or enable the non-default ONNX Runtime feature.

## 2. No-download verification gates

Run these before treating docs/API contract changes as launch-ready:

```bash
git diff --check
python3 -m py_compile scripts/public_api_contract_qa.py scripts/public_contract_smoke_artifact_qa.py
python3 scripts/ci_static_policy.py
python3 scripts/public_api_contract_qa.py
python3 scripts/public_contract_smoke_artifact_qa.py
bash -n scripts/public_api_contract_smoke.sh
bash -n scripts/backend_acceptance_smoke.sh
bash scripts/public_risk_scan.sh
npm --prefix frontend run qa:copy
```

For the real backend no-download contract check, run:

```bash
bash scripts/public_api_contract_smoke.sh
```

That smoke starts `fathom-server` with isolated temporary state/model/log directories and checks the public `/v1` routing/refusal boundary from [`docs/api/public-contract.json`](api/public-contract.json) and [`docs/api/v1-contract.md`](api/v1-contract.md). It does not install catalog models, download fixtures, enable ONNX features, call providers, or prove model quality.

To keep a share-safe pass/fail summary for release handoff, set `FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR` to a directory you control:

```bash
FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR=public-contract-artifacts \
  bash scripts/public_api_contract_smoke.sh
```

The optional `public-contract-smoke-summary.json` and `.md` files include commit, manifest path/name/status, endpoint checks, boundary checks, and scope caveats only. They intentionally omit local temp paths, server log tails, request secrets, and model/provider payloads. Validate them offline before sharing with `python3 scripts/public_contract_smoke_artifact_qa.py public-contract-artifacts`.

## 3. Backend/API quick smoke

For a backend-only manual pass:

```bash
bash scripts/start-backend.sh
BASE=http://127.0.0.1:8180
curl -fsS "$BASE/v1/health" | python3 -m json.tool
curl -fsS "$BASE/v1/models" | python3 -m json.tool
bash scripts/stop.sh
```

Use [`docs/api/backend-only-quickstart.md`](api/backend-only-quickstart.md) for catalog install examples, TinyStories chat, MiniLM embeddings, retrieval, and expected refusals. Those catalog demos require network access for pinned fixture downloads.
For a compact refusal/non-claim checklist before Phase 16 work, see [`docs/api/refusal-boundary-matrix.md`](api/refusal-boundary-matrix.md).

## 4. Optional networked acceptance smoke

When you need fuller backend evidence and network access is available:

```bash
FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh
```

Review `summary.md` first, then `summary.json` and the named JSON artifacts. Keep `summary.local.json` private unless you have reviewed it; local paths and runner-specific details belong there. Before sharing artifacts publicly, run `bash scripts/public_risk_scan.sh` and manually inspect logs/full payloads for local paths or request text.

## What this launch currently proves

- The documented no-download public `/v1` contract routes and refusal envelopes work against the real backend process.
- Default CI stays offline with respect to model downloads, networked acceptance smoke, and non-default ONNX feature tests.
- Offline artifact QA covers optional public-contract smoke summaries, and the optional backend acceptance smoke can produce share-safe success/failure summaries for the current pinned fixture path when networked downloads succeed.
- The current launch evidence snapshot is recorded in [`public-launch-evidence.md`](public-launch-evidence.md).

## What this launch does not prove

- It is not production readiness, performance capacity, model quality, or legal/license advice.
- It is not full OpenAI API parity; streaming chat and many OpenAI endpoints are unsupported.
- It does not prove external provider proxying; saved external entries are metadata placeholders and chat is refused.
- It does not prove general GGUF runtime, tokenizer execution, dequantization, generation, ONNX chat, PyTorch `.bin` loading, or arbitrary SafeTensors/Hugging Face execution.

## Troubleshooting

- **Missing tools:** install Rust (`cargo`), Node.js/npm, `curl`, and `python3`; Vite requires Node `20.19+` or `22.12+`.
- **Port conflicts:** stop old runs with `bash scripts/stop.sh`, or set `FATHOM_PORT` for backend-only runs. The contract smoke chooses a temporary local port automatically.
- **First build is slow:** `scripts/start.sh` and `scripts/start-backend.sh` build the Rust backend in release mode; the first run can take a few minutes.
- **Model download/network errors:** no-download gates should still pass without network. Catalog demos and `scripts/backend_acceptance_smoke.sh` need network access to fetch pinned fixtures.
- **Logs:** normal runs write backend logs under `~/.fathom/logs`; isolated smoke scripts print their temporary artifact/log locations while running and clean them unless configured to keep artifacts.
