# Security

Fathom is an early local inference sandbox. Treat it as developer software for local experiments, not as a hardened production service.

## Local threat model

- Fathom is designed to bind to `127.0.0.1` for local use.
- The local API has no built-in authentication or multi-user authorization.
- Do not expose Fathom directly to the public internet or an untrusted LAN.
- If you put a proxy, tunnel, container network, or remote desktop in front of Fathom, you are responsible for authentication, TLS, rate limits, request logging policy, and network isolation.

## Model artifact safety posture

Fathom should be truthful about what it loads and conservative about what it refuses.

- SafeTensors-first: verified local SafeTensors/Hugging Face lanes are the current runnable focus.
- PyTorch `.bin` / pickle artifacts are blocked; Fathom must not deserialize them as trusted code or data.
- Hugging Face `remote_code`, Python model code, and other remote code execution paths are not supported.
- ONNX support is narrow and feature-gated. The current ONNX path is for the pinned MiniLM embedding fixture only, with defensive package checks; no ONNX chat or general ONNX runtime support is claimed.
- GGUF support is metadata/readiness inspection only. GGUF tokenizer execution, runtime weight loading, dequantization, kernels, generation, and inference are not public runtime features yet.
- New loaders should fail closed: if a package is recognized but not safely runnable for the selected task, return a clear refusal instead of guessing or faking output.

## Artifact, prompt, and log privacy

Fathom runs locally, but generated artifacts can still contain sensitive data:

- API payloads may include prompts, retrieval snippets, document text, and model ids.
- Logs may include local paths, state directories, request summaries, or errors from model parsing.
- Acceptance-smoke artifacts include share-safe summaries where possible, but full JSON payloads and local logs still need human review before publication.

Before sharing logs, benchmark output, acceptance artifacts, screenshots, or bug reports publicly, review them for local paths, personal identifiers, prompt text, private documents, credentials, and model-store details. Run:

```bash
bash scripts/public_risk_scan.sh
```

The scanner is a guardrail for tracked files, not a complete privacy audit.

## Reporting security issues

Please do not post exploit details, private prompts, credentials, or sensitive local paths in a public issue.

If there is no private security contact available, open a minimal GitHub issue that says you have a security concern and request a private channel. Include only enough public detail to route the report, such as the affected area and impact category. Share reproduction steps and sensitive details only after a private channel is arranged.
