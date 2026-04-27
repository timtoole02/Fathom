# Fathom Runtime Safety and Supply Chain Policy

Fathom’s runtime promise is capability truth, not broad-but-unsafe loading. A model artifact is only runnable when its container, metadata, tokenizer/config, backend runtime, native dependencies, license, and provenance have all passed explicit gates.

This policy applies to local artifact registration, future catalog/download flows, and backend lanes. It should be enforced before any UI/API state can move from `detected`, `metadata-readable`, `planned`, or `blocked` to `runnable`.

## Core principles

1. **Safe by default.** Parsing metadata is allowed where the parser is memory-safe and non-executing. Deserialization formats that can execute code stay blocked unless imported through a separate trusted workflow.
2. **No silent execution.** Loading a model must never run repository code, Python pickle reducers, custom tokenization code, shell hooks, install scripts, or conversion scripts by default.
3. **Capability truth.** Detection is not support. Metadata readability is not runtime readiness. A backend becomes runnable only after real execution tests and clear compatibility checks.
4. **Local-first provenance.** Remote catalog downloads must be pinned, resumable, hash-verified, and license-gated before use.
5. **Platform honesty.** Platform-specific runtimes such as Core ML, MLX, TensorRT, CUDA, Metal, or ONNX Runtime must declare native dependency and hardware/version requirements.
6. **License visibility.** Fathom may inspect unknown-license artifacts locally, but catalog/import flows must surface model and runtime licenses before enabling operational use.

## Capability state policy

| State | Meaning | Allowed actions | Must not do |
| --- | --- | --- | --- |
| `detected` | File/package shape is recognized. | Show file type, size, obvious notes. | Parse unsafe payloads or claim loadability. |
| `metadata-readable` | Safe metadata/config/tokenizer fields were parsed without executing code. | Show architecture, dtype, shard map, tokenizer presence, license metadata. | Allocate full tensors or run inference. |
| `planned` | A backend lane exists, but gates are incomplete. | Show blockers and implementation status. | Mark as runnable or route `/v1/chat/completions` to it. |
| `blocked` | Known unsafe or unsupported artifact path. | Explain the block and safe alternatives. | Auto-convert, auto-download helpers, or attempt execution. |
| `runnable` | All policy, compatibility, and test gates passed. | Route generation requests and expose in `/v1/models`. | Hide warnings that affect correctness, safety, or licensing. |

## Format-specific risk review and gates

### SafeTensors / Hugging Face folders

**Risk profile:** SafeTensors avoids arbitrary code execution in the weight container, but HF model folders are still supply-chain bundles: `config.json`, `generation_config.json`, `tokenizer.json`, SentencePiece files, shard indexes, README/model cards, and optional custom code markers may influence runtime behavior.

**Allowed default status:** `detected` or `metadata-readable`; `runnable` only for the narrow SafeTensors/HF packages whose architecture, tokenizer, dtype, loader, runtime, and API behavior are already fixture-tested. Other SafeTensors/HF packages remain `metadata-readable`, `planned`, `blocked`, or `unsupported` until their gates pass.

**Required gates before runnable:**

- Use a non-executing SafeTensors parser; reject malformed headers, oversized metadata, invalid offsets, duplicate tensor names, overlapping tensor ranges, and unsupported dtypes.
- For `model.safetensors.index.json`, require every referenced shard to remain inside the model root after path normalization; reject absolute paths, `..`, symlinks escaping the root, and missing shards.
- Enforce bounded reads and memory-mapping limits: compare declared tensor sizes with actual file sizes before allocation.
- Parse `config.json` with an allowlist of supported `model_type`/`architectures`; unknown architectures stay `metadata-readable` or `planned`.
- Parse tokenizer artifacts with non-executing libraries only. Do not honor `trust_remote_code`, custom tokenizer Python, or arbitrary repo scripts.
- Require a complete generation path: config -> tokenizer/chat template -> tensor load -> forward pass -> sampler -> emitted tokens.
- Require fixtures for at least one small public or generated model package and API smoke coverage.
- Surface license/provenance metadata if present; catalog downloads require explicit license acceptance/visibility.

**Blocked by default:** HF repos requiring `trust_remote_code`, Python modules, custom ops, or post-download conversion to become usable.

### PyTorch `.bin` / pickle artifacts

**Risk profile:** Legacy `pytorch_model.bin` commonly uses Python pickle, which can execute arbitrary code during deserialization.

**Allowed default status:** `blocked`.

**Required gates for any future trusted-import flow:**

- Keep direct `.bin` runtime loading disabled in normal model registration.
- Prefer a SafeTensors sibling when present and tell the user why `.bin` is blocked.
- If trusted import is added, make it a separate, explicit, user-acknowledged workflow with clear warnings.
- Run import in an isolated sandbox/process with no network, restricted filesystem access, resource limits, and an allowlisted output path.
- Verify source provenance, expected hashes, declared license, and conversion output hashes.
- Convert only to a safe internal representation such as SafeTensors; never mark the original pickle artifact runnable.
- Log/import-record the tool versions and inputs used for reproducibility.

**Blocked by default:** Blind pickle load, `torch.load` in-process, remote code execution, and automatic conversion during ordinary registration.

### ONNX

**Risk profile:** ONNX is a graph plus weights, not usually an executable source format, but operator kernels and runtimes are native-code dependencies. Malformed graphs can trigger parser/runtime bugs, high memory use, or unsupported-op behavior.

**Allowed default status:** `detected` or `planned` until a chosen ONNX adapter passes tests. The only current runnable ONNX lane is the non-default, feature-gated MiniLM embedding fixture path; it requires the pinned single-file `model_quantized.onnx` artifact, package-root containment for model/config/tokenizer, and refusal of external-data sidecars or custom-op/plugin/shared-library configuration before ONNX Runtime sees the model path.

**Required gates before runnable:**

- Choose and document the runtime (`ort`, `tract`, or another adapter) and its license/native dependency footprint.
- Parse graph metadata without running inference; reject external data paths that escape the model root.
- Validate opset range, operator allowlist/support matrix, tensor shapes, dtypes, and memory budget before session creation.
- Require tokenizer/prompt wrapper support for text generation models; ONNX graph load alone is not chat support.
- Disable runtime graph optimizations or custom operators that load arbitrary shared libraries unless explicitly configured and disclosed.
- Test with a small ONNX fixture and one API path that performs real inference or a documented non-text task.

**Blocked by default:** Custom op shared libraries, external data outside root, and graph/runtime combinations with unknown native dependency provenance.

### Core ML / MLX

**Risk profile:** Apple-specific formats rely on platform frameworks and/or MLX conventions. Risks are mainly platform availability, native runtime behavior, package shape ambiguity, and license/provenance clarity rather than pickle-style execution.

**Allowed default status:** `planned` on Apple platforms; `unsupported` elsewhere.

**Required gates before runnable:**

- Gate by OS/architecture and runtime availability; report unsupported on non-Apple platforms.
- Detect package layout precisely (`.mlpackage`, `.mlmodel`, `.mlmodelc`, MLX `weights.safetensors`/`model.safetensors` plus config conventions) without overclaiming.
- Verify files remain within package root; reject symlink/path escapes.
- Document whether execution uses Core ML system frameworks, MLX Rust/Python bridge, or a custom adapter.
- Confirm model task type and tokenizer/story separately; a Core ML package is not automatically a chat model.
- Test on Apple Silicon/macOS with a small fixture and expose platform-specific limitations in capabilities.

**Blocked by default:** Any MLX path that requires arbitrary Python repo code or unreviewed package scripts.

### TensorRT plan/engine

**Risk profile:** TensorRT plans are compiled, hardware/version-specific native artifacts. They are not portable and may be incompatible across GPU architecture, CUDA, TensorRT, driver versions, precision modes, and plugin sets.

**Allowed default status:** `unsupported` unless NVIDIA runtime is explicitly requested/detected; otherwise `planned` with blockers.

**Required gates before runnable:**

- Require explicit NVIDIA runtime opt-in and detected compatible CUDA/TensorRT/driver/GPU capability.
- Record engine build metadata where available; reject plans with unknown or incompatible TensorRT/CUDA/GPU versions.
- Disallow arbitrary plugin shared libraries by default; if plugins are ever supported, require user opt-in, hash pinning, and path allowlists.
- Enforce input/output binding validation, shape constraints, precision expectations, and memory budget before execution.
- Treat TensorRT plans as backend-specific binary artifacts, not general Fathom model packages.
- Test on a matching NVIDIA environment with an engine fixture and clear skip behavior on non-NVIDIA hosts.

**Blocked by default:** Unknown plugins, incompatible engine plans, and plans downloaded without provenance/hash verification.

### Remote downloads and catalog artifacts

**Risk profile:** Catalog support introduces network, integrity, license, and update risks even for otherwise safe formats.

**Policy gates before any downloaded artifact can be registered:**

- Download only from HTTPS or configured trusted local mirrors.
- Pin repository ID, revision/commit, filename set, expected size, and SHA-256 or stronger hashes where available.
- Store download manifests with source URL, resolved revision, timestamp, file hashes, license fields, and runtime-lane expectation.
- Verify hash before detection/routing; never execute or parse unsafe formats during download.
- Use atomic writes and resume validation; incomplete downloads cannot be registered.
- Prevent path traversal during archive/external-data extraction.
- Surface model license and usage restrictions before enabling catalog install; require explicit acknowledgement for unknown, restrictive, or non-commercial catalog license status before download starts.
- Never honor remote `trust_remote_code` or auto-install repo dependencies.

### Native dependencies and backend adapters

**Risk profile:** ONNX Runtime, CUDA, TensorRT, Metal/Core ML, MLX, tokenizer native libraries, and BLAS/GPU stacks may introduce native-code security, deployment, and license obligations.

**Policy gates before a native dependency is accepted:**

- Document why the dependency is needed, alternatives considered, supported platforms, license, binary distribution model, and update policy.
- Prefer Rust crates or system frameworks with clear provenance; avoid opaque bundled binaries unless necessary and documented.
- Pin crate versions in `Cargo.lock` and track transitive licenses for release builds.
- Provide graceful capability degradation when the native runtime is unavailable.
- Do not dynamically load user-provided shared libraries by default.
- Include CI/test behavior for hosts without the native accelerator.

### Licenses

**Risk profile:** Model weights, tokenizer files, backend runtimes, and generated outputs may carry different license constraints. Fathom should not present license-unknown artifacts as safe for all use.

**Policy gates:**

- Parse common license locations where safe: `README.md`/model card front matter, `LICENSE`, `config.json` license fields, Hugging Face metadata manifests, and catalog metadata.
- Label license state as one of: `known-permissive`, `known-restrictive`, `non-commercial`, `unknown`, or `conflict`; API catalog responses currently expose the narrower install-policy status as `permissive`, `unknown`, or `restrictive`.
- Allow local inspection of unknown-license artifacts, but catalog install and production routing should show an explicit warning and require acknowledgement before network install.
- Never rewrite or hide upstream license text.
- Track runtime dependency licenses separately from model licenses.

## Implementation checklist

### Detection and package scanning

- [ ] Normalize and canonicalize all artifact paths before classification.
- [ ] Reject symlinks, shard paths, external data, or package entries that escape the model root.
- [ ] Add per-format size limits and declared-size-vs-actual-file checks before allocating tensors.
- [ ] Extend capability notes to include safety blockers separately from implementation blockers.
- [ ] Keep PyTorch `.bin` `blocked` until a trusted-import design exists.

### SafeTensors/HF runnable lane

- [ ] Use a safe SafeTensors parser and validate headers, offsets, dtypes, and shard completeness.
- [ ] Add architecture allowlist for first supported model family.
- [ ] Add tokenizer/chat-template support without remote code.
- [ ] Add real token generation fixture before marking any SafeTensors model runnable.
- [ ] Add API tests proving `/v1/models` and `/v1/chat/completions` only expose runnable models truthfully.

### Trusted import workflow

- [ ] Design a separate import command/API that cannot be triggered by ordinary registration.
- [ ] Require explicit user acknowledgement for pickle risk.
- [ ] Run conversion/import in an isolated process with no network and limited filesystem access.
- [ ] Emit SafeTensors or another safe internal artifact plus a manifest of input hashes/tool versions.
- [ ] Keep original `.bin` artifacts non-runnable after conversion.

### ONNX/Core ML/MLX/TensorRT lanes

- [ ] Pick runtime adapters and document dependency/license impact before adding them.
- [ ] Implement platform/runtime compatibility checks before session creation.
- [ ] Reject custom ops/plugins/shared libraries by default.
- [ ] Add fixture-backed tests per lane before claiming runnable support.
- [ ] Ensure unsupported host platforms return truthful `unsupported` or `planned`, not generic failure.

### Remote download/catalog flow

- [ ] Add a signed or hash-pinned catalog manifest format.
- [ ] Verify hashes and final file sizes before registration.
- [ ] Store provenance and license metadata next to downloaded artifacts.
- [ ] Block path traversal and package-root escapes.
- [ ] Show license and safety warnings before install/activation.

### Native dependency governance

- [ ] Add a dependency decision record for each backend adapter.
- [ ] Track native binary provenance and update cadence.
- [ ] Generate or review license inventory for release artifacts.
- [ ] Add feature flags so risky/platform-specific backends are opt-in until mature.

## Recommended near-term gates for Fathom

1. Keep `.bin` blocked in the capability router and docs.
2. Keep the implemented SafeTensors/HF lanes narrow: runnable only for validated GPT-2/TinyStories, Llama/SmolLM2, Qwen2, Phi, Mistral, Gemma, and MiniLM embedding package shapes that pass loader/runtime/API tests.
3. Treat ONNX chat, Core ML/MLX, and TensorRT as planned/platform-specific lanes until native dependency decisions are recorded. ONNX MiniLM embeddings remain a separate feature-gated fixture path, not chat support.
4. Keep catalog downloads pinned, hash-verified, license-visible, and manifest-recorded before registration.
5. Add safety blockers to capability reports so UI can distinguish “not implemented yet” from “unsafe by default.”
