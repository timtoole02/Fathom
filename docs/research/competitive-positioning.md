# Fathom Competitive Positioning

## Thesis

Fathom should not position itself as “another local model app” or “a llama.cpp wrapper.” The stronger wedge is:

> Drop in a model. Fathom tells you what it is, what can run it, what cannot, why, and gives you one clean API to use it.

That means Fathom is a **Rust-native local inference traffic controller**: artifact inspection, machine inspection, backend-lane routing, truthful capability reporting, and OpenAI-compatible access.

## Nearby products

### llama.cpp / llama-server

- Strength: fast, proven, massive ecosystem, GGUF-centered, practical OpenAI-compatible server.
- Gap Fathom targets: llama.cpp’s center of gravity is GGUF. Fathom’s differentiator is not making GGUF the only first-class path.

### Ollama

- Strength: simple install/use experience, strong local model UX, popular developer API story.
- Gap Fathom targets: Ollama is model-package/runtime oriented, not a universal artifact inspector/router that explains arbitrary local artifacts and machine-specific backend options.

### LM Studio / Jan / GPT4All / KoboldCpp

- Strength: approachable desktop UX and model browsing.
- Gap Fathom targets: these are usually app/runtime bundles. Fathom should be a lower-level universal runtime layer that can power apps like ForgeLocal.

### Shimmy / small Rust inference servers

- Strength: Rust, small, OpenAI-compatible.
- Gap Fathom targets: most are GGUF-focused or wrapper-focused. Fathom’s wedge is multi-format backend routing and truthfulness.

### ONNX Runtime GenAI / vendor runtimes

- Strength: strong lanes for specific deployment ecosystems.
- Gap Fathom targets: ONNX, MLX, Core ML, and TensorRT are lanes, not the unified traffic controller.

## Product claim to earn

Fathom has now earned the first narrow “run” step for selected SafeTensors/HF chat and embedding lanes, but it has not earned arbitrary model support or a broad “bring any model” claim. Continue earning the full product claim in stages:

1. **Understand**: detect local artifacts and machine capabilities.
2. **Explain**: show what can run, what cannot, and why.
3. **Route**: map artifacts to backend lanes.
4. **Run**: execute real generation for specific lanes.
5. **Unify**: expose one OpenAI-compatible API across lanes.

## Messaging

Bad claim:

> We run every model.

Better claim:

> Fathom makes local AI runtimes understandable, portable, and honest.

Future claim after real backends exist:

> Bring any model artifact. Fathom inspects it, chooses the right backend lane, and gives you one local API.

## Strategic sequencing

1. Build the capability router first.
2. Keep proving the shipped narrow SafeTensors/HF runnable lanes with backend-only acceptance, benchmarks, and fresh-clone QA before broadening architecture support.
3. Add GGUF as a native lane to compete where llama.cpp is strongest.
4. Add ONNX as the portability lane.
5. Add platform lanes: MLX/CoreML for Apple, TensorRT for NVIDIA.

## Non-negotiable truthfulness

Fathom must never mark a format “supported” because it recognizes the extension. Recognition is not runtime support. UI/API must distinguish:

- detected
- metadata readable
- mapped to planned backend lane
- blocked for safety/platform reasons
- runnable
