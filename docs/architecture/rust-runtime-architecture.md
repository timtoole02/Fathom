# Fathom Rust Runtime Architecture

## Workspace

- `fathom-core`: model artifact detection, loader traits, runtime traits, metadata.
- `fathom-server`: HTTP/OpenAI-compatible API and UI-facing state.
- future `fathom-backends-*`: GGUF, SafeTensors/Candle, ONNX, MLX/CoreML/TensorRT adapters.

## Core traits

- `ModelLoader`: detects and loads metadata/tensors.
- `InferenceRuntime`: reports capability and generates outputs.
- `TokenizerBackend`: normalizes tokenizer differences.
- `DeviceBackend`: CPU, CUDA, Metal, Vulkan, DirectML lanes.

## Rule

Every backend must expose capabilities. UI must show detected/imported/runnable separately.
