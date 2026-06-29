export const MODEL_REGISTRY = Object.freeze({
  "bonsai-1.7b-q1": Object.freeze({
    id: "bonsai-1.7b-q1",
    label: "Bonsai 1.7B",
    engine: "wllama",
    repo: "prism-ml/Bonsai-1.7B-gguf",
    file: "Bonsai-1.7B-Q1_0.gguf",
  }),
  "ternary-bonsai-1.7b-q2": Object.freeze({
    id: "ternary-bonsai-1.7b-q2",
    label: "Ternary Bonsai 1.7B",
    engine: "transformers",
    model: "onnx-community/Ternary-Bonsai-1.7B-ONNX",
    dtype: "q2",
    device: "webgpu",
  }),
});

export function modelConfig(modelId) {
  const config = MODEL_REGISTRY[modelId];
  if (!config) {
    throw new Error(`Unknown browser model: ${modelId}`);
  }
  return config;
}
