# Third-Party Notes

This repository vendors a small subset of the original Python MeloTTS implementation under `tools/melotts_py/`.

Upstream project:
- MeloTTS
- Copyright (c) 2024 MyShell.ai
- License: MIT

The vendored Python files are retained only to support ONNX export.

This repository also depends on third-party runtime and tooling components such as:
- ONNX Runtime
- PyTorch
- Transformers
- Hugging Face Hub
- cached_path
- numba

Please review each dependency's upstream license before redistribution.
