# MeloTTS.C

![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)
![Runtime](https://img.shields.io/badge/runtime-ONNX%20Runtime-green)
![Platform](https://img.shields.io/badge/platform-Windows%20x64-lightgrey)
![Status](https://img.shields.io/badge/status-English--only-orange)

MeloTTS.C is a standalone C++17 English-only text-to-speech runtime built around ONNX Runtime.

It includes:

- 英文 `BERT` ONNX 模型
- 导出的 `MeloTTS English infer` ONNX 模型

It does not depend on a Python bridge at runtime.

## Highlights

- Standalone C++ runtime for English MeloTTS inference
- ONNX-based deployment path for both BERT and acoustic inference
- Streaming-style chunked synthesis demo
- Local ONNX export tooling bundled in the repository
- Incremental `make` targets for dependencies, models, build, and cleanup
- Windows GitHub Actions build validation

## Repository Layout

- `include/melotts_engine.h`
  Public C++ API exposing `TTSEngine` and `WriteWaveFile`
- `demo/melotts_cli.cpp`
  Basic command-line synthesis demo
- `demo/melotts_stream_cli.cpp`
  Chunked streaming demo with per-chunk metrics
- `config/english_onnx.example.ini`
  Example runtime configuration
- `tools/export_english_onnx.py`
  Local ONNX export entry point

## Scope

- English-only
- Pure C++ text front-end
- CMU dictionary pronunciation when available
- Built-in fallback pronunciation rules for unknown words
- Pure C++ WordPiece tokenizer for BERT input preparation

The repository provides a practical standalone English C++ pipeline, but its text preprocessing is not intended to be token-for-token identical to the original Python `g2p_en` stack.

## Quick Start

### 1. Install Python export dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Build everything with automatic dependency and model recovery

```powershell
make all
```

This target will:

- ensure `third_party/onnxruntime-win-x64-1.20.1` exists
- ensure required files under `models/` exist
- configure and build the C++ targets

### 3. Run the basic demo

```powershell
./build/Release/melotts_cli.exe \
  --config config/english_onnx.example.ini \
  --text "Hello from MeloTTS C plus plus." \
  --output outputs/english_cpp_demo.wav
```

### 4. Run the streaming demo

```powershell
./build/Release/melotts_stream_cli.exe \
  --config config/english_onnx.example.ini \
  --text "Hello. This is a streaming C plus plus demo. It synthesizes chunk by chunk." \
  --output outputs/english_cpp_stream.wav \
  --chunk-dir outputs/stream_chunks
```

## Models

You can prepare the required model files yourself or generate them locally with:

```powershell
python tools/export_english_onnx.py --output-dir models
```

Expected outputs:

1. `bert_base_uncased.onnx`
Expected to output a tensor of shape `[1, token_count, 768]`
2. `bert-base-uncased-vocab.txt`
Standard BERT vocabulary
3. `melotts_en_infer.onnx`
Expected to be equivalent to the English `SynthesizerTrn.infer(...)` graph
4. 可选的 `cmudict.rep`
If provided, CMU dictionary pronunciation is preferred. Otherwise the runtime falls back to built-in English pronunciation rules.

If you already have a local English checkpoint, you can also export with:

```powershell
python tools/export_english_onnx.py --ckpt-path path/to/checkpoint.pth --config-path path/to/config.json
```

If your exported ONNX graph uses different input or output names, update `config/english_onnx.example.ini`.

## Build

```powershell
cmake -S . -B build -DONNXRUNTIME_ROOT="path/to/onnxruntime-win-x64-1.20.1"
cmake --build build --config Release
```

If ONNX Runtime is stored inside the repository, the recommended path is:

```text
third_party/onnxruntime-win-x64-1.20.1
```

So a repository-local build can be done with:

```powershell
cmake -S . -B build -DONNXRUNTIME_ROOT="third_party/onnxruntime-win-x64-1.20.1"
cmake --build build --config Release
```

Windows runtime note:

```text
third_party/onnxruntime-win-x64-1.20.1/lib
```

The CMake build copies the matching `onnxruntime.dll` and `onnxruntime_providers_shared.dll` into `build/Release/` automatically, which avoids loading an older system-wide ONNX Runtime DLL by mistake.

## API Example

```cpp
#include "melotts_engine.h"

int main() {
  auto config = melotts_engine::TTSEngine::LoadConfig("config/english_onnx.example.ini");
  melotts_engine::TTSEngine engine(std::move(config));

  melotts_engine::SynthesisRequest request;
  request.text = "This is a pure C++ English MeloTTS demo.";

  auto audio = engine.Synthesize(request);
  melotts_engine::WriteWaveFile("outputs/demo.wav", audio, engine.sample_rate());
}
```

Streaming callback example:

```cpp
#include "melotts_engine.h"

int main() {
  auto config = melotts_engine::TTSEngine::LoadConfig("config/english_onnx.example.ini");
  melotts_engine::TTSEngine engine(std::move(config));

  melotts_engine::SynthesisRequest request;
  request.text = "Hello. This is a streaming C plus plus demo.";

  melotts_engine::StreamingOptions options;
  options.max_chars = 64;
  options.silence_ms = 50;

  engine.StreamSynthesize(request, options, [&](const melotts_engine::StreamingChunk& chunk) {
    // chunk.audio contains the chunk audio.
    // Non-last chunks include the configured trailing silence.
    // chunk.first_chunk_latency_ms, chunk.synth_ms, chunk.audio_ms and chunk.rtf expose streaming metrics.
  });
}
```

## Make Targets

The repository root includes a `Makefile` that wraps the most common local workflows.

```makefile
make all
make deps
make model
make clean
make distclean
make purge
```

`make deps`
Ensures `third_party/onnxruntime-win-x64-1.20.1` exists. If missing, it is downloaded and extracted automatically.

`make all`
Runs `make deps`, checks whether required files under `models/` already exist, incrementally restores any missing artifacts, and then builds the C++ targets.

This is the recommended local developer workflow when you want both repository dependencies and model artifacts to be restored automatically.

`make model`
Runs `make deps` and incrementally restores missing ONNX model artifacts under `models/`.

`make clean`
Removes generated files and caches inside the repository, including `build/`, `outputs/`, Python cache directories, and root-level CMake temporary files.

`make distclean`
Runs `make clean` and additionally removes `third_party/onnxruntime-win-x64-1.20.1`.

`make purge`
Runs `make distclean` and additionally removes `models/`.

Current C++ library target name: `melotts_engine`

## Clean

If you prefer to use CMake directly instead of the repository `Makefile`, the equivalent cleanup behavior is:

1. Remove compiled build products

```powershell
cmake --build build --target clean --config Release
```

2. Remove generated runtime audio outputs

```powershell
cmake --build build --target distclean --config Release
```

Notes:
- `clean` is provided by the underlying CMake generator and removes compiled artifacts
- `distclean` is a custom CMake target defined in `CMakeLists.txt` and removes `outputs`
- removing `build/` is the simplest way to return to an unbuilt state
- `make clean` is more aggressive than `cmake --build build --target clean` because it removes the entire `build/` tree
- after `make distclean` or `make purge`, the next `make all` or `make model` will automatically restore ONNX Runtime
- model restoration is incremental: if only BERT files or only the acoustic model is missing, only the missing part is exported

## Technical Notes

- During English inference, `ja_bert` carries the 768-channel English BERT features
- `bert` is filled with zeros and uses shape `[1, 1024, T]`
- `phone`, `tone`, and `language` follow the MeloTTS `add_blank` convention
- Default values are `english_language_id=2` and `english_tone_start=7`
- The symbol table is embedded directly into the C++ runtime and does not require an external `symbols.py`
- `cmudict_path` is optional and falls back to built-in pronunciation rules when missing
- `StreamSynthesize(...)` provides library-level callback-based streaming output
- Chunk splitting is sentence-first, then word-boundary-first, and only hard-splits when necessary
- Each `StreamingChunk` carries `first_chunk_latency_ms`, `synth_ms`, `audio_ms`, and `rtf`
- `melotts_stream_cli` also prints summary metrics such as `chunks`, `samples`, `total_synth_ms`, `total_audio_ms`, and `avg_rtf`

## Repository Notes

- Runtime inference does not depend on an external `symbols.py`
- `cmudict_path` is optional
- Build, output, and cleanup behavior are repository-local
- `tools/export_english_onnx.py` is included in this repository
- Minimal Python export dependencies are vendored under `tools/melotts_py/`
- `requirements.txt` pins `transformers` below `4.53` to keep the current ONNX export path compatible
- See `THIRD_PARTY.md` for vendored component and dependency notes

## Notes

- The repository does not ship pretrained model weights by default
- If your acoustic ONNX export does not directly output waveform audio, the runtime interface may need adjustment
- The streaming demo is chunked sentence-style streaming, not frame-level low-latency vocoder streaming
- Each chunk is still synthesized independently, with 50 ms of silence inserted between non-final chunks for easier concatenation

## Roadmap

- Add CI builds for the standalone repository
- Add an HTTP or WebSocket streaming service demo
- Improve English text normalization and fallback pronunciation quality
- Add packaging guidance for redistributable releases
