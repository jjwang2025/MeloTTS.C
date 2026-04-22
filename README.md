# MeloTTS.C

这个仓库提供一个纯 C++17 的 English-only 语音合成接口，运行时使用 `ONNX Runtime` 加载：

- 英文 `BERT` ONNX 模型
- 导出的 `MeloTTS English infer` ONNX 模型

它不依赖 Python 桥接脚本。

## 包含内容

- `include/melotts_engine.h`
  C++ 接口库头文件，提供 `TTSEngine` 和 `WriteWaveFile`
- `demo/melotts_cli.cpp`
  命令行演示程序
- `config/english_onnx.example.ini`
  模型与输入输出名配置示例

## Scope

- 仅支持 English-only
- 文本前端为纯 C++ 实现
- 优先使用 `CMU dict` 发音
- 未登录词使用简单字母到音素回退规则
- BERT 侧使用纯 C++ `WordPiece` tokenizer

这个仓库提供一个可运行的纯 C++ English-only 方案，但英文前处理与原始 Python `g2p_en` 不会逐 token 完全一致。

## 需要准备的模型

你可以自己准备这些文件，也可以直接用仓库内的导出脚本生成：

```powershell
python tools/export_english_onnx.py --output-dir models
```

生成结果：

1. `bert_base_uncased.onnx`
要求输出一个张量，形状为 `[1, token_count, 768]`
2. `bert-base-uncased-vocab.txt`
标准 BERT 词表
3. `melotts_en_infer.onnx`
要求等价于 `SynthesizerTrn.infer(...)` 的 English 导出图
4. 可选的 `cmudict.rep`
如果提供，会优先使用 CMU 字典发音；如果不提供，则自动回退到内置英文规则发音

如果你已经有本地英文 checkpoint，也可以这样导出：

```powershell
python tools/export_english_onnx.py --ckpt-path path/to/checkpoint.pth --config-path path/to/config.json
```

如果你的 ONNX 输入名或输出名不同，修改 `config/english_onnx.example.ini` 即可。

## 构建

```powershell
cmake -S . -B build -DONNXRUNTIME_ROOT="path/to/onnxruntime-win-x64-1.20.1"
cmake --build build --config Release
```

## Python Export Dependencies

如果你需要在当前仓库内导出 ONNX，请先安装：

```powershell
python -m pip install -r requirements.txt
```

如果你把 ONNX Runtime 放在仓库内，推荐路径是：

```text
third_party/onnxruntime-win-x64-1.20.1
```

所以你在这个仓库里可以直接运行：

```powershell
cmake -S . -B build -DONNXRUNTIME_ROOT="third_party/onnxruntime-win-x64-1.20.1"
cmake --build build --config Release
```

Windows 运行时请确保把 `onnxruntime.dll` 复制到可执行文件目录，或者把以下目录加入 `PATH`：

```text
third_party/onnxruntime-win-x64-1.20.1/lib
```

当前仓库的 `CMake` 构建会在生成 demo 可执行文件后，自动把匹配版本的 `onnxruntime.dll` 和 `onnxruntime_providers_shared.dll` 复制到 `build/Release/`。
这样可以避免程序误加载系统里较旧版本的 ONNX Runtime DLL。

## 运行

```powershell
./build/Release/melotts_cli.exe \
  --config config/english_onnx.example.ini \
  --text "Hello from MeloTTS C plus plus." \
  --output outputs/english_cpp_demo.wav
```

句级流式 demo：

```powershell
./build/Release/melotts_stream_cli.exe \
  --config config/english_onnx.example.ini \
  --text "Hello. This is a streaming C plus plus demo. It synthesizes chunk by chunk." \
  --output outputs/english_cpp_stream.wav \
  --chunk-dir outputs/stream_chunks
```

## 接口示例

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

回调式流式接口示例：

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

仓库根目录提供了一个 `Makefile`，内部仍然调用 `cmake`。

```makefile
make all
make deps
make model
make clean
make distclean
make purge
```

`make deps` 会确保 `third_party/onnxruntime-win-x64-1.20.1` 存在；如果缺失，会自动下载并解压。
`make all` 会先执行 `make deps`，并检查 `models/` 下是否已经存在必需的模型文件；如果缺失，只会补生成缺失的部分，然后完成构建。
`make model` 会先执行 `make deps`，然后检查并补生成 `models/` 目录下缺失的 ONNX 模型和 BERT 词表。
`make clean` 会递归删除所有本仓库内的临时文件和生成物，包括整个 `build` 目录、`outputs`、Python 缓存目录以及根目录级 CMake 临时文件。
`make distclean` 会在 `make clean` 的基础上额外删除 `third_party/onnxruntime-win-x64-1.20.1`。
`make purge` 会在 `make distclean` 的基础上额外删除 `models/`。

当前库目标名为 `melotts_engine`。

## Clean

如果你希望在仓库内部实现类似 `make clean` 的行为，可以直接使用 `CMake`：

1. 清理构建产物

```powershell
cmake --build build --target clean --config Release
```

2. 清理运行产生的音频输出

```powershell
cmake --build build --target distclean --config Release
```

说明：
- `clean` 由底层生成器提供，用于删除编译产物
- `distclean` 是 `CMakeLists.txt` 里自定义的目标，用于删除 `outputs`
- 如果你想完全回到未构建状态，可以直接删除 `build`
- 如果你使用 `make clean`，它会一并清掉仓库根目录里常见的 CMake 临时文件
- `make clean` 比 `cmake --build build --target clean` 更彻底，因为它会直接删除整个 `build` 目录
- 如果你需要删除仓库内下载的 ONNX Runtime 目录，请使用 `make distclean`
- 如果你还需要删除导出的 `models/` 目录，请使用 `make purge`
- 在执行过 `make distclean` 或 `make purge` 后，下一次 `make all` 或 `make model` 都会自动重新下载 ONNX Runtime
- 在执行过 `make purge` 后，下一次 `make all` 也会自动重新生成 `models/` 目录中的必需文件
- 模型检查是增量的：如果只缺少 BERT 文件或只缺少 acoustic 文件，只会重导出缺失的那一部分

## 关键实现说明

- English 推理时，`ja_bert` 输入承载 `768` 维英文 BERT 特征
- `bert` 输入填充为全零，形状为 `[1, 1024, T]`
- `phone/tone/language` 会按照 MeloTTS 的 `add_blank` 规则插空
- 默认 `english_language_id=2`，`english_tone_start=7`
- symbol 表已内置在 C++ 运行时中，不依赖外部 `symbols.py`
- `cmudict_path` 是可选项，缺失时会回退到内置英文字母发音规则
- `StreamSynthesize(...)` 提供库级回调式流式输出
- 流式分块策略为：句边界优先，其次词边界，最后才按 `max_chars` 强制切分
- 每个 `StreamingChunk` 会返回统计字段：`first_chunk_latency_ms`、`synth_ms`、`audio_ms`、`rtf`
- `melotts_stream_cli` 还会输出汇总指标：`chunks`、`samples`、`total_synth_ms`、`total_audio_ms`、`avg_rtf`

## Repository Notes

- 运行时不依赖外部 `symbols.py`
- `cmudict_path` 可选
- 构建、输出、清理都只作用在当前仓库目录内
- `tools/export_english_onnx.py` 已迁入本仓库
- 导出脚本依赖的最小 Python 模块已 vendored 到 `tools/melotts_py/`
- 第三方依赖与来源说明见 `THIRD_PARTY.md`

## Notes

- 仓库默认不附带模型权重
- 如果你的 acoustic ONNX 导出图不是直接输出音频，接口需要按实际图结构调整
- 流式 demo 是句级/分块流式，不是声码器逐帧低延迟流式
- 每个 chunk 仍然独立完成一次完整推理，chunk 之间会插入 50ms 静音，便于拼接播放
