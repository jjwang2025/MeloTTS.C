#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/**
 * @namespace melotts_engine
 * @brief Public C++ API for English-only MeloTTS ONNX inference.
 *
 * The types in this namespace are designed for a self-contained C++ runtime that
 * performs English text normalization, BERT feature generation, acoustic model
 * inference, optional chunked streaming synthesis, and PCM wave serialization.
 */
namespace melotts_engine {

/**
 * @brief Complete runtime configuration for the ONNX-based synthesis engine.
 *
 * This structure contains both file-system configuration and model-level runtime
 * defaults. The three required file paths are the English BERT ONNX model, the
 * matching BERT vocabulary file, and the exported MeloTTS acoustic ONNX model.
 *
 * The remaining fields describe ONNX input/output tensor names and default
 * synthesis parameters. They are exposed so the engine can be adapted to ONNX
 * graphs exported with different symbolic names without recompiling the C++ code.
 */
struct ModelConfig {
  /** @brief Path to the English BERT ONNX model file. */
  std::string bert_model_path;

  /** @brief Path to the BERT vocabulary text file used by the WordPiece tokenizer. */
  std::string bert_vocab_path;

  /** @brief Path to the exported MeloTTS acoustic ONNX model. */
  std::string acoustic_model_path;

  /**
   * @brief Optional path to a CMU pronunciation dictionary file.
   *
   * When this path is empty or the file does not exist, the engine falls back to
   * a built-in rule-based pronunciation approximation for unknown English words.
   */
  std::string cmudict_path;

  /** @brief BERT ONNX input tensor name for token IDs. */
  std::string bert_input_ids_name = "input_ids";

  /** @brief BERT ONNX input tensor name for the attention mask. */
  std::string bert_attention_mask_name = "attention_mask";

  /** @brief BERT ONNX input tensor name for segment IDs. */
  std::string bert_token_type_ids_name = "token_type_ids";

  /** @brief BERT ONNX output tensor name for contextual hidden states. */
  std::string bert_output_name = "last_hidden_state";

  /**
   * @brief Indicates whether the exported BERT graph expects token type IDs.
   *
   * Most standard BERT exports do expect this input, but some simplified exports
   * may omit it.
   */
  bool bert_use_token_type_ids = true;

  /** @brief Acoustic model input name for the symbol/phone ID sequence. */
  std::string acoustic_x_name = "x";

  /** @brief Acoustic model input name for the text sequence length tensor. */
  std::string acoustic_x_lengths_name = "x_lengths";

  /** @brief Acoustic model input name for the speaker ID tensor. */
  std::string acoustic_sid_name = "sid";

  /** @brief Acoustic model input name for the tone ID sequence. */
  std::string acoustic_tone_name = "tone";

  /** @brief Acoustic model input name for the language ID sequence. */
  std::string acoustic_language_name = "language";

  /** @brief Acoustic model input name for the unused 1024-channel BERT branch. */
  std::string acoustic_bert_name = "bert";

  /** @brief Acoustic model input name for the English 768-channel BERT branch. */
  std::string acoustic_ja_bert_name = "ja_bert";

  /** @brief Acoustic model input name for decoder noise scale. */
  std::string acoustic_noise_scale_name = "noise_scale";

  /** @brief Acoustic model input name for output duration scaling. */
  std::string acoustic_length_scale_name = "length_scale";

  /** @brief Acoustic model input name for stochastic duration predictor noise. */
  std::string acoustic_noise_scale_w_name = "noise_scale_w";

  /** @brief Acoustic model input name for SDP blending ratio. */
  std::string acoustic_sdp_ratio_name = "sdp_ratio";

  /** @brief Acoustic model output name for the waveform tensor. */
  std::string acoustic_output_name = "audio";

  /** @brief Output sample rate expected from the acoustic model. */
  int sample_rate = 44100;

  /** @brief Default speaker ID used when a request does not override it. */
  int64_t speaker_id = 0;

  /** @brief English language ID expected by the exported acoustic model. */
  int64_t english_language_id = 2;

  /** @brief English tone offset used by the original MeloTTS symbol pipeline. */
  int64_t english_tone_start = 7;

  /**
   * @brief Enables insertion of blank symbols between phones.
   *
   * This should match the preprocessing convention used when the acoustic model
   * was trained or exported.
   */
  bool add_blank = true;

  /** @brief Default stochastic duration predictor blending ratio. */
  float sdp_ratio = 0.2F;

  /** @brief Default decoder noise scale. */
  float noise_scale = 0.6F;

  /** @brief Default stochastic duration predictor noise scale. */
  float noise_scale_w = 0.8F;

  /**
   * @brief Default synthesis speed multiplier.
   *
   * A value larger than 1.0 makes speech faster by shrinking durations.
   */
  float speed = 1.0F;
};

/**
 * @brief Request-time overrides for a single synthesis call.
 *
 * Any numeric field left at its sentinel value keeps the default stored in the
 * corresponding @ref ModelConfig instance.
 */
struct SynthesisRequest {
  /** @brief Input text to synthesize. */
  std::string text;

  /** @brief Optional speaker override. A negative value uses the configured default. */
  int64_t speaker_id = -1;

  /** @brief Optional speed override. A negative value uses the configured default. */
  float speed = -1.0F;

  /** @brief Optional SDP ratio override. A negative value uses the configured default. */
  float sdp_ratio = -1.0F;

  /** @brief Optional decoder noise scale override. A negative value uses the configured default. */
  float noise_scale = -1.0F;

  /** @brief Optional SDP noise scale override. A negative value uses the configured default. */
  float noise_scale_w = -1.0F;
};

/**
 * @brief Controls how text is split and padded during streaming synthesis.
 */
struct StreamingOptions {
  /**
   * @brief Preferred upper bound for the number of characters in one chunk.
   *
   * The engine still tries to cut at sentence boundaries first, then at word
   * boundaries, and only falls back to a hard split when necessary.
   */
  std::size_t max_chars = 120;

  /**
   * @brief Amount of silence appended after each non-final chunk.
   *
   * This makes independently synthesized chunks easier to concatenate and play
   * back as a single stream.
   */
  int silence_ms = 50;
};

/**
 * @brief One callback payload produced by @ref TTSEngine::StreamSynthesize.
 *
 * The chunk contains the text fragment, its corresponding audio samples, and a
 * small set of timing metrics useful for latency and realtime-factor reporting.
 */
struct StreamingChunk {
  /** @brief Zero-based chunk index in the current streaming session. */
  std::size_t index = 0;

  /** @brief Total number of chunks generated for the current request. */
  std::size_t total = 0;

  /** @brief Source text assigned to this chunk. */
  std::string text;

  /**
   * @brief Chunk audio samples in normalized floating-point PCM format.
   *
   * Sample values are expected to be in the range [-1.0, 1.0]. Non-final chunks
   * may include trailing silence according to @ref StreamingOptions::silence_ms.
   */
  std::vector<float> audio;

  /** @brief Wall-clock synthesis time for this chunk in milliseconds. */
  double synth_ms = 0.0;

  /** @brief Duration of the chunk audio payload in milliseconds. */
  double audio_ms = 0.0;

  /**
   * @brief Realtime factor for this chunk.
   *
   * This is defined as `synth_ms / audio_ms`. Values below 1.0 indicate faster
   * than realtime synthesis.
   */
  double rtf = 0.0;

  /**
   * @brief End-to-end latency from stream start to delivery of this chunk.
   *
   * For the first chunk this is effectively the first-audio latency metric.
   */
  double first_chunk_latency_ms = 0.0;

  /** @brief True when this chunk is the last one in the stream. */
  bool is_last = false;
};

/**
 * @brief High-level ONNX-backed English speech synthesis engine.
 *
 * The engine owns both ONNX Runtime sessions, the English tokenizer, and the
 * text front-end required to convert normalized English text into phone, tone,
 * language, and BERT feature tensors. Instances are movable but not copyable
 * because they manage heavyweight runtime state internally.
 */
class TTSEngine {
 public:
  /**
   * @brief Constructs a synthesis engine from a fully resolved configuration.
   * @param config Runtime configuration, model paths, and synthesis defaults.
   */
  explicit TTSEngine(ModelConfig config);

  /** @brief Releases all owned runtime resources. */
  ~TTSEngine();

  /** @brief Copy construction is disabled because the engine owns runtime sessions. */
  TTSEngine(const TTSEngine&) = delete;

  /** @brief Copy assignment is disabled because the engine owns runtime sessions. */
  TTSEngine& operator=(const TTSEngine&) = delete;

  /** @brief Moves the engine and transfers ownership of all internal resources. */
  TTSEngine(TTSEngine&&) noexcept;

  /** @brief Move-assigns the engine and transfers ownership of all internal resources. */
  TTSEngine& operator=(TTSEngine&&) noexcept;

  /**
   * @brief Synthesizes a complete waveform for one request.
   *
   * This method performs text normalization, tokenization, BERT inference,
   * acoustic model inference, and returns a single floating-point PCM waveform.
   *
   * @param request Per-request synthesis options and text.
   * @return Waveform samples in the range [-1.0, 1.0].
   */
  std::vector<float> Synthesize(const SynthesisRequest& request);

  /**
   * @brief Synthesizes a request chunk by chunk and reports each chunk through a callback.
   *
   * The engine splits the input text using a sentence-first heuristic, synthesizes
   * each chunk independently, optionally appends trailing silence to non-final
   * chunks, and invokes the callback in synthesis order.
   *
   * @param request Full request containing the input text and optional overrides.
   * @param options Chunking and silence settings for the stream.
   * @param on_chunk Callback invoked once for each synthesized chunk.
   */
  void StreamSynthesize(const SynthesisRequest& request,
                        const StreamingOptions& options,
                        const std::function<void(const StreamingChunk&)>& on_chunk);

  /**
   * @brief Returns the output sample rate configured for the engine.
   * @return Output sample rate in Hz.
   */
  int sample_rate() const;

  /**
   * @brief Loads a runtime configuration from an INI-style text file.
   *
   * Relative paths in the file are resolved against the directory containing the
   * configuration file.
   *
   * @param config_path Path to the configuration file.
   * @return Parsed and normalized runtime configuration.
   */
  static ModelConfig LoadConfig(const std::string& config_path);

 private:
  class Impl;
  Impl* impl_;
};

/**
 * @brief Writes floating-point PCM samples to a mono 16-bit WAV file.
 *
 * Input samples are clamped to [-1.0, 1.0] before conversion to signed 16-bit
 * PCM.
 *
 * @param output_path Destination WAV file path.
 * @param audio Normalized floating-point PCM samples.
 * @param sample_rate Output sample rate in Hz.
 * @return True when the file was written successfully, otherwise false.
 */
bool WriteWaveFile(const std::string& output_path, const std::vector<float>& audio, int sample_rate);

}  // namespace melotts_engine
