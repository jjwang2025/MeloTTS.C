#include "melotts_engine.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Incremental WAV writer used by the streaming demo.
 *
 * The class writes a placeholder WAV header, appends PCM samples as chunks are
 * synthesized, and patches the RIFF size fields when closed.
 */
class WaveStreamWriter {
 public:
  /**
   * @brief Opens a mono 16-bit PCM WAV stream for incremental writing.
   * @param output_path Destination WAV file path.
   * @param sample_rate Output sample rate in Hz.
   */
  WaveStreamWriter(std::string output_path, int sample_rate)
      : output_path_(std::move(output_path)), sample_rate_(sample_rate) {
    stream_.open(output_path_, std::ios::binary);
    if (!stream_) {
      throw std::runtime_error("Failed to open output wav: " + output_path_);
    }
    WriteHeaderPlaceholder();
  }

  /** @brief Closes the stream and finalizes the WAV header if still open. */
  ~WaveStreamWriter() { Close(); }

  WaveStreamWriter(const WaveStreamWriter&) = delete;
  WaveStreamWriter& operator=(const WaveStreamWriter&) = delete;

  /**
   * @brief Appends normalized floating-point PCM samples to the WAV stream.
   * @param audio Audio samples in the range [-1.0, 1.0].
   */
  void AppendSamples(const std::vector<float>& audio) {
    for (float sample : audio) {
      const float clamped = std::max(-1.0F, std::min(1.0F, sample));
      const auto pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0F));
      stream_.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
      data_size_ += sizeof(pcm);
    }
  }

  /**
   * @brief Appends a number of silent samples to the WAV stream.
   * @param sample_count Number of zero-valued samples to append.
   */
  void AppendSilence(int sample_count) {
    static const int16_t silence = 0;
    for (int i = 0; i < sample_count; ++i) {
      stream_.write(reinterpret_cast<const char*>(&silence), sizeof(silence));
      data_size_ += sizeof(silence);
    }
  }

  /**
   * @brief Finalizes the WAV header and closes the output stream.
   */
  void Close() {
    if (!stream_.is_open()) {
      return;
    }

    const uint32_t chunk_size = 36U + data_size_;
    stream_.seekp(4, std::ios::beg);
    stream_.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
    stream_.seekp(40, std::ios::beg);
    stream_.write(reinterpret_cast<const char*>(&data_size_), sizeof(data_size_));
    stream_.close();
  }

 private:
  /**
   * @brief Writes an initial RIFF/WAV header with placeholder data sizes.
   */
  void WriteHeaderPlaceholder() {
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate_ * channels * bits_per_sample / 8);
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t zero = 0;
    const uint32_t subchunk1_size = 16;
    const uint16_t audio_format = 1;

    stream_.write("RIFF", 4);
    stream_.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    stream_.write("WAVE", 4);
    stream_.write("fmt ", 4);
    stream_.write(reinterpret_cast<const char*>(&subchunk1_size), sizeof(subchunk1_size));
    stream_.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    stream_.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    stream_.write(reinterpret_cast<const char*>(&sample_rate_), sizeof(sample_rate_));
    stream_.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    stream_.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    stream_.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));
    stream_.write("data", 4);
    stream_.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
  }

  std::string output_path_;
  int sample_rate_ = 0;
  std::ofstream stream_;
  uint32_t data_size_ = 0;
};

/**
 * @brief Prints command-line usage for the streaming demo executable.
 */
void PrintUsage() {
  std::cout << "Usage:\n"
            << "  melotts_tts_stream_cli --config <ini> --text <text> --output <wav>"
            << " [--speaker <id>] [--speed <value>] [--max-chars <n>] [--chunk-dir <dir>]\n";
}

/**
 * @brief Returns the value following the current command-line option.
 * @param index Current argument index. The function advances it to the value position.
 * @param argc Argument count from `main`.
 * @param argv Argument vector from `main`.
 * @return The string value associated with the current option.
 * @throws std::runtime_error Thrown when the option is missing its required value.
 */
std::string RequireValue(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for argument: ") + argv[index]);
  }
  ++index;
  return argv[index];
}

}  // namespace

/**
 * @brief Entry point for the chunked streaming synthesis demo.
 *
 * The demo uses @ref melotts_engine::TTSEngine::StreamSynthesize to emit one
 * chunk at a time, appends all chunks to a single WAV file, optionally stores
 * each chunk as an individual WAV file, and prints per-chunk as well as summary
 * performance metrics.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Zero on success, non-zero on failure.
 */
int main(int argc, char** argv) {
  try {
    std::string config_path;
    std::string text;
    std::string output_path;
    std::string chunk_dir;
    std::size_t max_chars = 120;
    melotts_engine::SynthesisRequest request;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config") {
        config_path = RequireValue(i, argc, argv);
      } else if (arg == "--text") {
        text = RequireValue(i, argc, argv);
      } else if (arg == "--output") {
        output_path = RequireValue(i, argc, argv);
      } else if (arg == "--speaker") {
        request.speaker_id = std::stoll(RequireValue(i, argc, argv));
      } else if (arg == "--speed") {
        request.speed = std::stof(RequireValue(i, argc, argv));
      } else if (arg == "--max-chars") {
        max_chars = static_cast<std::size_t>(std::stoul(RequireValue(i, argc, argv)));
      } else if (arg == "--chunk-dir") {
        chunk_dir = RequireValue(i, argc, argv);
      } else if (arg == "--help" || arg == "-h") {
        PrintUsage();
        return 0;
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }

    if (config_path.empty() || text.empty() || output_path.empty()) {
      PrintUsage();
      return 1;
    }

    auto config = melotts_engine::TTSEngine::LoadConfig(config_path);
    melotts_engine::TTSEngine engine(std::move(config));

    const auto parent = std::filesystem::path(output_path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    if (!chunk_dir.empty()) {
      std::filesystem::create_directories(chunk_dir);
    }

    WaveStreamWriter writer(output_path, engine.sample_rate());
    request.text = text;

    melotts_engine::StreamingOptions options;
    options.max_chars = max_chars;
    options.silence_ms = 50;

    bool first_chunk = true;
    std::size_t total_chunks = 0;
    std::size_t total_samples = 0;
    double total_synth_ms = 0.0;
    double total_audio_ms = 0.0;
    double first_chunk_latency_ms = 0.0;
    engine.StreamSynthesize(
        request,
        options,
        [&](const melotts_engine::StreamingChunk& chunk) {
          if (first_chunk) {
            std::cout << "Streaming " << chunk.total << " chunk(s)\n";
            first_chunk_latency_ms = chunk.first_chunk_latency_ms;
            first_chunk = false;
          }

          writer.AppendSamples(chunk.audio);

          std::vector<float> chunk_audio = chunk.audio;
          if (!chunk.is_last && options.silence_ms > 0) {
            const int trim_samples = static_cast<int>(engine.sample_rate() * options.silence_ms / 1000.0F);
            if (trim_samples > 0 && chunk_audio.size() >= static_cast<std::size_t>(trim_samples)) {
              chunk_audio.resize(chunk_audio.size() - static_cast<std::size_t>(trim_samples));
            }
          }

          if (!chunk_dir.empty()) {
            const auto chunk_path =
                (std::filesystem::path(chunk_dir) / ("chunk_" + std::to_string(chunk.index) + ".wav")).string();
            melotts_engine::WriteWaveFile(chunk_path, chunk_audio, engine.sample_rate());
          }

          ++total_chunks;
          total_samples += chunk_audio.size();
          total_synth_ms += chunk.synth_ms;
          total_audio_ms += chunk.audio_ms;

          std::cout << "Chunk " << (chunk.index + 1) << "/" << chunk.total << ": " << chunk.text << "\n";
          std::cout << "  samples=" << chunk_audio.size() << " synth_ms=" << chunk.synth_ms
                    << " audio_ms=" << chunk.audio_ms << " rtf=" << chunk.rtf
                    << " first_latency_ms=" << chunk.first_chunk_latency_ms << "\n";
        });

    writer.Close();
    const double avg_rtf = total_audio_ms > 0.0 ? total_synth_ms / total_audio_ms : 0.0;
    std::cout << "Summary: chunks=" << total_chunks << " samples=" << total_samples
              << " total_synth_ms=" << total_synth_ms << " total_audio_ms=" << total_audio_ms
              << " avg_rtf=" << avg_rtf << " first_chunk_latency_ms=" << first_chunk_latency_ms << "\n";
    std::cout << "Saved streamed wav to " << output_path << "\n";
    std::cout << "Sample rate: " << engine.sample_rate() << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
