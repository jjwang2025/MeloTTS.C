#include "melotts_engine.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

/**
 * @brief Prints command-line usage for the non-streaming demo executable.
 */
void PrintUsage() {
  std::cout << "Usage:\n"
            << "  melotts_tts_cli --config <ini> --text <text> --output <wav> [--speaker <id>] [--speed <value>]\n";
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
 * @brief Entry point for the basic waveform synthesis demo.
 *
 * The demo loads an engine configuration, synthesizes one text request, and
 * writes the resulting waveform to a WAV file on disk.
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
    request.text = text;

    const auto audio = engine.Synthesize(request);
    const auto parent = std::filesystem::path(output_path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    if (!melotts_engine::WriteWaveFile(output_path, audio, engine.sample_rate())) {
      throw std::runtime_error("Failed to write wav file: " + output_path);
    }

    std::cout << "Saved wav to " << output_path << "\n";
    std::cout << "Samples: " << audio.size() << "\n";
    std::cout << "Sample rate: " << engine.sample_rate() << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
