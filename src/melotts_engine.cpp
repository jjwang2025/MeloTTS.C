#include "melotts_engine.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace melotts_engine {
namespace {

/** @brief Hidden size of the exported English BERT output. */
constexpr int64_t kEnglishBertHidden = 768;

/** @brief Size of the unused auxiliary BERT branch expected by the acoustic model. */
constexpr int64_t kUnusedBertHidden = 1024;

/** @brief Symbol used as the blank/padding item in the MeloTTS symbol table. */
constexpr char kPadSymbol[] = "_";

/** @brief Fallback symbol used when no known phone mapping is available. */
constexpr char kUnknownSymbol[] = "UNK";

/**
 * @brief Returns the built-in MeloTTS symbol inventory.
 *
 * The runtime embeds the symbol table directly so inference does not depend on
 * external Python text resources.
 */
const std::vector<std::string>& BuiltinSymbols() {
  static const std::vector<std::string> symbols = {
      "_", "\"", "(", ")", "*", "/", ":", "AA", "E", "EE", "En", "N", "OO", "Q", "V", "[", "\\",
      "]", "^", "a", "a:", "aa", "ae", "ah", "ai", "an", "ang", "ao", "aw", "ay", "b", "by", "c",
      "ch", "d", "dh", "dy", "e", "e:", "eh", "ei", "en", "eng", "er", "ey", "f", "g", "gy", "h",
      "hh", "hy", "i", "i0", "i:", "ia", "ian", "iang", "iao", "ie", "ih", "in", "ing", "iong", "ir",
      "iu", "iy", "j", "jh", "k", "ky", "l", "m", "my", "n", "ng", "ny", "o", "o:", "ong", "ou",
      "ow", "oy", "p", "py", "q", "r", "ry", "s", "sh", "t", "th", "ts", "ty", "u", "u:", "ua",
      "uai", "uan", "uang", "uh", "ui", "un", "uo", "uw", "v", "van", "ve", "vn", "w", "x", "y",
      "z", "zh", "zy", "~", "\xC3\xA6", "\xC3\xA7", "\xC3\xB0", "\xC3\xB8", "\xC5\x8B", "\xC5\x93",
      "\xC9\x90", "\xC9\x91", "\xC9\x92", "\xC9\x94", "\xC9\x95", "\xC9\x99", "\xC9\x9B", "\xC9\x9C",
      "\xC9\xA1", "\xC9\xA3", "\xC9\xA5", "\xC9\xA6", "\xC9\xAA", "\xC9\xAB", "\xC9\xAC", "\xC9\xAD",
      "\xC9\xAF", "\xC9\xB2", "\xC9\xB5", "\xC9\xB8", "\xC9\xB9", "\xC9\xBE", "\xCA\x81", "\xCA\x83",
      "\xCA\x8A", "\xCA\x8C", "\xCA\x8E", "\xCA\x8F", "\xCA\x91", "\xCA\x92", "\xCA\x9D", "\xCA\xB2",
      "\xCB\x88", "\xCB\x8C", "\xCB\x90", "\xCC\x83", "\xCC\xA9", "\xCE\xB2", "\xCE\xB8", "\xE1\x84\x80",
      "\xE1\x84\x81", "\xE1\x84\x82", "\xE1\x84\x83", "\xE1\x84\x84", "\xE1\x84\x85", "\xE1\x84\x86",
      "\xE1\x84\x87", "\xE1\x84\x88", "\xE1\x84\x89", "\xE1\x84\x8A", "\xE1\x84\x8B", "\xE1\x84\x8C",
      "\xE1\x84\x8D", "\xE1\x84\x8E", "\xE1\x84\x8F", "\xE1\x84\x90", "\xE1\x84\x91", "\xE1\x84\x92",
      "\xE1\x85\xA1", "\xE1\x85\xA2", "\xE1\x85\xA3", "\xE1\x85\xA4", "\xE1\x85\xA5", "\xE1\x85\xA6",
      "\xE1\x85\xA7", "\xE1\x85\xA8", "\xE1\x85\xA9", "\xE1\x85\xAA", "\xE1\x85\xAB", "\xE1\x85\xAC",
      "\xE1\x85\xAD", "\xE1\x85\xAE", "\xE1\x85\xAF", "\xE1\x85\xB0", "\xE1\x85\xB1", "\xE1\x85\xB2",
      "\xE1\x85\xB3", "\xE1\x85\xB4", "\xE1\x85\xB5", "\xE1\x86\xA8", "\xE1\x86\xAB", "\xE1\x86\xAE",
      "\xE1\x86\xAF", "\xE1\x86\xB7", "\xE1\x86\xB8", "\xE1\x86\xBC", "\xE3\x84\xB8", "!", "?",
      "\xE2\x80\xA6", ",", ".", "'", "-", "\xC2\xBF", "\xC2\xA1", "SP", "UNK"};
  return symbols;
}

/**
 * @brief Trims leading and trailing ASCII whitespace from a string.
 * @param value Input string.
 * @return Trimmed string.
 */
std::string Trim(const std::string& value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

/**
 * @brief Converts ASCII letters in a string to lowercase.
 * @param value Input string.
 * @return Lowercased copy of the input.
 */
std::string ToLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

/**
 * @brief Converts ASCII letters in a string to uppercase.
 * @param value Input string.
 * @return Uppercased copy of the input.
 */
std::string ToUpperAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::toupper(ch));
  });
  return value;
}

/**
 * @brief Replaces all occurrences of one substring with another.
 * @param text String to mutate in place.
 * @param from Substring to replace.
 * @param to Replacement substring.
 */
void ReplaceAll(std::string& text, const std::string& from, const std::string& to) {
  if (from.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

/**
 * @brief Resolves an optionally relative path against a base directory.
 * @param base_dir Base directory used for relative paths.
 * @param raw_path Path string from configuration.
 * @return Normalized absolute-or-relative path string.
 */
std::string ResolvePath(const std::filesystem::path& base_dir, const std::string& raw_path) {
  if (raw_path.empty()) {
    return {};
  }
  const std::filesystem::path path(raw_path);
  if (path.is_absolute()) {
    return path.lexically_normal().string();
  }
  return (base_dir / path).lexically_normal().string();
}

/**
 * @brief Loads a simple INI-style key-value file.
 *
 * The parser ignores blank lines and `#`/`;` comments. Sections are not used;
 * only `key=value` pairs are collected.
 *
 * @param config_path Path to the configuration file.
 * @return Map of parsed key-value pairs.
 */
std::unordered_map<std::string, std::string> LoadIni(const std::string& config_path) {
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("Failed to open config file: " + config_path);
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment_pos = line.find_first_of("#;");
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    const auto pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    const auto key = Trim(line.substr(0, pos));
    const auto value = Trim(line.substr(pos + 1));
    if (!key.empty()) {
      values[key] = value;
    }
  }
  return values;
}

bool ParseBool(const std::string& value) {
  const auto normalized = ToLowerAscii(Trim(value));
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

int ParseInt(const std::unordered_map<std::string, std::string>& values, const std::string& key, int fallback) {
  const auto it = values.find(key);
  return it == values.end() ? fallback : std::stoi(it->second);
}

int64_t ParseInt64(const std::unordered_map<std::string, std::string>& values, const std::string& key, int64_t fallback) {
  const auto it = values.find(key);
  return it == values.end() ? fallback : std::stoll(it->second);
}

float ParseFloat(const std::unordered_map<std::string, std::string>& values, const std::string& key, float fallback) {
  const auto it = values.find(key);
  return it == values.end() ? fallback : std::stof(it->second);
}

std::string ParseString(const std::unordered_map<std::string, std::string>& values,
                        const std::string& key,
                        const std::string& fallback = {}) {
  const auto it = values.find(key);
  return it == values.end() ? fallback : it->second;
}

std::vector<std::string> BasicBertTokenize(const std::string& text) {
  std::vector<std::string> tokens;
  std::string current;

  auto flush = [&]() {
    if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
    }
  };

  for (char ch : text) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isspace(uch)) {
      flush();
      continue;
    }
    if (std::isalnum(uch)) {
      current.push_back(static_cast<char>(std::tolower(uch)));
      continue;
    }
    flush();
    tokens.emplace_back(1, ch);
  }

  flush();
  return tokens;
}

class WordPieceTokenizer {
 public:
  /**
   * @brief Loads a BERT WordPiece vocabulary from disk.
   * @param vocab_path Path to the BERT vocabulary text file.
   */
  explicit WordPieceTokenizer(const std::string& vocab_path) {
    std::ifstream input(vocab_path);
    if (!input) {
      throw std::runtime_error("Failed to open BERT vocab: " + vocab_path);
    }

    std::string line;
    int64_t index = 0;
    while (std::getline(input, line)) {
      line = Trim(line);
      if (!line.empty()) {
        vocab_[line] = index;
      }
      ++index;
    }

    cls_id_ = RequireId("[CLS]");
    sep_id_ = RequireId("[SEP]");
    unk_id_ = RequireId("[UNK]");
  }

  /**
   * @brief Splits a token into BERT WordPiece units.
   * @param token Input token.
   * @return WordPiece token sequence or `[UNK]` when no decomposition exists.
   */
  std::vector<std::string> TokenizeWord(const std::string& token) const {
    if (token.empty()) {
      return {};
    }

    if (vocab_.count(token) > 0) {
      return {token};
    }

    std::vector<std::string> pieces;
    size_t start = 0;
    while (start < token.size()) {
      size_t end = token.size();
      bool found = false;
      std::string current_piece;
      while (end > start) {
        std::string piece = token.substr(start, end - start);
        if (start > 0) {
          piece = "##" + piece;
        }
        if (vocab_.count(piece) > 0) {
          current_piece = std::move(piece);
          found = true;
          break;
        }
        --end;
      }

      if (!found) {
        return {"[UNK]"};
      }

      pieces.push_back(current_piece);
      start = end;
    }

    return pieces;
  }

  /**
   * @brief Converts token pieces to vocabulary IDs and wraps them with special tokens.
   * @param pieces WordPiece token sequence without special tokens.
   * @return Encoded token IDs including `[CLS]` and `[SEP]`.
   */
  std::vector<int64_t> EncodeWithSpecialTokens(const std::vector<std::string>& pieces) const {
    std::vector<int64_t> ids;
    ids.reserve(pieces.size() + 2);
    ids.push_back(cls_id_);
    for (const auto& piece : pieces) {
      const auto it = vocab_.find(piece);
      ids.push_back(it == vocab_.end() ? unk_id_ : it->second);
    }
    ids.push_back(sep_id_);
    return ids;
  }

 private:
  int64_t RequireId(const std::string& token) const {
    const auto it = vocab_.find(token);
    if (it == vocab_.end()) {
      throw std::runtime_error("Required token missing in vocab: " + token);
    }
    return it->second;
  }

  std::unordered_map<std::string, int64_t> vocab_;
  int64_t cls_id_ = 0;
  int64_t sep_id_ = 0;
  int64_t unk_id_ = 0;
};

std::vector<int> DistributePhoneCounts(int phone_count, int piece_count) {
  if (piece_count <= 0) {
    return {};
  }
  std::vector<int> counts(static_cast<size_t>(piece_count), 0);
  for (int i = 0; i < phone_count; ++i) {
    const auto it = std::min_element(counts.begin(), counts.end());
    ++(*it);
  }
  return counts;
}

std::string ExpandNumbersBasic(const std::string& text) {
  static const std::vector<std::string> units = {
      "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
      "ten",  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
      "eighteen", "nineteen"};
  static const std::vector<std::string> tens = {
      "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

  std::function<std::string(int)> number_to_words = [&](int value) -> std::string {
    if (value < 20) {
      return units[static_cast<size_t>(value)];
    }
    if (value < 100) {
      const int ten = value / 10;
      const int rest = value % 10;
      return rest == 0 ? tens[static_cast<size_t>(ten)]
                       : tens[static_cast<size_t>(ten)] + " " + units[static_cast<size_t>(rest)];
    }
    if (value < 1000) {
      const int hundred = value / 100;
      const int rest = value % 100;
      return rest == 0 ? units[static_cast<size_t>(hundred)] + " hundred"
                       : units[static_cast<size_t>(hundred)] + " hundred " + number_to_words(rest);
    }
    if (value < 10000) {
      const int thousand = value / 1000;
      const int rest = value % 1000;
      return rest == 0 ? units[static_cast<size_t>(thousand)] + " thousand"
                       : units[static_cast<size_t>(thousand)] + " thousand " + number_to_words(rest);
    }
    return std::to_string(value);
  };

  std::ostringstream output;
  std::string digits;
  for (size_t i = 0; i < text.size(); ++i) {
    const char ch = text[i];
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      digits.push_back(ch);
      continue;
    }
    if (!digits.empty()) {
      output << number_to_words(std::stoi(digits));
      digits.clear();
    }
    output << ch;
  }
  if (!digits.empty()) {
    output << number_to_words(std::stoi(digits));
  }
  return output.str();
}

std::string NormalizeEnglishText(std::string text) {
  text = ToLowerAscii(text);
  ReplaceAll(text, "melotts", "melo tee tee ess");
  ReplaceAll(text, "tts", "tee tee ess");
  ReplaceAll(text, "asr", "ay ess ar");
  ReplaceAll(text, "nlp", "en el pee");
  ReplaceAll(text, "cli", "see el eye");
  ReplaceAll(text, "api", "ay pee eye");
  ReplaceAll(text, "json", "jay son");
  ReplaceAll(text, "html", "aych tee em el");
  ReplaceAll(text, "sql", "ess cue el");
  ReplaceAll(text, "c++", "see plus plus");
  ReplaceAll(text, "cpp", "see plus plus");
  ReplaceAll(text, "c#", "see sharp");
  ReplaceAll(text, "f#", "eff sharp");
  ReplaceAll(text, "mrs.", "misess");
  ReplaceAll(text, "mr.", "mister");
  ReplaceAll(text, "dr.", "doctor");
  ReplaceAll(text, "st.", "saint");
  ReplaceAll(text, "co.", "company");
  ReplaceAll(text, "jr.", "junior");
  ReplaceAll(text, "maj.", "major");
  ReplaceAll(text, "gen.", "general");
  ReplaceAll(text, "drs.", "doctors");
  ReplaceAll(text, "rev.", "reverend");
  ReplaceAll(text, "lt.", "lieutenant");
  ReplaceAll(text, "hon.", "honorable");
  ReplaceAll(text, "sgt.", "sergeant");
  ReplaceAll(text, "capt.", "captain");
  ReplaceAll(text, "esq.", "esquire");
  ReplaceAll(text, "ltd.", "limited");
  ReplaceAll(text, "col.", "colonel");
  ReplaceAll(text, "ft.", "fort");
  return ExpandNumbersBasic(text);
}

bool IsPunctuationToken(const std::string& token) {
  static const std::unordered_map<std::string, std::string> punctuation = {
      {"!", "!"}, {"?", "?"}, {",", ","}, {".", "."}, {"'", "'"}, {"-", "-"}, {":", ":"}, {";", ";"}};
  return punctuation.count(token) > 0;
}

std::vector<std::pair<std::string, int>> LookupLetterNamePhones(const std::string& token) {
  if (token.size() != 1 || !std::isalpha(static_cast<unsigned char>(token.front()))) {
    return {};
  }

  switch (static_cast<char>(std::tolower(static_cast<unsigned char>(token.front())))) {
    case 'a':
      return {{"ey", 0}};
    case 'b':
      return {{"b", 0}, {"iy", 0}};
    case 'c':
      return {{"s", 0}, {"iy", 0}};
    case 'd':
      return {{"d", 0}, {"iy", 0}};
    case 'e':
      return {{"iy", 0}};
    case 'f':
      return {{"eh", 0}, {"f", 0}};
    case 'g':
      return {{"jh", 0}, {"iy", 0}};
    case 'h':
      return {{"ey", 0}, {"ch", 0}};
    case 'i':
      return {{"ay", 0}};
    case 'j':
      return {{"jh", 0}, {"ey", 0}};
    case 'k':
      return {{"k", 0}, {"ey", 0}};
    case 'l':
      return {{"eh", 0}, {"l", 0}};
    case 'm':
      return {{"eh", 0}, {"m", 0}};
    case 'n':
      return {{"eh", 0}, {"n", 0}};
    case 'o':
      return {{"ow", 0}};
    case 'p':
      return {{"p", 0}, {"iy", 0}};
    case 'q':
      return {{"k", 0}, {"y", 0}, {"uw", 0}};
    case 'r':
      return {{"aa", 0}, {"r", 0}};
    case 's':
      return {{"eh", 0}, {"s", 0}};
    case 't':
      return {{"t", 0}, {"iy", 0}};
    case 'u':
      return {{"y", 0}, {"uw", 0}};
    case 'v':
      return {{"V", 0}, {"iy", 0}};
    case 'w':
      return {{"d", 0}, {"ah", 0}, {"b", 0}, {"ah", 0}, {"l", 0}, {"y", 0}, {"uw", 0}};
    case 'x':
      return {{"eh", 0}, {"k", 0}, {"s", 0}};
    case 'y':
      return {{"w", 0}, {"ay", 0}};
    case 'z':
      return {{"z", 0}, {"iy", 0}};
    default:
      return {};
  }
}

std::vector<std::pair<std::string, int>> LookupSpecialTokenPhones(const std::string& token) {
  if (const auto letter_name = LookupLetterNamePhones(token); !letter_name.empty()) {
    return letter_name;
  }

  const auto normalized = ToLowerAscii(token);
  if (normalized == "api") {
    return {{"ey", 0}, {"p", 0}, {"iy", 0}, {"ay", 0}};
  }
  if (normalized == "sql") {
    return {{"eh", 0}, {"s", 0}, {"k", 0}, {"y", 0}, {"uw", 0}, {"eh", 0}, {"l", 0}};
  }
  if (normalized == "html") {
    return {{"ey", 0}, {"ch", 0}, {"t", 0}, {"iy", 0}, {"eh", 0}, {"m", 0}, {"eh", 0}, {"l", 0}};
  }
  if (normalized == "tee") {
    return {{"t", 0}, {"iy", 0}};
  }
  if (normalized == "ess") {
    return {{"eh", 0}, {"s", 0}};
  }
  if (normalized == "see") {
    return {{"s", 0}, {"iy", 0}};
  }
  if (normalized == "ay") {
    return {{"ay", 0}};
  }
  if (normalized == "eff") {
    return {{"eh", 0}, {"f", 0}};
  }
  if (normalized == "ar") {
    return {{"aa", 0}, {"r", 0}};
  }
  if (normalized == "en") {
    return {{"eh", 0}, {"n", 0}};
  }
  if (normalized == "el") {
    return {{"eh", 0}, {"l", 0}};
  }
  if (normalized == "pee") {
    return {{"p", 0}, {"iy", 0}};
  }
  if (normalized == "cue") {
    return {{"k", 0}, {"y", 0}, {"uw", 0}};
  }
  if (normalized == "eye") {
    return {{"ay", 0}};
  }
  if (normalized == "jay") {
    return {{"jh", 0}, {"ey", 0}};
  }
  if (normalized == "son") {
    return {{"s", 0}, {"ah", 0}, {"n", 0}};
  }
  if (normalized == "aych") {
    return {{"ey", 0}, {"ch", 0}};
  }
  if (normalized == "em") {
    return {{"eh", 0}, {"m", 0}};
  }
  if (normalized == "plus") {
    return {{"p", 0}, {"l", 0}, {"ah", 0}, {"s", 0}};
  }
  if (normalized == "sharp") {
    return {{"sh", 0}, {"aa", 0}, {"r", 0}, {"p", 0}};
  }
  if (normalized == "melo") {
    return {{"m", 0}, {"eh", 0}, {"l", 0}, {"ow", 0}};
  }
  return {};
}

std::string MapPunctuationPhone(const std::string& token) {
  if (token == ":" || token == ";") {
    return ",";
  }
  if (token == "...") {
    return ".";
  }
  return token;
}

std::pair<std::string, int> RefineArpaPhone(const std::string& phone) {
  if (phone.empty()) {
    return {kUnknownSymbol, 0};
  }
  if (std::isdigit(static_cast<unsigned char>(phone.back()))) {
    const int tone = phone.back() - '0' + 1;
    return {ToLowerAscii(phone.substr(0, phone.size() - 1)), tone};
  }
  return {ToLowerAscii(phone), 0};
}

class CmuDictionary {
 public:
  explicit CmuDictionary(const std::string& path) {
    if (path.empty()) {
      return;
    }
    std::ifstream input(path);
    if (!input) {
      return;
    }

    std::string line;
    int line_index = 0;
    while (std::getline(input, line)) {
      ++line_index;
      if (line_index < 49) {
        continue;
      }
      const auto divider = line.find("  ");
      if (divider == std::string::npos) {
        continue;
      }

      const auto word = line.substr(0, divider);
      const auto pronunciation = line.substr(divider + 2);
      if (entries_.count(word) > 0) {
        continue;
      }

      std::vector<std::pair<std::string, int>> phones;
      std::stringstream syllables(pronunciation);
      std::string item;
      while (std::getline(syllables, item, ' ')) {
        if (item == "-" || item.empty()) {
          continue;
        }
        phones.push_back(RefineArpaPhone(item));
      }
      if (!phones.empty()) {
        entries_[word] = std::move(phones);
      }
    }
  }

  const std::vector<std::pair<std::string, int>>* Lookup(const std::string& word) const {
    const auto upper = ToUpperAscii(word);
    const auto it = entries_.find(upper);
    return it == entries_.end() ? nullptr : &it->second;
  }

 private:
  std::unordered_map<std::string, std::vector<std::pair<std::string, int>>> entries_;
};

class G2PLexicon {
 public:
  explicit G2PLexicon(const std::string& path) {
    if (path.empty()) {
      return;
    }
    std::ifstream input(path);
    if (!input) {
      return;
    }

    std::string line;
    while (std::getline(input, line)) {
      line = Trim(line);
      if (line.empty() || line[0] == '#') {
        continue;
      }

      const auto tab_pos = line.find('\t');
      const auto split_pos = tab_pos != std::string::npos ? tab_pos : line.find(' ');
      if (split_pos == std::string::npos) {
        continue;
      }

      const auto word = ToUpperAscii(Trim(line.substr(0, split_pos)));
      std::string phones_part = Trim(line.substr(split_pos + 1));
      if (word.empty() || phones_part.empty()) {
        continue;
      }

      std::vector<std::pair<std::string, int>> phones;
      std::stringstream stream(phones_part);
      std::string item;
      while (std::getline(stream, item, ' ')) {
        item = Trim(item);
        if (item.empty()) {
          continue;
        }
        phones.push_back(RefineArpaPhone(item));
      }
      if (!phones.empty()) {
        entries_[word] = std::move(phones);
      }
    }
  }

  const std::vector<std::pair<std::string, int>>* Lookup(const std::string& word) const {
    const auto upper = ToUpperAscii(word);
    const auto it = entries_.find(upper);
    return it == entries_.end() ? nullptr : &it->second;
  }

 private:
  std::unordered_map<std::string, std::vector<std::pair<std::string, int>>> entries_;
};

std::vector<std::pair<std::string, int>> FallbackPhones(const std::string& word) {
  static const std::unordered_map<char, std::string> letter_map = {
      {'a', "ae"}, {'b', "b"},  {'c', "k"},  {'d', "d"},  {'e', "eh"}, {'f', "f"},
      {'g', "g"},  {'h', "hh"}, {'i', "ih"}, {'j', "jh"}, {'k', "k"},  {'l', "l"},
      {'m', "m"},  {'n', "n"},  {'o', "ow"}, {'p', "p"},  {'q', "k"},  {'r', "r"},
      {'s', "s"},  {'t', "t"},  {'u', "uw"}, {'v', "V"},  {'w', "w"},  {'x', "s"},
      {'y', "y"},  {'z', "z"}};

  std::vector<std::pair<std::string, int>> result;
  for (char ch : word) {
    const auto it = letter_map.find(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    if (it != letter_map.end()) {
      result.emplace_back(it->second, 0);
    }
  }
  if (result.empty()) {
    result.emplace_back(kUnknownSymbol, 0);
  }
  return result;
}

template <typename T>
std::vector<T> Intersperse(const std::vector<T>& values, const T& blank) {
  std::vector<T> result(values.size() * 2 + 1, blank);
  for (size_t i = 0; i < values.size(); ++i) {
    result[i * 2 + 1] = values[i];
  }
  return result;
}

struct TextFeatures {
  std::string normalized_text;
  std::vector<int64_t> phones;
  std::vector<int64_t> tones;
  std::vector<int64_t> language;
  std::vector<int64_t> bert_input_ids;
  std::vector<int64_t> bert_attention_mask;
  std::vector<int64_t> bert_token_type_ids;
  std::vector<int> word2ph;
};

class SymbolTable {
 public:
  SymbolTable() {
    const auto& symbols = BuiltinSymbols();
    for (size_t i = 0; i < symbols.size(); ++i) {
      symbol_to_id_[symbols[i]] = static_cast<int64_t>(i);
    }
  }

  int64_t IdFor(std::string_view symbol) const {
    const auto it = symbol_to_id_.find(std::string(symbol));
    if (it != symbol_to_id_.end()) {
      return it->second;
    }
    return symbol_to_id_.at(kUnknownSymbol);
  }

 private:
  std::unordered_map<std::string, int64_t> symbol_to_id_;
};

TextFeatures BuildTextFeatures(const std::string& raw_text,
                               const ModelConfig& config,
                               const SymbolTable& symbols,
                               const G2PLexicon& g2p_lexicon,
                               const CmuDictionary& cmudict,
                               const WordPieceTokenizer& tokenizer) {
  TextFeatures features;
  features.normalized_text = NormalizeEnglishText(raw_text);

  const auto basic_tokens = BasicBertTokenize(features.normalized_text);
  std::vector<std::string> bert_pieces;
  std::vector<std::string> phones;
  std::vector<int64_t> tones;
  std::vector<int> word2ph = {1};

  phones.push_back(kPadSymbol);
  tones.push_back(0);

  for (const auto& token : basic_tokens) {
    if (token.empty()) {
      continue;
    }

    const auto pieces = tokenizer.TokenizeWord(token);
    bert_pieces.insert(bert_pieces.end(), pieces.begin(), pieces.end());

    std::vector<std::pair<std::string, int>> token_phones;
    if (const auto special_phones = LookupSpecialTokenPhones(token); !special_phones.empty()) {
      token_phones = special_phones;
    } else if (const auto* g2p_entry = g2p_lexicon.Lookup(token)) {
      token_phones = *g2p_entry;
    } else if (IsPunctuationToken(token)) {
      token_phones.emplace_back(MapPunctuationPhone(token), 0);
    } else if (const auto* cmudict_entry = cmudict.Lookup(token)) {
      token_phones = *cmudict_entry;
    } else {
      token_phones = FallbackPhones(token);
    }

    for (const auto& [phone, tone] : token_phones) {
      phones.push_back(phone);
      tones.push_back(tone);
    }

    const auto counts = DistributePhoneCounts(static_cast<int>(token_phones.size()), static_cast<int>(pieces.size()));
    word2ph.insert(word2ph.end(), counts.begin(), counts.end());
  }

  phones.push_back(kPadSymbol);
  tones.push_back(0);
  word2ph.push_back(1);

  features.bert_input_ids = tokenizer.EncodeWithSpecialTokens(bert_pieces);
  features.bert_attention_mask.assign(features.bert_input_ids.size(), 1);
  features.bert_token_type_ids.assign(features.bert_input_ids.size(), 0);
  if (features.bert_input_ids.size() != word2ph.size()) {
    throw std::runtime_error("BERT token count does not match word2ph count.");
  }

  features.word2ph = word2ph;
  for (auto& tone : tones) {
    tone += config.english_tone_start;
  }

  features.phones.reserve(phones.size());
  for (const auto& phone : phones) {
    features.phones.push_back(symbols.IdFor(phone));
  }
  features.tones = tones;
  features.language.assign(features.phones.size(), config.english_language_id);

  if (config.add_blank) {
    features.phones = Intersperse(features.phones, int64_t{0});
    features.tones = Intersperse(features.tones, int64_t{0});
    features.language = Intersperse(features.language, int64_t{0});
    for (auto& count : features.word2ph) {
      count *= 2;
    }
    features.word2ph.front() += 1;
  }

  return features;
}

Ort::Value MakeTensorInt64(const std::vector<int64_t>& values,
                           const std::vector<int64_t>& shape,
                           Ort::MemoryInfo& memory_info) {
  return Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(values.data()), values.size(),
                                           shape.data(), shape.size());
}

Ort::Value MakeTensorFloat(const std::vector<float>& values,
                           const std::vector<int64_t>& shape,
                           Ort::MemoryInfo& memory_info) {
  return Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(values.data()), values.size(),
                                         shape.data(), shape.size());
}

std::vector<float> FlattenBertToPhoneLevel(const float* token_embeddings,
                                           int64_t token_count,
                                           int64_t hidden_size,
                                           const std::vector<int>& word2ph) {
  if (token_count != static_cast<int64_t>(word2ph.size())) {
    throw std::runtime_error("BERT embedding token count mismatch.");
  }

  int64_t phone_count = 0;
  for (int count : word2ph) {
    phone_count += count;
  }

  std::vector<float> phone_level(static_cast<size_t>(hidden_size * phone_count), 0.0F);
  int64_t offset = 0;
  for (int64_t token_index = 0; token_index < token_count; ++token_index) {
    const float* token_vector = token_embeddings + token_index * hidden_size;
    for (int repeat = 0; repeat < word2ph[static_cast<size_t>(token_index)]; ++repeat) {
      for (int64_t hidden = 0; hidden < hidden_size; ++hidden) {
        phone_level[static_cast<size_t>(hidden * phone_count + offset)] = token_vector[hidden];
      }
      ++offset;
    }
  }
  return phone_level;
}

std::vector<float> ToOwnedFloatVector(const Ort::Value& value) {
  const auto* data = value.GetTensorData<float>();
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto count = static_cast<size_t>(info.GetElementCount());
  return std::vector<float>(data, data + count);
}

std::vector<float> ExtractAudio1D(const Ort::Value& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  const auto all = ToOwnedFloatVector(value);
  if (shape.empty()) {
    return all;
  }
  if (shape.size() == 3) {
    const int64_t samples = shape[2];
    return std::vector<float>(all.begin(), all.begin() + samples);
  }
  if (shape.size() == 2) {
    const int64_t samples = shape[1];
    return std::vector<float>(all.begin(), all.begin() + samples);
  }
  return all;
}

std::vector<const char*> ToNamePointers(const std::vector<std::string>& names) {
  std::vector<const char*> result;
  result.reserve(names.size());
  for (const auto& name : names) {
    result.push_back(name.c_str());
  }
  return result;
}

bool IsBoundaryChar(char ch) {
  switch (ch) {
    case '.':
    case '!':
    case '?':
    case ';':
    case ':':
      return true;
    default:
      return false;
  }
}

bool IsSoftBoundaryChar(char ch) {
  switch (ch) {
    case ',':
    case ' ':
    case '\t':
    case '\n':
      return true;
    default:
      return false;
  }
}

std::vector<std::string> SplitForStreaming(const std::string& text, std::size_t max_chars) {
  const std::size_t chunk_limit = max_chars == 0 ? 120 : max_chars;
  std::vector<std::string> chunks;
  std::size_t start = 0;

  while (start < text.size()) {
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
      ++start;
    }
    if (start >= text.size()) {
      break;
    }

    const std::size_t hard_end = std::min(start + chunk_limit, text.size());
    std::size_t best_end = std::string::npos;
    std::size_t soft_end = std::string::npos;

    for (std::size_t i = start; i < hard_end; ++i) {
      const char ch = text[i];
      if (IsBoundaryChar(ch)) {
        best_end = i + 1;
      } else if (IsSoftBoundaryChar(ch)) {
        soft_end = i + 1;
      }
    }

    std::size_t end = hard_end;
    if (hard_end < text.size()) {
      if (best_end != std::string::npos && best_end > start) {
        end = best_end;
      } else if (soft_end != std::string::npos && soft_end > start) {
        end = soft_end;
      }
    }

    auto chunk = Trim(text.substr(start, end - start));
    if (!chunk.empty()) {
      chunks.push_back(std::move(chunk));
    }
    start = end;
  }

  return chunks;
}

}  // namespace

class TTSEngine::Impl {
 public:
  /**
   * @brief Builds the internal runtime implementation and loads both ONNX sessions.
   * @param config Fully resolved runtime configuration.
   */
  explicit Impl(ModelConfig config)
      : config_(std::move(config)),
        env_(ORT_LOGGING_LEVEL_WARNING, "melotts_english_onnx"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        tokenizer_(config_.bert_vocab_path),
        g2p_lexicon_(config_.g2p_lexicon_path),
        cmudict_(config_.cmudict_path),
        symbols_(),
        bert_session_(nullptr),
        acoustic_session_(nullptr) {
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    bert_session_ = Ort::Session(env_, ToWide(config_.bert_model_path).c_str(), session_options_);
    acoustic_session_ = Ort::Session(env_, ToWide(config_.acoustic_model_path).c_str(), session_options_);
  }

  /**
   * @brief Synthesizes a complete waveform for a single request.
   * @param request Request text and runtime overrides.
   * @return Floating-point PCM waveform.
   */
  std::vector<float> Synthesize(const SynthesisRequest& request) {
    if (request.text.empty()) {
      throw std::runtime_error("Synthesis text must not be empty.");
    }

    const auto features = BuildTextFeatures(request.text, config_, symbols_, g2p_lexicon_, cmudict_, tokenizer_);
    const int64_t text_length = static_cast<int64_t>(features.phones.size());

    auto bert_hidden = RunBert(features);
    std::vector<float> bert_zeros(static_cast<size_t>(kUnusedBertHidden * text_length), 0.0F);

    const int64_t speaker_id = request.speaker_id >= 0 ? request.speaker_id : config_.speaker_id;
    const float speed = request.speed > 0.0F ? request.speed : config_.speed;
    const float length_scale = 1.0F / speed;
    const float sdp_ratio = request.sdp_ratio >= 0.0F ? request.sdp_ratio : config_.sdp_ratio;
    const float noise_scale = request.noise_scale >= 0.0F ? request.noise_scale : config_.noise_scale;
    const float noise_scale_w = request.noise_scale_w >= 0.0F ? request.noise_scale_w : config_.noise_scale_w;

    std::vector<int64_t> x_lengths = {text_length};
    std::vector<int64_t> sid = {speaker_id};
    std::vector<float> noise_scale_vec = {noise_scale};
    std::vector<float> length_scale_vec = {length_scale};
    std::vector<float> noise_scale_w_vec = {noise_scale_w};
    std::vector<float> sdp_ratio_vec = {sdp_ratio};

    std::vector<std::string> input_names = {
        config_.acoustic_x_name,
        config_.acoustic_x_lengths_name,
        config_.acoustic_sid_name,
        config_.acoustic_tone_name,
        config_.acoustic_language_name,
        config_.acoustic_bert_name,
        config_.acoustic_ja_bert_name,
        config_.acoustic_noise_scale_name,
        config_.acoustic_length_scale_name,
        config_.acoustic_noise_scale_w_name,
        config_.acoustic_sdp_ratio_name,
    };

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(input_names.size());
    input_tensors.push_back(MakeTensorInt64(features.phones, {1, text_length}, memory_info_));
    input_tensors.push_back(MakeTensorInt64(x_lengths, {1}, memory_info_));
    input_tensors.push_back(MakeTensorInt64(sid, {1}, memory_info_));
    input_tensors.push_back(MakeTensorInt64(features.tones, {1, text_length}, memory_info_));
    input_tensors.push_back(MakeTensorInt64(features.language, {1, text_length}, memory_info_));
    input_tensors.push_back(MakeTensorFloat(bert_zeros, {1, kUnusedBertHidden, text_length}, memory_info_));
    input_tensors.push_back(MakeTensorFloat(bert_hidden, {1, kEnglishBertHidden, text_length}, memory_info_));
    input_tensors.push_back(MakeTensorFloat(noise_scale_vec, {1}, memory_info_));
    input_tensors.push_back(MakeTensorFloat(length_scale_vec, {1}, memory_info_));
    input_tensors.push_back(MakeTensorFloat(noise_scale_w_vec, {1}, memory_info_));
    input_tensors.push_back(MakeTensorFloat(sdp_ratio_vec, {1}, memory_info_));

    std::vector<std::string> output_names = {config_.acoustic_output_name};
    const auto input_name_ptrs = ToNamePointers(input_names);
    const auto output_name_ptrs = ToNamePointers(output_names);

    auto outputs = acoustic_session_.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_tensors.data(),
                                        input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());
    if (outputs.empty()) {
      throw std::runtime_error("Acoustic ONNX session returned no outputs.");
    }
    return ExtractAudio1D(outputs.front());
  }

  /**
   * @brief Streams synthesis results chunk by chunk through a callback.
   * @param request Full synthesis request.
   * @param options Chunking and silence settings.
   * @param on_chunk Callback invoked for each produced chunk.
   */
  void StreamSynthesize(const SynthesisRequest& request,
                        const StreamingOptions& options,
                        const std::function<void(const StreamingChunk&)>& on_chunk) {
    if (!on_chunk) {
      throw std::runtime_error("Streaming callback must not be empty.");
    }

    const auto chunks = SplitForStreaming(request.text, options.max_chars);
    if (chunks.empty()) {
      throw std::runtime_error("No non-empty text chunks were produced.");
    }

    const int silence_samples = std::max(0, static_cast<int>(sample_rate() * options.silence_ms / 1000.0F));
    std::vector<float> silence(static_cast<std::size_t>(silence_samples), 0.0F);

    const auto stream_started = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < chunks.size(); ++i) {
      auto chunk_request = request;
      chunk_request.text = chunks[i];
      const auto synth_started = std::chrono::steady_clock::now();
      auto audio = Synthesize(chunk_request);
      const auto synth_finished = std::chrono::steady_clock::now();
      const auto synth_ms =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(synth_finished - synth_started)
              .count();
      const auto first_chunk_latency_ms =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(synth_finished - stream_started)
              .count();
      const auto audio_ms = static_cast<double>(audio.size()) * 1000.0 / static_cast<double>(sample_rate());
      const auto rtf = audio_ms > 0.0 ? synth_ms / audio_ms : 0.0;

      if (i + 1 < chunks.size() && !silence.empty()) {
        audio.insert(audio.end(), silence.begin(), silence.end());
      }

      StreamingChunk chunk;
      chunk.index = i;
      chunk.total = chunks.size();
      chunk.text = chunks[i];
      chunk.audio = std::move(audio);
      chunk.synth_ms = synth_ms;
      chunk.audio_ms = audio_ms;
      chunk.rtf = rtf;
      chunk.first_chunk_latency_ms = first_chunk_latency_ms;
      chunk.is_last = (i + 1 == chunks.size());
      on_chunk(chunk);
    }
  }

  int sample_rate() const { return config_.sample_rate; }

 private:
  /**
   * @brief Converts a UTF-8-ish narrow path to a wide path for the Windows ORT API.
   * @param value Narrow string path.
   * @return Wide string path.
   */
  static std::wstring ToWide(const std::string& value) {
    return std::wstring(value.begin(), value.end());
  }

  /**
   * @brief Runs the BERT ONNX model and expands token embeddings to phone resolution.
   * @param features Text front-end features for the current request.
   * @return Flattened English BERT tensor in `[1, 768, T]` layout.
   */
  std::vector<float> RunBert(const TextFeatures& features) {
    const int64_t token_count = static_cast<int64_t>(features.bert_input_ids.size());

    std::vector<std::string> input_names = {
        config_.bert_input_ids_name,
        config_.bert_attention_mask_name,
    };
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(config_.bert_use_token_type_ids ? 3 : 2);
    input_tensors.push_back(MakeTensorInt64(features.bert_input_ids, {1, token_count}, memory_info_));
    input_tensors.push_back(MakeTensorInt64(features.bert_attention_mask, {1, token_count}, memory_info_));
    if (config_.bert_use_token_type_ids) {
      input_names.push_back(config_.bert_token_type_ids_name);
      input_tensors.push_back(MakeTensorInt64(features.bert_token_type_ids, {1, token_count}, memory_info_));
    }

    std::vector<std::string> output_names = {config_.bert_output_name};
    const auto input_name_ptrs = ToNamePointers(input_names);
    const auto output_name_ptrs = ToNamePointers(output_names);

    auto outputs = bert_session_.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_tensors.data(),
                                    input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());
    if (outputs.empty()) {
      throw std::runtime_error("BERT ONNX session returned no outputs.");
    }

    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != token_count || shape[2] != kEnglishBertHidden) {
      throw std::runtime_error("Unexpected BERT output shape. Expected [1, tokens, 768].");
    }
    const auto* token_embeddings = outputs.front().GetTensorData<float>();
    return FlattenBertToPhoneLevel(token_embeddings, token_count, kEnglishBertHidden, features.word2ph);
  }

  ModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo memory_info_;
  WordPieceTokenizer tokenizer_;
  G2PLexicon g2p_lexicon_;
  CmuDictionary cmudict_;
  SymbolTable symbols_;
  Ort::Session bert_session_;
  Ort::Session acoustic_session_;
};

TTSEngine::TTSEngine(ModelConfig config) : impl_(new Impl(std::move(config))) {}

TTSEngine::~TTSEngine() { delete impl_; }

TTSEngine::TTSEngine(TTSEngine&& other) noexcept : impl_(other.impl_) { other.impl_ = nullptr; }

TTSEngine& TTSEngine::operator=(TTSEngine&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

std::vector<float> TTSEngine::Synthesize(const SynthesisRequest& request) { return impl_->Synthesize(request); }

void TTSEngine::StreamSynthesize(const SynthesisRequest& request,
                                 const StreamingOptions& options,
                                 const std::function<void(const StreamingChunk&)>& on_chunk) {
  impl_->StreamSynthesize(request, options, on_chunk);
}

int TTSEngine::sample_rate() const { return impl_->sample_rate(); }

ModelConfig TTSEngine::LoadConfig(const std::string& config_path) {
  const auto values = LoadIni(config_path);
  const auto base_dir = std::filesystem::path(config_path).parent_path();

  ModelConfig config;
  config.bert_model_path = ResolvePath(base_dir, ParseString(values, "bert_model_path"));
  config.bert_vocab_path = ResolvePath(base_dir, ParseString(values, "bert_vocab_path"));
  config.acoustic_model_path = ResolvePath(base_dir, ParseString(values, "acoustic_model_path"));
  config.g2p_lexicon_path = ResolvePath(base_dir, ParseString(values, "g2p_lexicon_path", ""));
  config.cmudict_path = ResolvePath(base_dir, ParseString(values, "cmudict_path", ""));

  config.bert_input_ids_name = ParseString(values, "bert_input_ids_name", config.bert_input_ids_name);
  config.bert_attention_mask_name = ParseString(values, "bert_attention_mask_name", config.bert_attention_mask_name);
  config.bert_token_type_ids_name = ParseString(values, "bert_token_type_ids_name", config.bert_token_type_ids_name);
  config.bert_output_name = ParseString(values, "bert_output_name", config.bert_output_name);
  config.bert_use_token_type_ids = values.count("bert_use_token_type_ids") == 0
                                       ? config.bert_use_token_type_ids
                                       : ParseBool(values.at("bert_use_token_type_ids"));

  config.acoustic_x_name = ParseString(values, "acoustic_x_name", config.acoustic_x_name);
  config.acoustic_x_lengths_name = ParseString(values, "acoustic_x_lengths_name", config.acoustic_x_lengths_name);
  config.acoustic_sid_name = ParseString(values, "acoustic_sid_name", config.acoustic_sid_name);
  config.acoustic_tone_name = ParseString(values, "acoustic_tone_name", config.acoustic_tone_name);
  config.acoustic_language_name = ParseString(values, "acoustic_language_name", config.acoustic_language_name);
  config.acoustic_bert_name = ParseString(values, "acoustic_bert_name", config.acoustic_bert_name);
  config.acoustic_ja_bert_name = ParseString(values, "acoustic_ja_bert_name", config.acoustic_ja_bert_name);
  config.acoustic_noise_scale_name = ParseString(values, "acoustic_noise_scale_name", config.acoustic_noise_scale_name);
  config.acoustic_length_scale_name = ParseString(values, "acoustic_length_scale_name", config.acoustic_length_scale_name);
  config.acoustic_noise_scale_w_name = ParseString(values, "acoustic_noise_scale_w_name", config.acoustic_noise_scale_w_name);
  config.acoustic_sdp_ratio_name = ParseString(values, "acoustic_sdp_ratio_name", config.acoustic_sdp_ratio_name);
  config.acoustic_output_name = ParseString(values, "acoustic_output_name", config.acoustic_output_name);

  config.sample_rate = ParseInt(values, "sample_rate", config.sample_rate);
  config.speaker_id = ParseInt64(values, "speaker_id", config.speaker_id);
  config.english_language_id = ParseInt64(values, "english_language_id", config.english_language_id);
  config.english_tone_start = ParseInt64(values, "english_tone_start", config.english_tone_start);
  config.add_blank = values.count("add_blank") == 0 ? config.add_blank : ParseBool(values.at("add_blank"));
  config.sdp_ratio = ParseFloat(values, "sdp_ratio", config.sdp_ratio);
  config.noise_scale = ParseFloat(values, "noise_scale", config.noise_scale);
  config.noise_scale_w = ParseFloat(values, "noise_scale_w", config.noise_scale_w);
  config.speed = ParseFloat(values, "speed", config.speed);

  if (config.bert_model_path.empty() || config.bert_vocab_path.empty() || config.acoustic_model_path.empty()) {
    throw std::runtime_error("Config must define bert_model_path, bert_vocab_path, and acoustic_model_path.");
  }

  return config;
}

bool WriteWaveFile(const std::string& output_path, const std::vector<float>& audio, int sample_rate) {
  std::ofstream output(output_path, std::ios::binary);
  if (!output) {
    return false;
  }

  const uint16_t channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
  const uint16_t block_align = channels * bits_per_sample / 8;
  const uint32_t data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
  const uint32_t chunk_size = 36U + data_size;

  output.write("RIFF", 4);
  output.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
  output.write("WAVE", 4);
  output.write("fmt ", 4);

  const uint32_t subchunk1_size = 16;
  const uint16_t audio_format = 1;
  output.write(reinterpret_cast<const char*>(&subchunk1_size), sizeof(subchunk1_size));
  output.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
  output.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
  output.write(reinterpret_cast<const char*>(&sample_rate), sizeof(sample_rate));
  output.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
  output.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
  output.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));

  output.write("data", 4);
  output.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

  for (float sample : audio) {
    const float clamped = std::max(-1.0F, std::min(1.0F, sample));
    const auto pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0F));
    output.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
  }

  return output.good();
}

}  // namespace melotts_engine
