// Binding-agnostic wrapper around BlingFire's sentence / word breaking.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace lk_blingfire {

struct TokenizeResult {
  // '\n'-separated sentences or ' '-separated words.
  std::string text;
  // Per-token (start, end) offsets into the original input, in the
  // requested `Offsets` unit; `end` is exclusive.
  std::vector<std::pair<int, int>> spans;
};

enum class Offsets {
  kCodepoints,  // Unicode codepoints (Python str indexing)
  kUtf16,       // UTF-16 code units (JavaScript string indexing)
};

std::string text_to_sentences(const std::string &s);
std::string text_to_words(const std::string &s);

TokenizeResult text_to_sentences_with_offsets(const std::string &s, Offsets offsets);
TokenizeResult text_to_words_with_offsets(const std::string &s, Offsets offsets);

}  // namespace lk_blingfire
