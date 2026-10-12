#include "blingfire_wrap.h"

#include <algorithm>

#include "blingfiretools/blingfiretokdll/blingfiretokdll.h"

namespace lk_blingfire {

namespace {

// Output is at most 2x the input including BlingFire's trailing NUL: tokens
// re-encode to their exact input bytes (invalid utf-8 is rejected outright)
// and separators are bounded by the token count, itself bounded by the
// input length.
constexpr int kCapFactor = 2;

// Convert BlingFire's byte offsets (flattened [start0, end0, start1, ...],
// ends inclusive) into codepoint or UTF-16 offsets, ends exclusive.
std::vector<int> byte_offs_to_units(const std::vector<int> &byte_offs, const std::string &utf8,
                                    Offsets offsets) {
  std::vector<int> unit_offs;
  unit_offs.reserve(byte_offs.size());

  int unit_idx = 0;
  bool is_end = false;
  for (int byte_i = 0, n = static_cast<int>(utf8.size());
       byte_i < n && unit_offs.size() < byte_offs.size(); ++byte_i) {
    unsigned char b = static_cast<unsigned char>(utf8[byte_i]);
    if ((b & 0xC0) != 0x80) {
      while (unit_offs.size() < byte_offs.size() &&
             byte_offs[unit_offs.size()] + (is_end ? 1 : 0) == byte_i) {
        unit_offs.push_back(unit_idx);
        is_end = !is_end;
      }
      // 4-byte utf-8 sequences encode codepoints >= U+10000, which take a
      // surrogate pair (2 code units) in UTF-16.
      unit_idx += (offsets == Offsets::kUtf16 && b >= 0xF0) ? 2 : 1;
    }
  }
  while (unit_offs.size() < byte_offs.size())
    unit_offs.push_back(unit_idx);

  return unit_offs;
}

std::string plain(const std::string &s, decltype(&BlingFire::TextToSentences) fn) {
  int in = static_cast<int>(s.size());
  int cap = in * kCapFactor;

  std::vector<char> buf(cap);
  int out = fn(s.c_str(), in, buf.data(), cap);
  if (out <= 0 || out > cap)
    return {};

  return std::string(buf.data(), out - 1);
}

TokenizeResult with_offsets(const std::string &s, Offsets offsets, char sep,
                            decltype(&BlingFire::TextToSentencesWithOffsets) fn) {
  int in = static_cast<int>(s.size());
  int cap = in * kCapFactor;

  std::vector<char> buf(cap);
  std::vector<int> start(cap), end(cap);

  int out = fn(s.c_str(), in, buf.data(), start.data(), end.data(), cap);
  if (out <= 0 || out > cap)
    return {};

  TokenizeResult res;
  res.text.assign(buf.data(), out - 1);
  int tokens = 1 + static_cast<int>(std::count(res.text.begin(), res.text.end(), sep));

  std::vector<int> byte_offs;
  byte_offs.reserve(tokens * 2);
  for (int i = 0; i < tokens; ++i) {
    byte_offs.push_back(start[i]);
    byte_offs.push_back(end[i]);
  }

  auto units = byte_offs_to_units(byte_offs, s, offsets);

  res.spans.reserve(tokens);
  for (int i = 0; i < tokens; ++i)
    res.spans.emplace_back(units[2 * i], units[2 * i + 1]);

  return res;
}

}  // namespace

std::string text_to_sentences(const std::string &s) {
  return plain(s, &BlingFire::TextToSentences);
}

std::string text_to_words(const std::string &s) {
  return plain(s, &BlingFire::TextToWords);
}

TokenizeResult text_to_sentences_with_offsets(const std::string &s, Offsets offsets) {
  return with_offsets(s, offsets, '\n', &BlingFire::TextToSentencesWithOffsets);
}

TokenizeResult text_to_words_with_offsets(const std::string &s, Offsets offsets) {
  return with_offsets(s, offsets, ' ', &BlingFire::TextToWordsWithOffsets);
}

}  // namespace lk_blingfire
