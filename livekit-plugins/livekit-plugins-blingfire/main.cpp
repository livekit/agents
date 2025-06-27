#include "blingfiretools/blingfiretokdll/blingfiretokdll.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace BlingFire;

static std::vector<int> utf8_to_codepoints(const std::vector<int> &byte_offs,
                                           const std::string &utf8) {
  std::vector<int> cp_offs;
  cp_offs.reserve(byte_offs.size());

  int cp_idx = 0;
  bool is_end = false;
  for (int byte_i = 0, n = static_cast<int>(utf8.size()); byte_i < n;
       ++byte_i) {
    unsigned char b = static_cast<unsigned char>(utf8[byte_i]);
    if ((b & 0xC0) != 0x80) {
      while (cp_offs.size() < byte_offs.size() &&
             byte_offs[cp_offs.size()] + (is_end ? 1 : 0) == byte_i) {
        cp_offs.push_back(cp_idx);
        is_end = !is_end;
      }
      ++cp_idx;
    }
  }
  while (cp_offs.size() < byte_offs.size())
    cp_offs.push_back(cp_idx);

  return cp_offs;
}

std::string text_to_sentences(const std::string &s) {
  int in = static_cast<int>(s.size());
  int cap = in * 2;

  std::vector<char> buf(cap);
  int out = TextToSentences(s.c_str(), in, buf.data(), cap);
  if (out < 0 || out > cap)
    return {};

  return std::string(buf.data(), out - 1);
}

py::tuple text_to_sentences_with_offsets(const std::string &s) {
  int in = static_cast<int>(s.size());
  int cap = in * 2;

  std::vector<char> buf(cap);
  std::vector<int> start(cap), end(cap);

  int out = TextToSentencesWithOffsets(s.c_str(), in, buf.data(), start.data(),
                                       end.data(), cap);
  if (out < 0 || out > cap)
    return py::make_tuple(std::string(), py::list());

  std::string utf8(buf.data(), out - 1);
  int tokens = 1 + std::count(utf8.begin(), utf8.end(), '\n');

  std::vector<int> byte_offs;
  byte_offs.reserve(tokens * 2);
  for (int i = 0; i < tokens; ++i) {
    byte_offs.push_back(start[i]);
    byte_offs.push_back(end[i]);
  }

  auto cp = utf8_to_codepoints(byte_offs, s);

  std::vector<std::pair<int, int>> spans;
  spans.reserve(tokens);
  for (int i = 0; i < tokens; ++i)
    spans.emplace_back(cp[2 * i], cp[2 * i + 1]);

  return py::make_tuple(utf8, spans);
}

std::string text_to_words(const std::string &s) {
  int in = static_cast<int>(s.size());
  int cap = in * 3;

  std::vector<char> buf(cap);
  int out = TextToWords(s.c_str(), in, buf.data(), cap);
  if (out < 0 || out > cap)
    return {};

  return std::string(buf.data(), out - 1);
}

py::tuple text_to_words_with_offsets(const std::string &s) {
  int in = static_cast<int>(s.size());
  int cap = in * 2;

  std::vector<char> buf(cap);
  std::vector<int> start(cap), end(cap);

  int out = TextToWordsWithOffsets(s.c_str(), in, buf.data(), start.data(),
                                   end.data(), cap);
  if (out < 0 || out > cap)
    return py::make_tuple(std::string(), py::list());

  std::string utf8(buf.data(), out - 1);
  int tokens = 1 + std::count(utf8.begin(), utf8.end(), ' ');

  std::vector<int> byte_offs;
  byte_offs.reserve(tokens * 2);
  for (int i = 0; i < tokens; ++i) {
    byte_offs.push_back(start[i]);
    byte_offs.push_back(end[i]);
  }

  auto cp = utf8_to_codepoints(byte_offs, s);

  std::vector<std::pair<int, int>> spans;
  spans.reserve(tokens);
  for (int i = 0; i < tokens; ++i)
    spans.emplace_back(cp[2 * i], cp[2 * i + 1]);

  return py::make_tuple(utf8, spans);
}

PYBIND11_MODULE(livekit_blingfire, m) {
  m.doc() = "Exact-behaviour BlingFire bindings (matches reference ctypes)";

  m.def("text_to_sentences", &text_to_sentences,
        "TextToSentences (buffer size len*2)");
  m.def("text_to_sentences_with_offsets", &text_to_sentences_with_offsets,
        "TextToSentencesWithOffsets; returns (str, [(start,end), ...])");
  m.def("text_to_words", &text_to_words, "TextToWords (buffer size len*3)");
  m.def("text_to_words_with_offsets", &text_to_words_with_offsets,
        "TextToWordsWithOffsets; returns (str, [(start,end), ...])");
}
