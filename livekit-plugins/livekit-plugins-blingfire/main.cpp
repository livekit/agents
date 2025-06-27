#include "blingfiretools/blingfiretokdll/blingfiretokdll.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace BlingFire;

std::string text_to_sentences(const std::string &input) {
  int in_size = int(input.size());
  int max_out = in_size * 4 + 1;
  std::vector<char> out_buf(max_out);
  int out_len =
      TextToSentences(input.c_str(), in_size, out_buf.data(), max_out);
  return std::string(out_buf.data(), out_len);
}

std::tuple<std::string, std::vector<int>, std::vector<int>>
text_to_sentences_with_offsets(const std::string &input) {
  int in_size = int(input.size());
  int max_out = in_size * 4 + 1;
  std::vector<char> out_buf(max_out);
  // worst‐case each char, one sentence
  std::vector<int> starts(in_size);
  std::vector<int> ends(in_size);

  int count = TextToSentencesWithOffsets(input.c_str(), in_size, out_buf.data(),
                                         starts.data(), ends.data(), max_out);

  std::string out_str(out_buf.data(), /*len=*/int(strlen(out_buf.data())));
  starts.resize(count);
  ends.resize(count);
  return {out_str, starts, ends};
}

std::string text_to_words(const std::string &input) {
  int in_size = int(input.size());
  int max_out = in_size * 4 + 1;
  std::vector<char> out_buf(max_out);
  int out_len = TextToWords(input.c_str(), in_size, out_buf.data(), max_out);
  return std::string(out_buf.data(), out_len);
}

std::tuple<std::string, std::vector<int>, std::vector<int>>
text_to_words_with_offsets(const std::string &input) {
  int in_size = int(input.size());
  int max_out = in_size * 4 + 1;
  std::vector<char> out_buf(max_out);
  std::vector<int> starts(in_size);
  std::vector<int> ends(in_size);

  int count = TextToWordsWithOffsets(input.c_str(), in_size, out_buf.data(),
                                     starts.data(), ends.data(), max_out);

  std::string out_str(out_buf.data(), /*len=*/int(strlen(out_buf.data())));
  starts.resize(count);
  ends.resize(count);
  return {out_str, starts, ends};
}

PYBIND11_MODULE(livekit_blingfire, m) {
  m.doc() = "BlingFire tokenization bindings";

  m.def("text_to_sentences", &text_to_sentences,
        "Split UTF-8 text into sentences (returns a single string with "
        "delimiters)");

  m.def("text_to_sentences_with_offsets", &text_to_sentences_with_offsets,
        "Split text into sentences and return a tuple "
        "(sentences_str, start_offsets, end_offsets)");

  m.def(
      "text_to_words", &text_to_words,
      "Split UTF-8 text into words (returns a single string with delimiters)");

  m.def("text_to_words_with_offsets", &text_to_words_with_offsets,
        "Split text into words and return a tuple "
        "(words_str, start_offsets, end_offsets) ");
}
