// C ABI over `lk_blingfire`, for the Rust binding.
//
// The C++ wrapper returns `std::string` and `std::vector<std::pair<int,int>>`,
// neither of which Rust can touch directly. Rather than reproduce those layouts,
// results stay owned by C++ behind an opaque handle and Rust reads through
// accessors. That keeps allocation and deallocation on one side of the boundary.

#include "blingfire_wrap.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

namespace {

struct Result {
  std::string text;
  std::vector<std::pair<int, int>> spans;
};

lk_blingfire::Offsets to_offsets(int value) {
  return value == 1 ? lk_blingfire::Offsets::kUtf16
                    : lk_blingfire::Offsets::kCodepoints;
}

// Duplicate into a malloc'd buffer so Rust frees it with `lk_bf_free_string`.
char *dup_cstr(const std::string &s) {
  char *out = static_cast<char *>(std::malloc(s.size() + 1));
  if (out == nullptr) {
    return nullptr;
  }
  std::memcpy(out, s.data(), s.size());
  out[s.size()] = '\0';
  return out;
}

}  // namespace

extern "C" {

// mode: 0 = sentences, 1 = words. Returns NULL on allocation failure.
char *lk_bf_tokenize_plain(const char *input, int mode) {
  if (input == nullptr) {
    return nullptr;
  }
  try {
    const std::string s(input);
    return dup_cstr(mode == 1 ? lk_blingfire::text_to_words(s)
                              : lk_blingfire::text_to_sentences(s));
  } catch (...) {
    return nullptr;
  }
}

void lk_bf_free_string(char *p) { std::free(p); }

// mode: 0 = sentences, 1 = words. offsets: 0 = codepoints, 1 = UTF-16.
Result *lk_bf_tokenize_with_offsets(const char *input, int mode, int offsets) {
  if (input == nullptr) {
    return nullptr;
  }
  try {
    const std::string s(input);
    auto out = mode == 1 ? lk_blingfire::text_to_words_with_offsets(s, to_offsets(offsets))
                         : lk_blingfire::text_to_sentences_with_offsets(s, to_offsets(offsets));
    auto *result = new (std::nothrow) Result{std::move(out.text), std::move(out.spans)};
    return result;
  } catch (...) {
    return nullptr;
  }
}

const char *lk_bf_result_text(const Result *r) {
  return r == nullptr ? nullptr : r->text.c_str();
}

size_t lk_bf_result_text_len(const Result *r) {
  return r == nullptr ? 0 : r->text.size();
}

size_t lk_bf_result_span_count(const Result *r) {
  return r == nullptr ? 0 : r->spans.size();
}

// Writes the i-th span; leaves the outputs untouched when out of range.
void lk_bf_result_span(const Result *r, size_t i, int *start, int *end) {
  if (r == nullptr || i >= r->spans.size()) {
    return;
  }
  *start = r->spans[i].first;
  *end = r->spans[i].second;
}

void lk_bf_result_free(Result *r) { delete r; }

}  // extern "C"
