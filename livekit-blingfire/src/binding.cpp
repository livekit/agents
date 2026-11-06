// A rebuilt CPython boundary over Microsoft's unchanged public BlingFire C API.
// The linked tokenizer and model sources retain their upstream MIT license.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <blingfiretokdll.h>

namespace py = pybind11;
using Spans = std::vector<std::pair<int, int>>;
using TextFunction = decltype(&BlingFire::TextToSentences);
using OffsetFunction = decltype(&BlingFire::TextToSentencesWithOffsets);

static int capacity(const std::string& input) {
#if defined(__linux__)
    // Preserve the published Linux SDK wheel's empty-input RuntimeError contract.
    // Its independently captured CP314 oracle differs from macOS libc++ here.
    if (input.empty()) throw std::runtime_error("basic_string: construction from null is not valid");
#endif
    constexpr auto limit = static_cast<std::size_t>((std::numeric_limits<int>::max() - 1) / 2);
    if (input.size() > limit) throw std::overflow_error("BlingFire input exceeds its C ABI");
    return static_cast<int>(2 * input.size() + 1);
}

static std::string text(const std::string& input, TextFunction function) {
    const int limit = capacity(input);
    std::vector<char> output(static_cast<std::size_t>(limit));
    const int size = function(input.data(), static_cast<int>(input.size()), output.data(), limit);
    if (size < 0) return {};
    if (size == 0) throw std::length_error("basic_string");
    if (size > limit) throw std::runtime_error("BlingFire output exceeded its bound");
    // Preserve the original SDK's empty-input length_error/ValueError contract.
    return std::string(output.data(), static_cast<std::size_t>(size - 1));
}

static std::pair<std::string, Spans> offsets(
    const std::string& input, OffsetFunction function, char delimiter) {
    const int limit = capacity(input);
    std::vector<char> output(static_cast<std::size_t>(limit));
    std::vector<int> starts(static_cast<std::size_t>(limit));
    std::vector<int> ends(static_cast<std::size_t>(limit));
    const int size = function(input.data(), static_cast<int>(input.size()), output.data(),
                              starts.data(), ends.data(), limit);
    if (size < 0) return {{}, {}};
    if (size == 0) throw std::length_error("basic_string");
    if (size > limit) throw std::runtime_error("BlingFire output exceeded its bound");
    std::string result(output.data(), static_cast<std::size_t>(size - 1));
    const auto count = static_cast<std::size_t>(std::count(result.begin(), result.end(), delimiter)) + 1;
    if (count > starts.size()) throw std::runtime_error("BlingFire token count exceeded its bound");

    // Preserve Microsoft's public Python offset conversion, including its final
    // end fallback when a C byte offset is not a Unicode boundary. That detail
    // is observable for non-ASCII inputs which produce no word tokens.
    // Source: BlingFire/dist-pypi/blingfire/__init__.py at 18e9a19 (MIT).
    Spans spans(count);
    std::size_t position = 0;
    int characters = 0;
    for (std::size_t index = 0; index < input.size(); ++index) {
        const auto byte = static_cast<unsigned char>(input[index]);
        if ((byte & 0xc0) == 0x80) continue;
        while (position < 2 * count) {
            const bool ending = position % 2 != 0;
            const int offset = ending ? ends[position / 2] : starts[position / 2];
            if (offset < 0 || static_cast<std::size_t>(offset) >= input.size()) {
                throw std::runtime_error("BlingFire returned an invalid byte span");
            }
            if (static_cast<std::size_t>(offset) + static_cast<std::size_t>(ending) != index) break;
            auto& value = ending ? spans[position / 2].second : spans[position / 2].first;
            value = characters;
            ++position;
        }
        ++characters;
    }
    if (position < 2 * count) {
        auto& value = position % 2 != 0 ? spans[position / 2].second : spans[position / 2].first;
        value = characters;
        ++position;
    }
    if (position != 2 * count) throw std::runtime_error("BlingFire returned incomplete byte spans");
    return {std::move(result), std::move(spans)};
}

#ifndef BLINGFIRE_MODULE
#define BLINGFIRE_MODULE lk_blingfire
#endif

PYBIND11_MODULE(BLINGFIRE_MODULE, module) {
    module.def("text_to_sentences", [](const std::string& input) {
        return text(input, BlingFire::TextToSentences);
    });
    module.def("text_to_words", [](const std::string& input) {
        return text(input, BlingFire::TextToWords);
    });
    module.def("text_to_sentences_with_offsets", [](const std::string& input) -> py::tuple {
        const auto result = offsets(input, BlingFire::TextToSentencesWithOffsets, '\n');
        return py::make_tuple(result.first, result.second);
    });
    module.def("text_to_words_with_offsets", [](const std::string& input) -> py::tuple {
        const auto result = offsets(input, BlingFire::TextToWordsWithOffsets, ' ');
        return py::make_tuple(result.first, result.second);
    });
}
