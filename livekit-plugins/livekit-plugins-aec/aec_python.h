#ifndef LKAEC_AGENTS_PYTHON_HPP
#define LKAEC_AGENTS_PYTHON_HPP

#include "modules/audio_processing/aec3/echo_canceller3.h"
#include "modules/audio_processing/audio_buffer.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct AecOptions {
  int sample_rate;
  int num_channels;
};

class Aec {
public:
  Aec(const AecOptions &options);

  void CancelEcho(py::array_t<int16_t> cap, const py::array_t<int16_t> rend);

private:
  AecOptions options_;
  webrtc::EchoCanceller3 *aec3_;
  webrtc::AudioBuffer *cap_buf_;
  webrtc::AudioBuffer *rend_buf_;
};

#endif // LKAEC_AGENTS_PYTHON_HPP
