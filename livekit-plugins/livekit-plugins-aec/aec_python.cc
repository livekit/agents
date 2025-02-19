#include "aec_python.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"
#include "modules/audio_processing/audio_buffer.h"

#include <cstdint>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

Aec::Aec(const AecOptions &options) : options_(options) {
  webrtc::Environment env = webrtc::CreateEnvironment();

  aec3_ = new webrtc::EchoCanceller3(
      env, webrtc::EchoCanceller3Config(), std::nullopt, options.sample_rate,
      options.num_channels, options.num_channels);

  cap_buf_ = new webrtc::AudioBuffer(options.sample_rate, options.num_channels,
                                     options.sample_rate, options.num_channels,
                                     options.sample_rate, options.num_channels);

  rend_buf_ = new webrtc::AudioBuffer(
      options.sample_rate, options.num_channels, options.sample_rate,
      options.num_channels, options.sample_rate, options.num_channels);
}

void Aec::CancelEcho(py::array_t<int16_t> cap,
                     const py::array_t<int16_t> rend) {
  webrtc::StreamConfig stream_cfg(options_.sample_rate, options_.num_channels);

  cap_buf_->CopyFrom(cap.mutable_data(), stream_cfg);
  rend_buf_->CopyFrom(rend.data(), stream_cfg);

  if (options_.sample_rate > 16000) {
    cap_buf_->SplitIntoFrequencyBands();
    rend_buf_->SplitIntoFrequencyBands();
  }

  aec3_->AnalyzeCapture(cap_buf_);
  aec3_->AnalyzeRender(rend_buf_);
  aec3_->ProcessCapture(cap_buf_, false);

  cap_buf_->CopyTo(stream_cfg, cap.mutable_data());
}

PYBIND11_MODULE(lkaec_python, m) {
  m.doc() = "Acoustic Echo Cancellation (AEC) for LiveKit Agents";

  py::class_<AecOptions>(m, "AecOptions")
      .def(py::init<>())
      .def_readwrite("sample_rate", &AecOptions::sample_rate)
      .def_readwrite("num_channels", &AecOptions::num_channels);

  py::class_<Aec>(m, "Aec")
      .def(py::init<const AecOptions &>())
      .def("cancel_echo", &Aec::CancelEcho);
}