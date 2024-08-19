#include "agents_python.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include "app.hpp"
#include "include/internal/cef_mac.h"

namespace py = pybind11;

BrowserApp::BrowserApp(const AppOptions& options) : options_(options) {
  app_ = new AgentApp(options_.dev_mode, options_.initialized_callback);
}

std::shared_ptr<BrowserImpl> BrowserApp::CreateBrowser(
    const std::string& url,
    const BrowserOptions& options) {
  app_->CreateBrowser(url, options.framerate, options.width, options.height,
                      options.created_callback,
                      [options](std::vector<CefRect> dirtyRects, const void* buffer,
                         int width, int height) {

                        PaintData event{};
                        std::vector<PaintRect> rects;
                        rects.reserve(dirtyRects.size());

                        for (const auto& rect : dirtyRects) {
                          rects.push_back({rect.x, rect.y, rect.width, rect.height});
                        }

                        event.dirtyRect = rects;
                        event.buffer = buffer;
                        event.width = width;
                        event.height = height;
                        options.paint_callback(event);
                      });
  return nullptr;  // std::make_shared<BrowserImpl>();
}

int BrowserApp::Run() {
  return RunAgentApp(app_);
}

BrowserImpl::BrowserImpl() {}

void BrowserImpl::SetSize(int width, int height) {}

py::memoryview paint_data_to_memoryview(const PaintData& event) {
  return py::memoryview::from_buffer(
      const_cast<uint32_t*>(static_cast<const uint32_t*>(event.buffer)),
      {event.height * event.width},
      {sizeof(uint32_t)}, true);
}

PYBIND11_MODULE(lkcef_python, m) {
  // Isn't that fucking cool? llm using browsers
  m.doc() = "Chromium Embedded Framework (CEF) for LiveKit Agents";

  py::class_<AppOptions>(m, "AppOptions")
      .def(py::init())
      .def_readwrite("dev_mode", &AppOptions::dev_mode)
      .def_readwrite("initialized_callback", &AppOptions::initialized_callback);

  py::class_<BrowserOptions>(m, "BrowserOptions")
      .def(py::init())
      .def_readwrite("framerate", &BrowserOptions::framerate)
      .def_readwrite("width", &BrowserOptions::width)
      .def_readwrite("height", &BrowserOptions::height)
      .def_readwrite("created_callback", &BrowserOptions::created_callback)
      .def_readwrite("paint_callback", &BrowserOptions::paint_callback);

  py::class_<BrowserApp>(m, "BrowserApp")
      .def(py::init<const AppOptions&>())
      .def("create_browser", &BrowserApp::CreateBrowser)
      .def("run", &BrowserApp::Run);

  py::class_<BrowserImpl>(m, "BrowserImpl")
      .def("set_size", &BrowserImpl::SetSize);

  py::class_<PaintRect>(m, "PaintRect")
      .def_readwrite("x", &PaintRect::x)
      .def_readwrite("y", &PaintRect::y)
      .def_readwrite("width", &PaintRect::width)
      .def_readwrite("height", &PaintRect::height);

  py::class_<PaintData>(m, "PaintData")
      .def(py::init())
      .def_readwrite("dirty_rects", &PaintData::dirtyRect)
      .def_readwrite("width", &PaintData::width)
      .def_readwrite("height", &PaintData::height)
      .def_property_readonly("buffer", [](const PaintData& event) {
        return paint_data_to_memoryview(event);
      });
}
