#include "agents_python.hpp"

#include "app.hpp"
#include "include/internal/cef_mac.h"

namespace py = pybind11;

BrowserApp::BrowserApp(const AppOptions& options) {}

std::shared_ptr<BrowserImpl> BrowserApp::CreateBrowser(
    const std::string& url,
    const BrowserOptions& options) {
  return std::make_shared<BrowserImpl>();
}

bool BrowserApp::Start() {
  // AgentApp::run();
}

bool BrowserApp::Close() {}

BrowserImpl::BrowserImpl() {}

void BrowserImpl::SetSize(int width, int height) {}

PYBIND11_MODULE(lkcef_python, m) {
  // Isn't that fucking cool? llm using browsers
  m.doc() = "Chromium Embedded Framework (CEF) for LiveKit Agents";

  py::class_<AppOptions>(m, "AppOptions")
      .def(py::init())
      .def_readwrite("dev_mode", &AppOptions::dev_mode);

  py::class_<BrowserOptions>(m, "BrowserOptions")
      .def(py::init())
      .def_readwrite("framerate", &BrowserOptions::framerate);

  py::class_<BrowserApp>(m, "BrowserApp")
      .def(py::init<const AppOptions&>())
      .def("create_browser", &BrowserApp::CreateBrowser)
      .def("start", &BrowserApp::Start)
      .def("close", &BrowserApp::Close);

  py::class_<BrowserImpl>(m, "BrowserImpl")
      .def("set_size", &BrowserImpl::SetSize);
}
