#include "agents_python.hpp"

#include "app.hpp"
#include "include/internal/cef_mac.h"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace py = pybind11;

BrowserApp::BrowserApp(const AppOptions& options) : options_(options) {
  app_ = new AgentApp(options_.dev_mode, options_.initialized_callback);
}

std::shared_ptr<BrowserImpl> BrowserApp::CreateBrowser(
    const std::string& url,
    const BrowserOptions& options) {

  app_->CreateBrowser(url, options.framerate, options.width, options.height, options.created_callback);
  return nullptr;//std::make_shared<BrowserImpl>();
}

int BrowserApp::Run() {
  return RunAgentApp(app_);
}

BrowserImpl::BrowserImpl() {}

void BrowserImpl::SetSize(int width, int height) {}

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
      .def_readwrite("created_callback", &BrowserOptions::created_callback);

  py::class_<BrowserApp>(m, "BrowserApp")
      .def(py::init<const AppOptions&>())
      .def("create_browser", &BrowserApp::CreateBrowser)
      .def("run", &BrowserApp::Run);

  py::class_<BrowserImpl>(m, "BrowserImpl")
      .def("set_size", &BrowserImpl::SetSize);
}
