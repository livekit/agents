#include "agents_python.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "app.hpp"
#include "include/base/cef_callback.h"
#include "include/internal/cef_mac.h"
#include "include/wrapper/cef_closure_task.h"

namespace py = pybind11;

BrowserApp::BrowserApp(const AppOptions& options) : options_(options) {
  app_ = new AgentApp(options_.dev_mode, options.remote_debugging_port,
                      options.root_cache_path, options.framework_path,
                      options.main_bundle_path, options.subprocess_path,
                      options_.initialized_callback);
}

bool BrowserApp::CreateBrowser(const std::string& url,
                               const BrowserOptions& options) {
  if (CefCurrentlyOn(TID_UI)) {
    CreateBrowserOnUIThread(url, options);
    return true;
  }

  // TODO(theomonnom): Document base::Unretained
  CefPostTask(TID_UI, base::BindOnce(&BrowserApp::CreateBrowserOnUIThread,
                                     base::Unretained(this), url, options));

  return true;
}

void BrowserApp::CreateBrowserOnUIThread(const std::string& url,
                                         const BrowserOptions& options) {
  std::shared_ptr<BrowserImpl> browser_impl = std::make_shared<BrowserImpl>();
  browsers_.push_back(browser_impl);

  CefRefPtr<BrowserHandle> handle = app_->CreateBrowser(
      url, options.framerate, options.width, options.height,
      [options, browser_impl]() { options.created_callback(browser_impl); },
      [options](std::vector<CefRect> dirtyRects, const void* buffer, int width,
                int height) {
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
      },
      options.close_callback);

  browser_impl->handle = handle;
}

int BrowserApp::Run() {
  return RunAgentApp(app_);
}

BrowserImpl::BrowserImpl() {}

void BrowserImpl::SetSize(int width, int height) {
  if (handle)
    handle->SetSize(width, height);
}

void BrowserImpl::Close() {
  if (handle)
    handle->Close();
}

int BrowserImpl::Identifier() const {
  return handle->GetBrowser()->GetIdentifier();
}

py::memoryview paint_data_to_memoryview(const PaintData& event) {
  return py::memoryview::from_buffer(
      const_cast<uint32_t*>(static_cast<const uint32_t*>(event.buffer)),
      {event.height * event.width}, {sizeof(uint32_t)}, true);
}

PYBIND11_MODULE(lkcef_python, m) {
  // Isn't that fucking cool? llm using browsers
  m.doc() = "Chromium Embedded Framework (CEF) for LiveKit Agents";

  py::class_<AppOptions>(m, "AppOptions")
      .def(py::init())
      .def_readwrite("dev_mode", &AppOptions::dev_mode)
      .def_readwrite("remote_debugging_port",
                     &AppOptions::remote_debugging_port)
      .def_readwrite("root_cache_path", &AppOptions::root_cache_path)
      .def_readwrite("framework_path", &AppOptions::framework_path)
      .def_readwrite("main_bundle_path", &AppOptions::main_bundle_path)
      .def_readwrite("subprocess_path", &AppOptions::subprocess_path)
      .def_readwrite("initialized_callback", &AppOptions::initialized_callback);

  py::class_<BrowserOptions>(m, "BrowserOptions")
      .def(py::init())
      .def_readwrite("framerate", &BrowserOptions::framerate)
      .def_readwrite("width", &BrowserOptions::width)
      .def_readwrite("height", &BrowserOptions::height)
      .def_readwrite("created_callback", &BrowserOptions::created_callback)
      .def_readwrite("paint_callback", &BrowserOptions::paint_callback)
      .def_readwrite("close_callback", &BrowserOptions::close_callback);

  py::class_<BrowserApp>(m, "BrowserApp")
      .def(py::init<const AppOptions&>())
      .def("create_browser", &BrowserApp::CreateBrowser)
      .def("run", &BrowserApp::Run, py::call_guard<py::gil_scoped_release>());

  py::class_<BrowserImpl, std::shared_ptr<BrowserImpl>>(m, "BrowserImpl")
      .def("set_size", &BrowserImpl::SetSize)
      .def("close", &BrowserImpl::Close)
      .def("identifier", &BrowserImpl::Identifier);

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
