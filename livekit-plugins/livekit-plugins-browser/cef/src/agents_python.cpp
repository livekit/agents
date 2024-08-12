#include "agents_python.hpp"

#include "app.hpp"
#include "include/internal/cef_mac.h"

namespace py = pybind11;

BrowserImpl::BrowserImpl() {

}


void BrowserImpl::start() {
    py::print("BrowserImpl::start()");
    AgentApp::run();
}


PYBIND11_MODULE(lkcef_python, m)
{
  m.doc() = "Chromium Embedded Framework (CEF) for LiveKit Agents";
  
    
  py::class_<BrowserImpl>(m, "BrowserImpl")
    .def(py::init<>())
    .def("start", &BrowserImpl::start);
}
