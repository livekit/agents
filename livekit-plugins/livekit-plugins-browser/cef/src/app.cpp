#include "app.hpp"

#include <string>

#include "include/cef_command_line.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_helpers.h"

AgentApp::AgentApp(bool dev_mode) : dev_mode_(dev_mode) {

}

void AgentApp::OnBeforeCommandLineProcessing(
    const CefString& process_type,
    CefRefPtr<CefCommandLine> command_line) {
  command_line->AppendSwitch("--disable-gpu");
  command_line->AppendSwitch("--disable-gpu-compositing");
  // command_line->AppendSwitch("--enable-begin-frame-scheduling");
}

void AgentApp::OnContextInitialized() {
  CEF_REQUIRE_UI_THREAD();
  client_ = CefRefPtr<AgentHandler>(new AgentHandler());

  if (dev_mode_)
    dev_renderer_ = std::make_unique<DevRenderer>();
}

CefRefPtr<CefClient> AgentApp::GetDefaultClient() {
  return client_;
}

int AgentApp::Run() {
  while (true) {
    CefDoMessageLoopWork();

    if (dev_renderer_)
      dev_renderer_->Update();
  }

  return 0;
}




