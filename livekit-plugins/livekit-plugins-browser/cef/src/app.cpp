#include "app.hpp"

#include <iostream>
#include <string>
#include <utility>

#include "include/cef_command_line.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_helpers.h"

AgentApp::AgentApp(bool dev_mode, std::function<void()> initialized_callback)
    : dev_mode_(dev_mode),
      initialized_callback_(std::move(initialized_callback)) {
  if (dev_mode)
    dev_renderer_ = CefRefPtr<DevRenderer>(new DevRenderer());
}

void AgentApp::OnBeforeCommandLineProcessing(
    const CefString& process_type,
    CefRefPtr<CefCommandLine> command_line) {
  command_line->AppendSwitch("--disable-gpu");
  command_line->AppendSwitch("--disable-gpu-compositing");
  // command_line->AppendSwitch("--enable-begin-frame-scheduling");
}

void AgentApp::OnContextInitialized() {
  CEF_REQUIRE_UI_THREAD();  // Main thread in our case
  client_ = CefRefPtr<AgentHandler>(new AgentHandler(dev_renderer_));

  if (initialized_callback_)
    initialized_callback_();
}

CefRefPtr<CefClient> AgentApp::GetDefaultClient() {
  return client_;
}

CefRefPtr<BrowserHandle> AgentApp::CreateBrowser(
    const std::string& url,
    int framerate,
    int width,
    int height,
    std::function<void()> created_callback) {
  CEF_REQUIRE_UI_THREAD();
  CefWindowInfo windowInfo;
  // windowInfo.SetAsWindowless(dev_renderer_->getNativeWindowHandle());
  windowInfo.SetAsWindowless(nullptr);

  CefRefPtr<CefCommandLine> command_line =
      CefCommandLine::GetGlobalCommandLine();

  CefBrowserSettings settings;
  settings.background_color = CefColorSetARGB(255, 255, 255, 255);

  CefRefPtr<BrowserHandle> browser_handle =
        new BrowserHandle(created_callback, width, height);

  client_->AddPendingHandle(browser_handle);

  bool result = CefBrowserHost::CreateBrowser(windowInfo, client_, url,
                                              settings, nullptr, nullptr);
  if (!result) {
    client_->RemovePendingHandle(browser_handle);
    return nullptr;
  }
  return browser_handle;
}

int AgentApp::Run() {
  if (dev_mode_) {
    dev_renderer_->Run();
  } else {
    CefRunMessageLoop();
  }

  return 0;
}
