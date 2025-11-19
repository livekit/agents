#include "app.hpp"

#include <iostream>
#include <string>
#include <utility>

#include "include/cef_command_line.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_helpers.h"

AgentApp::AgentApp(bool dev_mode,
                   int remote_debugging_port,
                   std::string root_cache_path,
                   std::string framework_path,
                   std::string main_bundle_path,
                   std::string subprocess_path,
                   std::function<void()> initialized_callback)
    : dev_mode_(dev_mode),
      remote_debugging_port_(remote_debugging_port),
      root_cache_path_(std::move(root_cache_path)),
      framework_path_(std::move(framework_path)),
      main_bundle_path_(std::move(main_bundle_path)),
      subprocess_path_(std::move(subprocess_path)),
      initialized_callback_(std::move(initialized_callback)) {
  browser_store_ = CefRefPtr<BrowserStore>(new BrowserStore());

  if (dev_mode)
    dev_renderer_ = CefRefPtr<DevRenderer>(new DevRenderer(browser_store_));
}

void AgentApp::OnBeforeCommandLineProcessing(
    const CefString& process_type,
    CefRefPtr<CefCommandLine> command_line) {
  command_line->AppendSwitch("--disable-gpu");
  command_line->AppendSwitch("--disable-gpu-compositing");
  command_line->AppendSwitch("--enable-chrome-runtime");
  // command_line->AppendSwitch("--enable-begin-frame-scheduling");
}

void AgentApp::OnContextInitialized() {
  CEF_REQUIRE_UI_THREAD();  // Main thread in our case
  client_ =
      CefRefPtr<AgentHandler>(new AgentHandler(browser_store_, dev_renderer_));
  dev_client_ = CefRefPtr<DevToolsHandler>(new DevToolsHandler());

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
    std::function<void()> created_callback,
    std::function<void(std::vector<CefRect> dirtyRects,
                       const void* buffer,
                       int width,
                       int height)> paint_callback,
    std::function<void()> close_callback) {
  CEF_REQUIRE_UI_THREAD();

  // windowInfo.SetAsWindowless(dev_renderer_->getNativeWindowHandle());
  CefWindowInfo windowInfo;
  windowInfo.SetAsWindowless(nullptr);

  CefBrowserSettings settings;
  settings.windowless_frame_rate = framerate;
  settings.background_color = CefColorSetARGB(255, 255, 255, 255);

  CefRefPtr<BrowserHandle> browser_handle =
      new BrowserHandle(std::move(created_callback), std::move(paint_callback),
                        std::move(close_callback), width, height);

  browser_store_->AddPendingHandle(browser_handle);

  bool result = CefBrowserHost::CreateBrowser(windowInfo, client_, url,
                                              settings, nullptr, nullptr);
  if (!result) {
    browser_store_->RemovePendingHandle(browser_handle);
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

  // Close all browsers

  return 0;
}
