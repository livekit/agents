#ifndef LKCEF_APP_HPP
#define LKCEF_APP_HPP

#include "dev_renderer.hpp"
#include "handler.hpp"
#include "include/cef_app.h"
#include "include/cef_base.h"
#include "include/cef_browser_process_handler.h"
#include "include/cef_client.h"
#include "include/internal/cef_ptr.h"

class AgentApp : public CefApp, public CefBrowserProcessHandler {
 public:
  AgentApp(bool dev_mode, std::function<void()> initialized_callback);

  CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override {
    return this;
  }

  void OnBeforeCommandLineProcessing(
      const CefString& process_type,
      CefRefPtr<CefCommandLine> command_line) override;

  void OnContextInitialized() override;

  CefRefPtr<CefClient> GetDefaultClient() override;

  CefRefPtr<BrowserHandle> CreateBrowser(
      const std::string& url,
      int framerate,
      std::function<void()> created_callback);

  int Run();

 private:
  IMPLEMENT_REFCOUNTING(AgentApp);

  CefRefPtr<AgentHandler> client_;
  CefRefPtr<DevRenderer> dev_renderer_;

  bool dev_mode_;
  std::function<void()> initialized_callback_;
};

int RunAgentApp(CefRefPtr<AgentApp> app);

#endif  // LKCEF_APP_HPP
