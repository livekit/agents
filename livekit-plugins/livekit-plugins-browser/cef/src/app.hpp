#ifndef LKCEF_APP_HPP
#define LKCEF_APP_HPP

#include "handler.hpp"
#include "dev_renderer.hpp"
#include "include/cef_app.h"
#include "include/cef_base.h"
#include "include/cef_browser_process_handler.h"
#include "include/cef_client.h"
#include "include/internal/cef_ptr.h"

class AgentApp : public CefApp, public CefBrowserProcessHandler {
public:
  AgentApp(bool dev_mode);

  CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override {
    return this;
  }

  void OnBeforeCommandLineProcessing(
      const CefString &process_type,
      CefRefPtr<CefCommandLine> command_line) override;
  void OnContextInitialized() override;
  CefRefPtr<CefClient> GetDefaultClient() override;

  int Run();

private:
  IMPLEMENT_REFCOUNTING(AgentApp);

  CefRefPtr<AgentHandler> client_;
  std::unique_ptr<DevRenderer> dev_renderer_;

  bool dev_mode_;
};

#endif // LKCEF_APP_HPP
