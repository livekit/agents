#ifndef LKCEF_APP_HPP
#define LKCEF_APP_HPP

#include "include/cef_app.h"
#include "include/cef_base.h"
#include "include/cef_browser_process_handler.h"
#include "include/cef_client.h"
#include "include/internal/cef_ptr.h"

class AgentApp : public CefApp, public CefBrowserProcessHandler {
public:
  AgentApp();

  CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override {
    return this;
  }

  void OnContextInitialized() override;
  CefRefPtr<CefClient> GetDefaultClient() override;

  static int run();

private:
  IMPLEMENT_REFCOUNTING(AgentApp);
};

#endif // LKCEF_APP_HPP
