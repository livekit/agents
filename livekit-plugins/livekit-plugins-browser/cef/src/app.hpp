#ifndef LKCEF_APP_HPP
#define LKCEF_APP_HPP

#include "browser_handle.hpp"
#include "dev_renderer.hpp"
#include "handler.hpp"
#include "include/cef_app.h"
#include "include/cef_base.h"
#include "include/cef_browser_process_handler.h"
#include "include/cef_client.h"
#include "include/internal/cef_ptr.h"

class AgentApp : public CefApp, public CefBrowserProcessHandler {
 public:
  AgentApp(bool dev_mode,
           int remote_debugging_port,
           std::string root_cache_path,
           std::string framework_path,
           std::string main_bundle_path,
           std::string subprocess_path,
           std::function<void()> initialized_callback);

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
      int width,
      int height,
      std::function<void()> created_callback,
      std::function<void(std::vector<CefRect> dirtyRect,
                         const void* buffer,
                         int width,
                         int height)> paint_callback,
      std::function<void()> close_callback);

  int Run();

  bool IsDevMode() const { return dev_mode_; }
  int GetRemoteDebuggingPort() const { return remote_debugging_port_; }
  std::string GetRootCachePath() const { return root_cache_path_; }
  std::string GetFrameworkPath() const { return framework_path_; }
  std::string GetMainBundlePath() const { return main_bundle_path_; }
  std::string GetSubprocessPath() const { return subprocess_path_; }

 private:
  IMPLEMENT_REFCOUNTING(AgentApp);

  CefRefPtr<BrowserStore> browser_store_;
  CefRefPtr<AgentHandler> client_;
  CefRefPtr<DevToolsHandler> dev_client_;
  CefRefPtr<DevRenderer> dev_renderer_;

  bool dev_mode_;
  int remote_debugging_port_;
  std::string root_cache_path_;
  std::string framework_path_;
  std::string main_bundle_path_;
  std::string subprocess_path_;
  std::function<void()> initialized_callback_;
};

int RunAgentApp(CefRefPtr<AgentApp> app);

#endif  // LKCEF_APP_HPP
