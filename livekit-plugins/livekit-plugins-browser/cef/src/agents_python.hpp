#ifndef LKCEF_AGENTS_PYTHON_HPP
#define LKCEF_AGENTS_PYTHON_HPP

#include <functional>
#include <memory>

#include "app.hpp"

class BrowserImpl;

struct AppOptions {
  bool dev_mode = false;
  std::function<void()> initialized_callback = nullptr;
};

struct BrowserOptions {
  int framerate = 30;
  int width = 800;
  int height = 600;
  std::function<void()> created_callback = nullptr;
};

struct BrowserApp {
  BrowserApp(const AppOptions& options);

  std::shared_ptr<BrowserImpl> CreateBrowser(const std::string& url,
                                             const BrowserOptions& options);
  int Run();

 private:
  AppOptions options_;
  CefRefPtr<AgentApp> app_;
};

struct BrowserImpl {
  BrowserImpl();

  void SetSize(int width, int height);
};

#endif  // LKCEF_AGENTS_PYTHON_HPP
