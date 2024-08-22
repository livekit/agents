#ifndef LKCEF_AGENTS_PYTHON_HPP
#define LKCEF_AGENTS_PYTHON_HPP

#include <functional>
#include <memory>

#include "app.hpp"

class BrowserImpl;
struct PaintData;

struct AppOptions {
  bool dev_mode = false;
  int remote_debugging_port = 0;
  std::string root_cache_path;
  std::string framework_path;
  std::string main_bundle_path;
  std::string subprocess_path;
  std::function<void()> initialized_callback = nullptr;
};

struct BrowserOptions {
  int framerate = 30;
  int width = 800;
  int height = 600;
  std::function<void(std::shared_ptr<BrowserImpl>)> created_callback = nullptr;
  std::function<void(const PaintData&)> paint_callback = nullptr;
  std::function<void()> close_callback = nullptr;
};

struct BrowserApp {
  BrowserApp(const AppOptions& options);

  bool CreateBrowser(const std::string& url, const BrowserOptions& options);
  void CreateBrowserOnUIThread(const std::string& url, const BrowserOptions& options);

  int Run();

 private:
  AppOptions options_;
  CefRefPtr<AgentApp> app_;
  std::list<std::shared_ptr<BrowserImpl>> browsers_;
};

struct BrowserImpl {
  BrowserImpl();

  void SetSize(int width, int height);
  void Close();
  int Identifier() const;

  CefRefPtr<BrowserHandle> handle = nullptr;
};

struct PaintRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct PaintData {
  std::vector<PaintRect> dirtyRect;
  const void* buffer;
  int width;
  int height;
};

#endif  // LKCEF_AGENTS_PYTHON_HPP
