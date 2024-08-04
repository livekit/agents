#ifndef LKCEF_AGENTS_PYTHON_HPP
#define LKCEF_AGENTS_PYTHON_HPP

#include <pybind11/pybind11.h>

class BrowserImpl;

struct AppOptions {
  AppOptions() : dev_mode(false) {}

  bool dev_mode;
};


struct BrowserOptions{
        BrowserOptions() : framerate(30) {}

  int framerate;
};


struct BrowserApp {
  BrowserApp(const AppOptions &options);

  std::shared_ptr<BrowserImpl> CreateBrowser(const std::string& url, const BrowserOptions &options);
  bool Start();
  bool Close();
};


struct BrowserImpl {
  BrowserImpl();

  void SetSize(int width, int height);
};

#endif // LKCEF_AGENTS_PYTHON_HPP
