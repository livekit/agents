#ifndef LKCEF_BROWSER_HANDLE_HPP
#define LKCEF_BROWSER_HANDLE_HPP

#include <list>

#include "include/cef_client.h"
#include "include/wrapper/cef_helpers.h"

class BrowserHandle : public CefBaseRefCounted {
 public:
  BrowserHandle(
              std::function<void()> created_callback,
                std::function<void(std::vector<CefRect> dirtyRects,
                                   const void* buffer,
                                   int width,
                                   int height)> paint_callback,
                std::function<void()> close_callback,
                int width,
                int height)
      : created_callback_(std::move(created_callback)),
        paint_callback_(std::move(paint_callback)),
        close_callback_(std::move(close_callback)),
        width_(width),
        height_(height) {}

  CefRefPtr<CefBrowser> browser_ = nullptr;
  std::function<void()> created_callback_ = nullptr;
  std::function<void(std::vector<CefRect> dirtyRect,
                     const void* buffer,
                     int width,
                     int height)>
      paint_callback_ = nullptr;
  std::function<void()> close_callback_ = nullptr;

  void SetSize(int width, int height);
  void Close();

  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }

  CefRefPtr<CefBrowser> GetBrowser() const { return browser_; }

 private:
  int width_ = 0;
  int height_ = 0;

  IMPLEMENT_REFCOUNTING(BrowserHandle);
};

struct BrowserStore : public CefBaseRefCounted {
  std::unordered_map<int, CefRefPtr<BrowserHandle>> browser_handles_;
  std::list<CefRefPtr<BrowserHandle>> pending_handles_;

  void AddPendingHandle(CefRefPtr<BrowserHandle> handle) {
    CEF_REQUIRE_UI_THREAD();
    pending_handles_.push_back(handle);
  }

  void RemovePendingHandle(CefRefPtr<BrowserHandle> handle) {
    CEF_REQUIRE_UI_THREAD();
    pending_handles_.remove(handle);
  }

  CefRefPtr<BrowserHandle> GetBrowserHandle(int identifier) {
    CEF_REQUIRE_UI_THREAD();
    return browser_handles_[identifier];
  }

  IMPLEMENT_REFCOUNTING(BrowserStore);
};

#endif  // LKCEF_BROWSER_HANDLE_HPP
