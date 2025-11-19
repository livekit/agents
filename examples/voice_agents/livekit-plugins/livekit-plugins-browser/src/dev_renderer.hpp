#ifndef LKCEF_DEV_RENDERER_HPP
#define LKCEF_DEV_RENDERER_HPP

#include "include/cef_app.h"
#include "browser_handle.hpp"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#define GLFW_EXPOSE_NATIVE_COCOA
//#define GLFW_NATIVE_INCLUDE_NONE
#include <GLFW/glfw3native.h>


class DevRenderer: public CefBaseRefCounted {
 public:
  DevRenderer(CefRefPtr<BrowserStore> browser_store);

  void Run();
  void Close();

  void OnTitleChange(CefRefPtr<CefBrowser> browser,
                     const CefString &title);

  void OnLoadingStateChange(CefRefPtr<CefBrowser> browser,
                            bool isLoading,
                            bool canGoBack,
                            bool canGoForward);

  void OnAfterCreated(CefRefPtr<CefBrowser> browser);

  void OnPaint(CefRefPtr<CefBrowser> browser,
                             CefRenderHandler::PaintElementType type,
                             const CefRenderHandler::RectList&ts,
                             const void* buffer,
                             int width,
                             int height);

  void OnBeforeClose(CefRefPtr<CefBrowser> browser);

  void* getNativeWindowHandle() const {
    return glfwGetCocoaWindow(window_);
  }

 private:
  struct BrowserData{
    CefRefPtr<CefBrowser> browser;
    unsigned int texture_id;
    int view_width;
    int view_height;
    std::string title;
    std::string url;
  };

  GLFWwindow* window_ = nullptr;
  std::unordered_map<int, BrowserData> browser_data_;

  CefRefPtr<BrowserStore> browser_store_;

  IMPLEMENT_REFCOUNTING(DevRenderer);
};

#endif // LKCEF_DEV_RENDERER_HPP
