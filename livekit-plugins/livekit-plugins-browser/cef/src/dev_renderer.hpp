#ifndef LKCEF_DEV_RENDERER_HPP
#define LKCEF_DEV_RENDERER_HPP

#include "include/cef_app.h"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#define GLFW_EXPOSE_NATIVE_COCOA
//#define GLFW_NATIVE_INCLUDE_NONE
#include <GLFW/glfw3native.h>


class DevRenderer: public CefBaseRefCounted {
 public:
  DevRenderer();

  void Run();
  void Close();


  void OnAfterCreated(CefRefPtr<CefBrowser> browser);

  void OnPaint(CefRefPtr<CefBrowser> browser,
                             CefRenderHandler::PaintElementType type,
                             const CefRenderHandler::RectList&ts,
                             const void* buffer,
                             int width,
                             int height);

  void OnBeforeClose(CefRefPtr<CefBrowser> browser);

  void* getNativeWindowHandle() {
    return glfwGetCocoaWindow(window_);
  }

 private:
  struct RenderData{
    unsigned int texture_id;
    int view_width;
    int view_height;
  };

  GLFWwindow* window_ = nullptr;
  std::unordered_map<int, RenderData> render_data_;

  IMPLEMENT_REFCOUNTING(DevRenderer);
};

#endif // LKCEF_DEV_RENDERER_HPP
