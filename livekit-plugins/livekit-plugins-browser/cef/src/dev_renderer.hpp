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

  void OnPaint(CefRefPtr<CefBrowser> browser,
                             CefRenderHandler::PaintElementType type,
                             const CefRenderHandler::RectList& dirtyRects,
                             const void* buffer,
                             int width,
                             int height);

  void* getNativeWindowHandle() {
    return glfwGetCocoaWindow(window_);
  }

 private:
  GLFWwindow* window_ = nullptr;

  IMPLEMENT_REFCOUNTING(DevRenderer);
};

#endif // LKCEF_DEV_RENDERER_HPP
