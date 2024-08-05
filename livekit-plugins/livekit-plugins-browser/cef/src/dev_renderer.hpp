#ifndef LKCEF_DEV_RENDERER_HPP
#define LKCEF_DEV_RENDERER_HPP

#include "handler.hpp"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers


class DevRenderer {
 public:
  DevRenderer();

  bool Start();
  void Update();
  bool Close();

 private:
  bool started_;

  GLFWwindow* window_ = nullptr;
};

#endif // LKCEF_DEV_RENDERER_HPP
