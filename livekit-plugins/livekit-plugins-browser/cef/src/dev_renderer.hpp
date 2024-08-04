#ifndef LKCEF_APP_HPP
#define LKCEF_APP_HPP

#include "handler.hpp"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

class DevRenderer {
 public:
  DevRenderer();


 private:
  GLFWwindow* window_;
};

#endif // LKCEF_APP_HPP
