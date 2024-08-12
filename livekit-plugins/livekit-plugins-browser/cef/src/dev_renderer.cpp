#include "dev_renderer.hpp"

#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "include/wrapper/cef_helpers.h"

#include "include/cef_app.h"

// DCHECK on gl errors.
#if DCHECK_IS_ON()
#define VERIFY_NO_ERROR                                                      \
  {                                                                          \
    int _gl_error = glGetError();                                            \
    DCHECK(_gl_error == GL_NO_ERROR) << "glGetError returned " << _gl_error; \
  }
#else
#define VERIFY_NO_ERROR
#endif

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

DevRenderer::DevRenderer() {
}

void DevRenderer::OnAfterCreated(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();
  int identifier = browser->GetIdentifier();

  unsigned int texture_id;
  glGenTextures(1, &texture_id);
  VERIFY_NO_ERROR;

  RenderData render_data{};
  render_data.texture_id = texture_id;
  render_data_.insert({identifier, render_data});

  glBindTexture(GL_TEXTURE_2D, texture_id);
  VERIFY_NO_ERROR;
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  VERIFY_NO_ERROR;
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

void DevRenderer::OnPaint(CefRefPtr<CefBrowser> browser,
             CefRenderHandler::PaintElementType type,
             const CefRenderHandler::RectList& dirtyRects,
             const void* buffer,
             int width,
             int height) {
  CEF_REQUIRE_UI_THREAD();

  if (type != CefRenderHandler::PaintElementType::PET_VIEW){
    std::cout << "Ignoring PET_POPUP" << std::endl;
    return; // Ignore PET_POPUP for now, bc I'm lazy
  }

  int identifier = browser->GetIdentifier();
  RenderData* render_data = &render_data_[identifier];

  int old_width = render_data->view_width;
  int old_height = render_data->view_height;

  render_data->view_width = width;
  render_data->view_height = height;

  glBindTexture(GL_TEXTURE_2D, render_data->texture_id);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, width);
  VERIFY_NO_ERROR;

  bool has_fullscreen_rect = dirtyRects.size() == 1 &&
                         dirtyRects[0] == CefRect(0, 0, width, height);

  if (old_width != width || old_height != height || has_fullscreen_rect) {
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    VERIFY_NO_ERROR;
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    VERIFY_NO_ERROR;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, buffer);
    VERIFY_NO_ERROR;
  } else {
    CefRenderHandler::RectList::const_iterator i = dirtyRects.begin();
    for (; i != dirtyRects.end(); ++i) {
      const CefRect& rect = *i;
      glPixelStorei(GL_UNPACK_SKIP_PIXELS, rect.x);
      VERIFY_NO_ERROR;
      glPixelStorei(GL_UNPACK_SKIP_ROWS, rect.y);
      VERIFY_NO_ERROR;
      glTexSubImage2D(GL_TEXTURE_2D, 0, rect.x, rect.y, rect.width,
                      rect.height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                      buffer);
      VERIFY_NO_ERROR;
    }
  }
}

void DevRenderer::OnBeforeClose(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();
  int identifier = browser->GetIdentifier();
  RenderData* render_data = &render_data_[identifier];
  glDeleteTextures(1, &render_data->texture_id);
  render_data_.erase(identifier);
}

void DevRenderer::Run() {
  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return;
  }

  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  window_ =
      glfwCreateWindow(800, 600, "livekit-plugins-browser (Development Window)",
                       nullptr, nullptr);

  if (!window_) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);  // Enable vsync

  IMGUI_CHECKVERSION();

  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init(glsl_version);


  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    CefDoMessageLoopWork();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::ShowDemoWindow();


    for (auto& [identifier, render_data] : render_data_) {
      ImGui::Begin("Browser");
      ImGui::Text("Browser %d", identifier);
      ImGui::Image((void*)(intptr_t)render_data.texture_id,
                   ImVec2(render_data.view_width, render_data.view_height));
      ImGui::End();
    }



    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window_);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window_);
  glfwTerminate();
}

void DevRenderer::Close() {
  //glfwSetWindowShouldClose(window_, GLFW_TRUE);
}
