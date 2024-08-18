#include "dev_renderer.hpp"

#include <iostream>

#include "handler.hpp"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include "include/cef_app.h"
#include "include/wrapper/cef_helpers.h"

#define GLEQ_IMPLEMENTATION
#define GLEQ_STATIC
#include "gleq.h"

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

uint32_t glfw_mods_to_cef_mods(int glfw_mods) {
  uint32_t cef_flags = 0;

  if (glfw_mods & 0x0001) {  // GLFW_MOD_SHIFT
    cef_flags |= (1 << 1);   // EVENTFLAG_SHIFT_DOWN
  }
  if (glfw_mods & 0x0002) {  // GLFW_MOD_CONTROL
    cef_flags |= (1 << 2);   // EVENTFLAG_CONTROL_DOWN
  }
  if (glfw_mods & 0x0004) {  // GLFW_MOD_ALT
    cef_flags |= (1 << 3);   // EVENTFLAG_ALT_DOWN
  }
  if (glfw_mods & 0x0008) {  // GLFW_MOD_SUPER
    cef_flags |=
        (1 << 7);  // EVENTFLAG_COMMAND_DOWN (Super key -> Command on Mac)
  }
  if (glfw_mods & 0x0010) {  // GLFW_MOD_CAPS_LOCK
    cef_flags |= (1 << 0);   // EVENTFLAG_CAPS_LOCK_ON
  }
  if (glfw_mods & 0x0020) {  // GLFW_MOD_NUM_LOCK
    cef_flags |= (1 << 8);   // EVENTFLAG_NUM_LOCK_ON
  }

  return cef_flags;
}

std::optional<CefBrowserHost::MouseButtonType> glfw_button_to_cef_button(
    int button) {
  switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      return CefBrowserHost::MouseButtonType::MBT_LEFT;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      return CefBrowserHost::MouseButtonType::MBT_MIDDLE;
    case GLFW_MOUSE_BUTTON_RIGHT:
      return CefBrowserHost::MouseButtonType::MBT_RIGHT;
    default:
      return std::nullopt;
  }
}

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

/*static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int
action, int mods) { DevRenderer* dev_renderer =
static_cast<DevRenderer*>(glfwGetWindowUserPointer(window));

  if (action == GLFW_PRESS || action == GLFW_RELEASE) {
    // Handle key event forwarding to CEF
    CefKeyEvent cef_event;
    cef_event.windows_key_code = key;
    cef_event.native_key_code = scancode;
    cef_event.is_system_key = false;
    cef_event.type = (action == GLFW_PRESS) ? KEYEVENT_RAWKEYDOWN :
KEYEVENT_KEYUP; cef_event.modifiers = mods;

    auto browser = GetBrowserHandleFromWindow(window);
    browser->GetHost()->SendKeyEvent(cef_event);
  }
}



void glfw_char_callback(GLFWwindow* window, unsigned int codepoint) {
  DevRenderer* dev_renderer =
static_cast<DevRenderer*>(glfwGetWindowUserPointer(window));

  CefKeyEvent cef_event;
  cef_event.character = codepoint;
  cef_event.type = KEYEVENT_CHAR;

  auto browser = GetBrowserHandleFromWindow(window);
  browser->GetHost()->SendKeyEvent(cef_event);
}*/

/*static void glfw_mouse_button_callback(GLFWwindow* window, int button, int
action, int mods) { DevRenderer* dev_renderer =
static_cast<DevRenderer*>(glfwGetWindowUserPointer(window));

  if (button >= 0 && button <= 2) {  // GLFW only supports buttons 0-2 (left,
right, middle) CefMouseEvent cef_event; double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    cef_event.x = static_cast<int>(xpos);
    cef_event.y = static_cast<int>(ypos);

    auto browser = GetBrowserHandleFromWindow(window);

    if (action == GLFW_PRESS) {
      browser->GetHost()->SendMouseClickEvent(cef_event,
static_cast<cef_mouse_button_type_t>(button), false, 1); } else if (action ==
GLFW_RELEASE) { browser->GetHost()->SendMouseClickEvent(cef_event,
static_cast<cef_mouse_button_type_t>(button), true, 1);
    }
  }
}

// Mouse move callback
static void glfw_cursor_position_callback(GLFWwindow* window, double xpos,
double ypos) { DevRenderer* dev_renderer =
static_cast<DevRenderer*>(glfwGetWindowUserPointer(window));
  DevRenderer::BrowserData* data = dev_renderer->getSelectedBrowserData();

  if (!data) return;

  CefMouseEvent cef_event;
  cef_event.x = static_cast<int>(xpos);
  cef_event.y = static_cast<int>(ypos);

  // Translate the coordinate to the ImGUI window

  data->browser->GetHost()->SendMouseMoveEvent(cef_event, false);
}

// Scroll callback
static void glfw_scroll_callback(GLFWwindow* window, double xoffset, double
yoffset) { DevRenderer* dev_renderer =
static_cast<DevRenderer*>(glfwGetWindowUserPointer(window));

  CefMouseEvent cef_event;
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  cef_event.x = static_cast<int>(xpos);
  cef_event.y = static_cast<int>(ypos);

  auto browser = GetBrowserHandleFromWindow(window);
  browser->GetHost()->SendMouseWheelEvent(cef_event, static_cast<int>(xoffset *
100), static_cast<int>(yoffset * 100));
}*/

DevRenderer::DevRenderer() {}

void DevRenderer::OnTitleChange(CefRefPtr<CefBrowser> browser,
                                const CefString& title) {
  CEF_REQUIRE_UI_THREAD();
  int identifier = browser->GetIdentifier();
  BrowserData* data = &browser_data_[identifier];
  data->title = title;
}

void DevRenderer::OnLoadingStateChange(CefRefPtr<CefBrowser> browser,
                                       bool isLoading,
                                       bool canGoBack,
                                       bool canGoForward) {
  if (!isLoading) {
    int identifier = browser->GetIdentifier();
    BrowserData* data = &browser_data_[identifier];
    data->url = browser->GetMainFrame()->GetURL();
  }
}

void DevRenderer::OnAfterCreated(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();
  int identifier = browser->GetIdentifier();

  unsigned int texture_id;
  glGenTextures(1, &texture_id);
  VERIFY_NO_ERROR;

  BrowserData data{};
  data.browser = browser;
  data.texture_id = texture_id;
  browser_data_.insert({identifier, data});

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

  if (type != CefRenderHandler::PaintElementType::PET_VIEW) {
    std::cout << "Ignoring PET_POPUP" << std::endl;
    return;  // Ignore PET_POPUP for now, bc I'm lazy
  }

  int identifier = browser->GetIdentifier();
  BrowserData* data = &browser_data_[identifier];

  int old_width = data->view_width;
  int old_height = data->view_height;

  data->view_width = width;
  data->view_height = height;

  glBindTexture(GL_TEXTURE_2D, data->texture_id);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, width);
  VERIFY_NO_ERROR;

  bool has_fullscreen_rect =
      dirtyRects.size() == 1 && dirtyRects[0] == CefRect(0, 0, width, height);

  if (old_width != width || old_height != height || has_fullscreen_rect) {
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    VERIFY_NO_ERROR;
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    VERIFY_NO_ERROR;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA,
                 GL_UNSIGNED_INT_8_8_8_8_REV, buffer);
    VERIFY_NO_ERROR;
  } else {
    CefRenderHandler::RectList::const_iterator i = dirtyRects.begin();
    for (; i != dirtyRects.end(); ++i) {
      const CefRect& rect = *i;
      glPixelStorei(GL_UNPACK_SKIP_PIXELS, rect.x);
      VERIFY_NO_ERROR;
      glPixelStorei(GL_UNPACK_SKIP_ROWS, rect.y);
      VERIFY_NO_ERROR;
      glTexSubImage2D(GL_TEXTURE_2D, 0, rect.x, rect.y, rect.width, rect.height,
                      GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, buffer);
      VERIFY_NO_ERROR;
    }
  }
}

void DevRenderer::OnBeforeClose(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();
  int identifier = browser->GetIdentifier();
  BrowserData* data = &browser_data_[identifier];
  glDeleteTextures(1, &data->texture_id);
  browser_data_.erase(identifier);
}

void DevRenderer::Run() {
  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return;
  }

  gleqInit();

  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  window_ =
      glfwCreateWindow(800, 600, "livekit-plugins-browser (Development Window)",
                       nullptr, nullptr);

  gleqTrackWindow(window_);

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

  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    CefDoMessageLoopWork();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Focused browser input states
    BrowserData* focused_browser = nullptr;
    int browser_view_x = 0;
    int browser_view_y = 0;

    for (auto& [identifier, data] : browser_data_) {
      std::string name =
          (data.title.empty() ? "Browser #" + std::to_string(identifier)
                              : data.title) +
          "###Browser" + std::to_string(identifier);

      if (ImGui::Begin(name.c_str())) {
        ImVec2 size = ImGui::GetContentRegionAvail();

        // Resize the browser view if needed
        if (size.x > 0 && size.y > 0 &&
            (data.view_width != static_cast<int>(size.x) ||
             data.view_height != static_cast<int>(size.y))) {
          AgentHandler::GetInstance()
              ->GetBrowserHandle(identifier)
              ->SetSize(static_cast<int>(size.x), static_cast<int>(size.y));
        }

        if (ImGui::InputText("URL", &data.url,
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
          data.browser->GetMainFrame()->LoadURL(data.url);
        }

        ImVec2 cursor_pos = ImGui::GetCursorScreenPos();

        bool is_focused = ImGui::IsWindowFocused();
        if (is_focused) {
          focused_browser = &data;
          browser_view_x = static_cast<int>(cursor_pos.x);
          browser_view_y = static_cast<int>(cursor_pos.y);
        }

        // Render the browser tex
        ImGui::Image((void*)(intptr_t)data.texture_id,
                     ImVec2((float)data.view_width, (float)data.view_height));
      }
      ImGui::End();
    }

    GLEQevent event;

    while (gleqNextEvent(&event)) {
      switch (event.type) {
        case GLEQ_CURSOR_MOVED:
        case GLEQ_BUTTON_PRESSED:
        case GLEQ_BUTTON_RELEASED:
          if (focused_browser) {
            CefMouseEvent cef_event;

            if (event.type == GLEQ_CURSOR_MOVED) {
              cef_event.x = event.pos.x - browser_view_x;
              cef_event.y = event.pos.y - browser_view_y;
              focused_browser->browser->GetHost()->SendMouseMoveEvent(cef_event,
                                                                      false);
            } else {
              double xpos, ypos;
              glfwGetCursorPos(window_, &xpos, &ypos);
              cef_event.x = static_cast<int>(xpos) - browser_view_x;
              cef_event.y = static_cast<int>(ypos) - browser_view_y;
              cef_event.modifiers = glfw_mods_to_cef_mods(event.mouse.mods);

              std::optional<CefBrowserHost::MouseButtonType> cef_button =
                  glfw_button_to_cef_button(event.mouse.button);

              if (cef_button.has_value()) {
                focused_browser->browser->GetHost()->SendMouseClickEvent(
                    cef_event, cef_button.value(),
                    event.type == GLEQ_BUTTON_RELEASED, 1);
              }
            }
          }
          break;
        default:
          break;
      }

      gleqFreeEvent(&event);
    }

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
  // glfwSetWindowShouldClose(window_, GLFW_TRUE);
}
