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
#include "keyboard_codes.h"

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

int glfw_key_to_cef_key(int glfwKey) {
  switch (glfwKey) {
    case GLFW_KEY_SPACE:
      return WebCore::VK_SPACE;
    case GLFW_KEY_APOSTROPHE:
      return WebCore::VK_OEM_7;
    case GLFW_KEY_COMMA:
      return WebCore::VK_OEM_COMMA;
    case GLFW_KEY_MINUS:
      return WebCore::VK_OEM_MINUS;
    case GLFW_KEY_PERIOD:
      return WebCore::VK_OEM_PERIOD;
    case GLFW_KEY_SLASH:
      return WebCore::VK_OEM_2;
    case GLFW_KEY_0:
      return WebCore::VK_0;
    case GLFW_KEY_1:
      return WebCore::VK_1;
    case GLFW_KEY_2:
      return WebCore::VK_2;
    case GLFW_KEY_3:
      return WebCore::VK_3;
    case GLFW_KEY_4:
      return WebCore::VK_4;
    case GLFW_KEY_5:
      return WebCore::VK_5;
    case GLFW_KEY_6:
      return WebCore::VK_6;
    case GLFW_KEY_7:
      return WebCore::VK_7;
    case GLFW_KEY_8:
      return WebCore::VK_8;
    case GLFW_KEY_9:
      return WebCore::VK_9;
    case GLFW_KEY_SEMICOLON:
      return WebCore::VK_OEM_1;
    case GLFW_KEY_EQUAL:
      return WebCore::VK_OEM_PLUS;
    case GLFW_KEY_A:
      return WebCore::VK_A;
    case GLFW_KEY_B:
      return WebCore::VK_B;
    case GLFW_KEY_C:
      return WebCore::VK_C;
    case GLFW_KEY_D:
      return WebCore::VK_D;
    case GLFW_KEY_E:
      return WebCore::VK_E;
    case GLFW_KEY_F:
      return WebCore::VK_F;
    case GLFW_KEY_G:
      return WebCore::VK_G;
    case GLFW_KEY_H:
      return WebCore::VK_H;
    case GLFW_KEY_I:
      return WebCore::VK_I;
    case GLFW_KEY_J:
      return WebCore::VK_J;
    case GLFW_KEY_K:
      return WebCore::VK_K;
    case GLFW_KEY_L:
      return WebCore::VK_L;
    case GLFW_KEY_M:
      return WebCore::VK_M;
    case GLFW_KEY_N:
      return WebCore::VK_N;
    case GLFW_KEY_O:
      return WebCore::VK_O;
    case GLFW_KEY_P:
      return WebCore::VK_P;
    case GLFW_KEY_Q:
      return WebCore::VK_Q;
    case GLFW_KEY_R:
      return WebCore::VK_R;
    case GLFW_KEY_S:
      return WebCore::VK_S;
    case GLFW_KEY_T:
      return WebCore::VK_T;
    case GLFW_KEY_U:
      return WebCore::VK_U;
    case GLFW_KEY_V:
      return WebCore::VK_V;
    case GLFW_KEY_W:
      return WebCore::VK_W;
    case GLFW_KEY_X:
      return WebCore::VK_X;
    case GLFW_KEY_Y:
      return WebCore::VK_Y;
    case GLFW_KEY_Z:
      return WebCore::VK_Z;
    case GLFW_KEY_LEFT_BRACKET:
      return WebCore::VK_OEM_4;
    case GLFW_KEY_BACKSLASH:
      return WebCore::VK_OEM_5;
    case GLFW_KEY_RIGHT_BRACKET:
      return WebCore::VK_OEM_6;
    case GLFW_KEY_GRAVE_ACCENT:
      return WebCore::VK_OEM_3;
    case GLFW_KEY_ESCAPE:
      return WebCore::VK_ESCAPE;
    case GLFW_KEY_ENTER:
      return WebCore::VK_RETURN;
    case GLFW_KEY_TAB:
      return WebCore::VK_TAB;
    case GLFW_KEY_BACKSPACE:
      return WebCore::VK_BACK;
    case GLFW_KEY_INSERT:
      return WebCore::VK_INSERT;
    case GLFW_KEY_DELETE:
      return WebCore::VK_DELETE;
    case GLFW_KEY_RIGHT:
      return WebCore::VK_RIGHT;
    case GLFW_KEY_LEFT:
      return WebCore::VK_LEFT;
    case GLFW_KEY_DOWN:
      return WebCore::VK_DOWN;
    case GLFW_KEY_UP:
      return WebCore::VK_UP;
    case GLFW_KEY_PAGE_UP:
      return WebCore::VK_PRIOR;
    case GLFW_KEY_PAGE_DOWN:
      return WebCore::VK_NEXT;
    case GLFW_KEY_HOME:
      return WebCore::VK_HOME;
    case GLFW_KEY_END:
      return WebCore::VK_END;
    case GLFW_KEY_CAPS_LOCK:
      return WebCore::VK_CAPITAL;
    case GLFW_KEY_SCROLL_LOCK:
      return WebCore::VK_SCROLL;
    case GLFW_KEY_NUM_LOCK:
      return WebCore::VK_NUMLOCK;
    case GLFW_KEY_PRINT_SCREEN:
      return WebCore::VK_SNAPSHOT;
    case GLFW_KEY_PAUSE:
      return WebCore::VK_PAUSE;
    case GLFW_KEY_F1:
      return WebCore::VK_F1;
    case GLFW_KEY_F2:
      return WebCore::VK_F2;
    case GLFW_KEY_F3:
      return WebCore::VK_F3;
    case GLFW_KEY_F4:
      return WebCore::VK_F4;
    case GLFW_KEY_F5:
      return WebCore::VK_F5;
    case GLFW_KEY_F6:
      return WebCore::VK_F6;
    case GLFW_KEY_F7:
      return WebCore::VK_F7;
    case GLFW_KEY_F8:
      return WebCore::VK_F8;
    case GLFW_KEY_F9:
      return WebCore::VK_F9;
    case GLFW_KEY_F10:
      return WebCore::VK_F10;
    case GLFW_KEY_F11:
      return WebCore::VK_F11;
    case GLFW_KEY_F12:
      return WebCore::VK_F12;
    // Add more cases as needed
    default:
      return WebCore::VK_UNKNOWN;
  }
}

static uint32_t glfw_mods_to_cef_mods(int glfw_mods) {
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

static std::optional<CefBrowserHost::MouseButtonType> glfw_button_to_cef_button(
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

DevRenderer::DevRenderer(CefRefPtr<BrowserStore> browser_store)
    : browser_store_(browser_store) {}

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

  ImVec4 clear_color = ImVec4(0.03f, 0.03f, 0.03f, 1.0f);
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    CefDoMessageLoopWork();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Flags used for the "invisible" dockspace frame
    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("Editor", nullptr, windowFlags);
    ImGui::PopStyleVar(3);
    ImGui::DockSpace(ImGui::GetID("EditorDockSpace"), ImVec2(),
                     ImGuiDockNodeFlags_PassthruCentralNode);

    // Focused browser input states
    BrowserData* focused_browser = nullptr;
    int browser_view_x = 0;
    int browser_view_y = 0;

    for (auto& [identifier, data] : browser_data_) {
      std::string name =
          (data.title.empty() ? "Browser #" + std::to_string(identifier)
                              : data.title) +
          "###Browser" + std::to_string(identifier);

      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
      if (ImGui::Begin(name.c_str())) {
        ImGui::BeginDisabled(!data.browser->CanGoBack());
        if (ImGui::ArrowButton("##BrowserBack", ImGuiDir_Left)) {
          data.browser->GoBack();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();

        ImGui::BeginDisabled(!data.browser->CanGoForward());
        if (ImGui::ArrowButton("##BrowserForward", ImGuiDir_Right)) {
          data.browser->GoForward();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();

        if (ImGui::InputText("##BrowserURL", &data.url,
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
          data.browser->GetMainFrame()->LoadURL(data.url);
        }

        ImGui::SameLine();

        if (ImGui::Button("Show DevTools")) {
          CefWindowInfo windowInfo{};
          CefBrowserSettings settings{};

          data.browser->GetHost()->ShowDevTools(
              windowInfo, DevToolsHandler::GetInstance(), settings, CefPoint());
        }

        ImVec2 size = ImGui::GetContentRegionAvail();

        // Resize the browser view if needed
        if (size.x > 0 && size.y > 0 &&
            (data.view_width != static_cast<int>(size.x) ||
             data.view_height != static_cast<int>(size.y))) {
          browser_store_->GetBrowserHandle(identifier)
              ->SetSize(static_cast<int>(size.x), static_cast<int>(size.y));
        }

        ImVec2 cursor_pos = ImGui::GetCursorScreenPos();

        bool is_focused = ImGui::IsWindowFocused();
        if (is_focused) {
          focused_browser = &data;
          browser_view_x = static_cast<int>(cursor_pos.x);
          browser_view_y = static_cast<int>(cursor_pos.y);
          data.browser->GetHost()->SetFocus(true);
        }

        // Render the browser tex
        ImGui::Image((void*)(intptr_t)data.texture_id,
                     ImVec2((float)data.view_width, (float)data.view_height));
      }
      ImGui::End();
      ImGui::PopStyleVar();
    }

    GLEQevent event;

    while (gleqNextEvent(&event)) {
      switch (event.type) {
        case GLEQ_CURSOR_MOVED:
        case GLEQ_BUTTON_PRESSED:
        case GLEQ_SCROLLED:
        case GLEQ_BUTTON_RELEASED:
          if (focused_browser) {
            CefMouseEvent cef_event;

            if (event.type == GLEQ_CURSOR_MOVED) {
              cef_event.x = event.pos.x - browser_view_x;
              cef_event.y = event.pos.y - browser_view_y;
              focused_browser->browser->GetHost()->SendMouseMoveEvent(cef_event,
                                                                      false);
            } else if (event.type == GLEQ_SCROLLED) {
              double xpos, ypos;
              glfwGetCursorPos(window_, &xpos, &ypos);
              cef_event.x = static_cast<int>(xpos) - browser_view_x;
              cef_event.y = static_cast<int>(ypos) - browser_view_y;

              static const int scrollbarPixelsPerTick = 20;
              int scroll_x =
                  static_cast<int>(event.scroll.x * scrollbarPixelsPerTick);
              int scroll_y =
                  static_cast<int>(event.scroll.y * scrollbarPixelsPerTick);

              focused_browser->browser->GetHost()->SendMouseWheelEvent(
                  cef_event, scroll_x, scroll_y);
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
        case GLEQ_KEY_PRESSED:
        case GLEQ_KEY_RELEASED:
          if (focused_browser) {
            CefKeyEvent cef_event;
            cef_event.windows_key_code =
                glfw_key_to_cef_key(event.keyboard.key);
            cef_event.native_key_code = event.keyboard.scancode;
            cef_event.modifiers = glfw_mods_to_cef_mods(event.keyboard.mods);
            cef_event.is_system_key = false;

            if (event.type == GLEQ_KEY_PRESSED) {
              cef_event.type = KEYEVENT_RAWKEYDOWN;
              focused_browser->browser->GetHost()->SendKeyEvent(cef_event);
            } else {
              cef_event.type = KEYEVENT_KEYUP;
              focused_browser->browser->GetHost()->SendKeyEvent(cef_event);
            }
          }
          break;
        case GLEQ_CODEPOINT_INPUT:
          if (focused_browser) {
            CefKeyEvent cef_event;
            cef_event.type = KEYEVENT_CHAR;
            cef_event.windows_key_code = 0;
            cef_event.native_key_code = 0;
            cef_event.modifiers = 0;
            cef_event.is_system_key = false;
            cef_event.unmodified_character = event.codepoint;
            cef_event.character = event.codepoint;
            focused_browser->browser->GetHost()->SendKeyEvent(cef_event);
          }
          break;
        default:
          break;
      }

      gleqFreeEvent(&event);
    }

    ImGui::End();
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
