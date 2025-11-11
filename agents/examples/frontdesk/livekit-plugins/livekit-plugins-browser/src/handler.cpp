#include "handler.hpp"

#include <iostream>

#include "include/base/cef_callback.h"
#include "include/cef_parser.h"
#include "include/views/cef_browser_view.h"
#include "include/wrapper/cef_closure_task.h"
#include "include/wrapper/cef_helpers.h"

DevToolsHandler* g_dev_instance = nullptr;

DevToolsHandler::DevToolsHandler() {
  g_dev_instance = this;
}

DevToolsHandler::~DevToolsHandler() {
  g_dev_instance = nullptr;
}

DevToolsHandler* DevToolsHandler::GetInstance() {
  return g_dev_instance;
}

AgentHandler* g_instance = nullptr;

AgentHandler::AgentHandler(CefRefPtr<BrowserStore> browser_store,
                           CefRefPtr<DevRenderer> dev_renderer)
    : browser_store_(std::move(browser_store)),
      dev_renderer_(std::move(dev_renderer)) {
  g_instance = this;
}

AgentHandler::~AgentHandler() {
  g_instance = nullptr;
}

AgentHandler* AgentHandler::GetInstance() {
  return g_instance;
}

void AgentHandler::OnTitleChange(CefRefPtr<CefBrowser> browser,
                                 const CefString& title) {
  CEF_REQUIRE_UI_THREAD();
  if (dev_renderer_)
    dev_renderer_->OnTitleChange(browser, title);
}

void AgentHandler::OnPaint(CefRefPtr<CefBrowser> browser,
                           PaintElementType type,
                           const RectList& dirtyRects,
                           const void* buffer,
                           int width,
                           int height) {
  CEF_REQUIRE_UI_THREAD();

  int identifier = browser->GetIdentifier();
  CefRefPtr<BrowserHandle> handle =
      browser_store_->browser_handles_[identifier];
  if (handle->paint_callback_)
    handle->paint_callback_(dirtyRects, buffer, width, height);

  if (dev_renderer_)
    dev_renderer_->OnPaint(browser, type, dirtyRects, buffer, width, height);
}

void AgentHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) {
  CEF_REQUIRE_UI_THREAD();

  int identifier = browser->GetIdentifier();
  CefRefPtr<BrowserHandle>& handle =
      browser_store_->browser_handles_[identifier];
  rect.Set(0, 0, handle->GetWidth(), handle->GetHeight());
};

void AgentHandler::OnAudioStreamPacket(CefRefPtr<CefBrowser> browser,
                                       const float** data,
                                       int frames,
                                       int64_t pts) {
  // std::cout << "OnAudioStreamPacket" << std::endl;
}

void AgentHandler::OnAudioStreamStarted(CefRefPtr<CefBrowser> browser,
                                        const CefAudioParameters& params,
                                        int channels) {}

void AgentHandler::OnAudioStreamStopped(CefRefPtr<CefBrowser> browser) {}

void AgentHandler::OnAudioStreamError(CefRefPtr<CefBrowser> browser,
                                      const CefString& message) {}

bool AgentHandler::OnBeforePopup(CefRefPtr<CefBrowser> browser,
                                 CefRefPtr<CefFrame> frame,
                                 const CefString& target_url,
                                 const CefString& target_frame_name,
                                 WindowOpenDisposition target_disposition,
                                 bool user_gesture,
                                 const CefPopupFeatures& popupFeatures,
                                 CefWindowInfo& windowInfo,
                                 CefRefPtr<CefClient>& client,
                                 CefBrowserSettings& settings,
                                 CefRefPtr<CefDictionaryValue>& extra_info,
                                 bool* no_javascript_access) {
  browser->GetMainFrame()->LoadURL(target_url);
  return true;
}

void AgentHandler::OnAfterCreated(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();

  if (browser->IsPopup()) {
    return;
  }

  int identifier = browser->GetIdentifier();
  CefRefPtr<BrowserHandle> handle = browser_store_->pending_handles_.front();
  browser_store_->pending_handles_.pop_front();

  handle->browser_ = browser;
  browser_store_->browser_handles_[identifier] = handle;

  if (handle->created_callback_)
    handle->created_callback_();

  if (dev_renderer_)
    dev_renderer_->OnAfterCreated(browser);
}

bool AgentHandler::DoClose(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();
  int identifier = browser->GetIdentifier();
  CefRefPtr<BrowserHandle> handle =
      browser_store_->browser_handles_[identifier];
  browser_store_->browser_handles_.erase(identifier);

  if (handle->close_callback_)
    handle->close_callback_();

  return false;
}

void AgentHandler::OnBeforeClose(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();

  if (dev_renderer_)
    dev_renderer_->OnBeforeClose(browser);
}

void AgentHandler::OnLoadingStateChange(CefRefPtr<CefBrowser> browser,
                                        bool isLoading,
                                        bool canGoBack,
                                        bool canGoForward) {
  CEF_REQUIRE_UI_THREAD();

  if (dev_renderer_)
    dev_renderer_->OnLoadingStateChange(browser, isLoading, canGoBack,
                                        canGoForward);
}

void AgentHandler::CloseAllBrowsers(bool force_close) {
  if (!CefCurrentlyOn(TID_UI)) {
    // Execute on the UI thread.
    CefPostTask(TID_UI, base::BindOnce(&AgentHandler::CloseAllBrowsers, this,
                                       force_close));
    return;
  }

  if (browser_store_->browser_handles_.empty()) {
    return;
  }

  for (const auto& pair : browser_store_->browser_handles_) {
    pair.second->browser_->GetHost()->CloseBrowser(force_close);
  }
}

#if !defined(OS_MAC)
void AgentHandler::PlatformShowWindow(CefRefPtr<CefBrowser> browser) {
  NOTIMPLEMENTED();
}
#endif
