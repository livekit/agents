#include "handler.hpp"

#include <iostream>

#include "include/base/cef_callback.h"
#include "include/cef_app.h"
#include "include/cef_parser.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_closure_task.h"
#include "include/wrapper/cef_helpers.h"

namespace {

// Returns a data: URI with the specified contents.
std::string GetDataURI(const std::string& data, const std::string& mime_type) {
  return "data:" + mime_type + ";base64," +
         CefURIEncode(CefBase64Encode(data.data(), data.size()), false)
             .ToString();
}

}  // namespace

AgentHandler* g_instance = nullptr;

AgentHandler::AgentHandler(CefRefPtr<DevRenderer> dev_renderer)
    : dev_renderer_(std::move(dev_renderer)) {
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
  if (dev_renderer_)
    dev_renderer_->OnPaint(browser, type, dirtyRects, buffer, width, height);
}

void AgentHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) {
  CEF_REQUIRE_UI_THREAD();

  int identifier = browser->GetIdentifier();
  CefRefPtr<BrowserHandle>& handle = browser_handles_[identifier];
  rect.Set(0, 0, handle->GetWidth(), handle->GetHeight());
};

void AgentHandler::OnAudioStreamPacket(CefRefPtr<CefBrowser> browser,
                                       const float** data,
                                       int frames,
                                       int64_t pts) {
  std::cout << "OnAudioStreamPacket" << std::endl;
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

  int identifier = browser->GetIdentifier();
  CefRefPtr<BrowserHandle> handle = pending_handles_.front();
  pending_handles_.pop_front();

  handle->browser_ = browser;
  if (handle->created_callback_)
    handle->created_callback_();

  browser_handles_[identifier] = handle;

  if (dev_renderer_)
    dev_renderer_->OnAfterCreated(browser);
}

bool AgentHandler::DoClose(CefRefPtr<CefBrowser> browser) {
  CEF_REQUIRE_UI_THREAD();

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

void AgentHandler::OnLoadError(CefRefPtr<CefBrowser> browser,
                               CefRefPtr<CefFrame> frame,
                               ErrorCode errorCode,
                               const CefString& errorText,
                               const CefString& failedUrl) {
  CEF_REQUIRE_UI_THREAD();

  // Allow Chrome to show the error page.
  if (IsChromeRuntimeEnabled()) {
    return;
  }

  // Don't display an error for downloaded files.
  if (errorCode == ERR_ABORTED) {
    return;
  }

  // Display a load error message using a data: URI.
  std::stringstream ss;
  ss << "<html><body bgcolor=\"white\">"
        "<h2>Failed to load URL "
     << std::string(failedUrl) << " with error " << std::string(errorText)
     << " (" << errorCode << ").</h2></body></html>";

  frame->LoadURL(GetDataURI(ss.str(), "text/html"));
}

/*
void AgentHandler::CloseAllBrowsers(bool force_close) {
  if (!CefCurrentlyOn(TID_UI)) {
    // Execute on the UI thread.
    CefPostTask(TID_UI, base::BindOnce(&AgentHandler::CloseAllBrowsers, this,
                                       force_close));
    return;
  }

  if (browser_list_.empty()) {
    return;
  }

  BrowserList::const_iterator it = browser_list_.begin();
  for (; it != browser_list_.end(); ++it) {
    (*it)->GetHost()->CloseBrowser(force_close);
  }
}
 */

bool AgentHandler::IsChromeRuntimeEnabled() {
  static bool enabled = []() {
    return CefCommandLine::GetGlobalCommandLine()->HasSwitch(
        "enable-chrome-runtime");
  }();
  return enabled;
}

#if !defined(OS_MAC)
void AgentHandler::PlatformShowWindow(CefRefPtr<CefBrowser> browser) {
  NOTIMPLEMENTED();
}
#endif
